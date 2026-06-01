"""Task controller for pausing and resuming the bridge request-action flow."""

from __future__ import annotations

from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node


class TaskFlowController(Node):
    """Advance bridge tasks when the commanded twist remains near zero."""

    def __init__(self) -> None:
        super().__init__("pi_bridge_task_flow_controller")

        self.declare_parameter("bridge_node_name", "pi_websocket_bridge")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("tasks", ["task_0"])
        self.declare_parameter("tasks_config_path", "config/task_flow_controller.yaml")
        self.declare_parameter("stop_linear_x_threshold", 0.01)
        self.declare_parameter("stop_angular_z_threshold", 0.05)
        self.declare_parameter("stop_consecutive_count", 10)
        self.declare_parameter("parameter_retry_rate_hz", 5.0)

        self._bridge_node_name = str(self.get_parameter("bridge_node_name").value).lstrip("/")
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._tasks_config_path = str(self.get_parameter("tasks_config_path").value)
        self._tasks = self._load_tasks()
        self._stop_linear_x_threshold = float(self.get_parameter("stop_linear_x_threshold").value)
        self._stop_angular_z_threshold = float(self.get_parameter("stop_angular_z_threshold").value)
        self._stop_consecutive_count = max(1, int(self.get_parameter("stop_consecutive_count").value))
        self._parameter_retry_period_sec = 1.0 / max(
            float(self.get_parameter("parameter_retry_rate_hz").value),
            1e-3,
        )

        self._parameter_client = self.create_client(
            SetParameters,
            f"/{self._bridge_node_name}/set_parameters",
        )
        self._cmd_vel_subscription = self.create_subscription(
            Twist,
            self._cmd_vel_topic,
            self._on_cmd_vel,
            10,
        )
        self._timer = self.create_timer(self._parameter_retry_period_sec, self._on_timer)

        self._task_index = 0
        self._next_task_index: int | None = None
        self._low_twist_count = 0
        self._stage = "idle"
        self._pending_request_future = None
        self._service_unavailable_logged = False

        if not self._tasks:
            self.get_logger().warning("No tasks configured; controller will stay idle")
            return

        self._stage = "start"
        self.get_logger().info(
            f"Controlling bridge node '{self._bridge_node_name}' on topic '{self._cmd_vel_topic}'"
        )

    def _load_tasks(self) -> list[str]:
        config_path = Path(self._tasks_config_path)
        if not config_path.is_absolute():
            config_path = Path(get_package_share_directory("pi_bridge")) / config_path

        if config_path.is_file():
            loaded_tasks = self._parse_tasks_config(config_path.read_text(encoding="utf-8"))
            if loaded_tasks:
                self.get_logger().info(
                    f"Loaded {len(loaded_tasks)} tasks from {config_path}"
                )
                return loaded_tasks

            self.get_logger().warning(
                f"Task config {config_path} did not contain a non-empty tasks list; using parameter fallback"
            )

        fallback_tasks = [str(task) for task in self.get_parameter("tasks").value if str(task).strip()]
        if fallback_tasks:
            self.get_logger().info("Loaded tasks from parameter fallback")
        return fallback_tasks

    def _parse_tasks_config(self, content: str) -> list[str]:
        tasks: list[str] = []
        in_tasks_block = False
        tasks_block_indent: int | None = None

        for raw_line in content.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue

            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if stripped.startswith("tasks:"):
                in_tasks_block = True
                tasks_block_indent = indent
                inline_list = stripped[len("tasks:"):].strip()
                if inline_list.startswith("[") and inline_list.endswith("]"):
                    items = inline_list[1:-1].split(",")
                    tasks.extend(self._normalize_task_item(item) for item in items)
                    break
                continue

            if in_tasks_block:
                if tasks_block_indent is not None and indent <= tasks_block_indent:
                    break
                if stripped.startswith("- "):
                    tasks.append(self._normalize_task_item(stripped[2:]))
                    continue

            if stripped.startswith("-"):
                tasks.append(self._normalize_task_item(stripped[1:]))

        return [task for task in tasks if task]

    def _normalize_task_item(self, value: str) -> str:
        return value.strip().strip('"').strip("'")

    def _on_cmd_vel(self, msg: Twist) -> None:
        if self._stage != "monitor" or self._pending_request_future is not None:
            return

        if self._is_stop(msg):
            self._low_twist_count += 1
            if self._low_twist_count < self._stop_consecutive_count:
                return

            self._low_twist_count = 0
            if self._task_index >= len(self._tasks) - 1:
                self._stage = "complete"
                self._submit_bridge_update(enable_request_action_flow=False)
                self.get_logger().info("Final task complete; paused request-action flow")
                return

            self._next_task_index = self._task_index + 1
            self._stage = "pause"
            self._submit_bridge_update(enable_request_action_flow=False)
            self.get_logger().info(
                f"Detected stop condition for task {self._task_index}; pausing request-action flow"
            )
            return

        self._low_twist_count = 0

    def _on_timer(self) -> None:
        self._drain_pending_request()

        if self._pending_request_future is not None:
            return

        if not self._parameter_client.service_is_ready():
            if not self._service_unavailable_logged:
                self.get_logger().info("Waiting for bridge parameter service")
                self._service_unavailable_logged = True
            return

        self._service_unavailable_logged = False

        if self._stage == "start":
            if self._submit_bridge_update(
                prompt=self._tasks[self._task_index],
                enable_request_action_flow=True,
            ):
                self.get_logger().info(
                    f"Starting task {self._task_index}: {self._tasks[self._task_index]}"
                )
        elif self._stage == "resume" and self._next_task_index is not None:
            next_task_index = self._next_task_index
            if self._submit_bridge_update(
                prompt=self._tasks[next_task_index],
                enable_request_action_flow=True,
            ):
                self.get_logger().info(
                    f"Resuming task {next_task_index}: {self._tasks[next_task_index]}"
                )

    def _drain_pending_request(self) -> None:
        if self._pending_request_future is None or not self._pending_request_future.done():
            return

        future = self._pending_request_future
        self._pending_request_future = None

        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to update bridge parameters: {exc}")
            if self._stage == "pause":
                self._stage = "monitor"
            elif self._stage == "resume":
                self._stage = "pause"
            elif self._stage == "complete":
                self._stage = "monitor"
            return

        if not all(result.successful for result in response.results):
            self.get_logger().error("Bridge parameter update was rejected")
            if self._stage == "pause":
                self._stage = "monitor"
            elif self._stage == "resume":
                self._stage = "pause"
            elif self._stage == "complete":
                self._stage = "monitor"
            return

        if self._stage == "pause":
            self._stage = "resume"
        elif self._stage == "start":
            self._stage = "monitor"
            self._low_twist_count = 0
            self.get_logger().info(
                f"Started task {self._task_index}: {self._tasks[self._task_index]}"
            )
        elif self._stage == "resume" and self._next_task_index is not None:
            self._task_index = self._next_task_index
            self._next_task_index = None
            self._stage = "monitor"
            self._low_twist_count = 0
            self.get_logger().info(
                f"Resumed task {self._task_index}: {self._tasks[self._task_index]}"
            )
        elif self._stage == "complete":
            self.get_logger().info("All configured tasks have been traversed")

    def _submit_bridge_update(
        self,
        *,
        prompt: str | None = None,
        enable_request_action_flow: bool | None = None,
    ) -> bool:
        if self._pending_request_future is not None:
            return False

        request = SetParameters.Request()
        parameters = []

        if prompt is not None:
            parameters.append(self._make_string_parameter("prompt", prompt))
        if enable_request_action_flow is not None:
            parameters.append(
                self._make_bool_parameter(
                    "enable_request_action_flow",
                    enable_request_action_flow,
                )
            )

        if not parameters:
            return True

        request.parameters = parameters
        self._pending_request_future = self._parameter_client.call_async(request)
        return True

    def _is_stop(self, msg: Twist) -> bool:
        return (
            abs(float(msg.linear.x)) <= self._stop_linear_x_threshold
            and abs(float(msg.angular.z)) <= self._stop_angular_z_threshold
        )

    def _make_string_parameter(self, name: str, value: str) -> ParameterMsg:
        parameter = ParameterMsg()
        parameter.name = name
        parameter.value = ParameterValue()
        parameter.value.type = ParameterType.PARAMETER_STRING
        parameter.value.string_value = value
        return parameter

    def _make_bool_parameter(self, name: str, value: bool) -> ParameterMsg:
        parameter = ParameterMsg()
        parameter.name = name
        parameter.value = ParameterValue()
        parameter.value.type = ParameterType.PARAMETER_BOOL
        parameter.value.bool_value = value
        return parameter


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TaskFlowController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()