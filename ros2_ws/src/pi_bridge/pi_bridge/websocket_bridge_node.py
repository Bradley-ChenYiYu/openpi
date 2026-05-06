import base64
import concurrent.futures
from collections import deque
import json
import os
from pathlib import Path
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import String


def _import_openpi_client_policy():
    try:
        from openpi_client import websocket_client_policy as client_policy

        return client_policy
    except ModuleNotFoundError:
        candidate_roots = [
            Path(os.environ.get("OPENPI_ROOT", "")),
            Path("/openpi"),
            Path.cwd(),
        ]
        for root in candidate_roots:
            if not root:
                continue
            candidate = root / "packages" / "openpi-client" / "src"
            if candidate.is_dir():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.append(candidate_str)
                from openpi_client import websocket_client_policy as client_policy

                return client_policy
        raise


_websocket_client_policy = _import_openpi_client_policy()


@dataclass
class BufferedImage:
    stamp_ns: int
    frame_id: str
    image_array: np.ndarray


@dataclass
class BufferedOdometry:
    stamp_ns: int
    linear_x: float
    angular_z: float


class WebsocketClientManager:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key: str | None,
        reconnect_interval_sec: float,
        max_reconnect_interval_sec: float,
        logger,
    ) -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._reconnect_interval_sec = reconnect_interval_sec
        self._max_reconnect_interval_sec = max_reconnect_interval_sec
        self._logger = logger

        self._stop_event = threading.Event()
        self._disconnect_event = threading.Event()
        self._client_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._client: _websocket_client_policy.WebsocketClientPolicy | None = None

        self._on_connected = None
        self._on_metadata = None
        self._on_disconnected = None
        self._on_error = None

    def set_callbacks(self, *, on_connected, on_metadata, on_disconnected, on_error) -> None:
        self._on_connected = on_connected
        self._on_metadata = on_metadata
        self._on_disconnected = on_disconnected
        self._on_error = on_error

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ws-manager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._disconnect_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def connected(self) -> bool:
        with self._client_lock:
            return self._client is not None and not self._disconnect_event.is_set()

    def infer(self, request: dict[str, Any]) -> dict[str, Any] | None:
        with self._client_lock:
            client = self._client

        if client is None:
            return None

        try:
            return client.infer(request)
        except Exception as exc:
            if self._on_error is not None:
                self._on_error(f"Infer request failed: {exc}")
            self._disconnect_event.set()
            return None

    def _run(self) -> None:
        backoff = self._reconnect_interval_sec
        while not self._stop_event.is_set():
            self._disconnect_event.clear()

            try:
                self._logger.info(f"Connecting websocket: {self._host}:{self._port}")
                client = _websocket_client_policy.WebsocketClientPolicy(
                    host=self._host,
                    port=self._port,
                    api_key=self._api_key,
                )
                with self._client_lock:
                    self._client = client

                metadata = client.get_server_metadata()
                backoff = self._reconnect_interval_sec

                if self._on_connected is not None:
                    self._on_connected()

                if self._on_metadata is not None:
                    self._on_metadata(metadata)

                while not self._stop_event.is_set() and not self._disconnect_event.is_set():
                    time.sleep(0.05)

            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(f"Websocket connect/loop error: {exc}")
            finally:
                with self._client_lock:
                    self._client = None

                if self._on_disconnected is not None:
                    self._on_disconnected()

            if not self._stop_event.is_set():
                self._logger.warning(f"Websocket disconnected. Reconnecting in {backoff:.2f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self._max_reconnect_interval_sec)


class PiWebsocketBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("pi_websocket_bridge")

        self.declare_parameter("websocket_host", "127.0.0.1")
        self.declare_parameter("websocket_port", 8000)
        self.declare_parameter("api_key", "")
        self.declare_parameter("robot_config", "tracer")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("front_image_topic", "/camera/front/color/image_raw")
        self.declare_parameter("left_image_topic", "/camera/left/color/image_raw")
        self.declare_parameter("right_image_topic", "/camera/right/color/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("image_topic_type", "image")
        self.declare_parameter("inferred_cmd_topic", "/pi_bridge/inferred_cmd_vel")
        self.declare_parameter("ack_topic", "/pi_bridge/control_ack")
        self.declare_parameter("raw_response_topic", "/pi_bridge/raw_response")
        self.declare_parameter("diagnostics_topic", "/pi_bridge/diagnostics")
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("send_rate_hz", 10.0)
        self.declare_parameter("reconnect_interval_sec", 1.0)
        self.declare_parameter("max_reconnect_interval_sec", 8.0)
        self.declare_parameter("sync_tolerance_sec", 0.25)
        self.declare_parameter("prompt", "do something")
        self.declare_parameter("action_rate_hz", 20.0)

        self._websocket_host = str(self.get_parameter("websocket_host").value)
        self._websocket_port = int(self.get_parameter("websocket_port").value)
        api_key = str(self.get_parameter("api_key").value)
        self._api_key = api_key if api_key else None
        self._robot_config = str(self.get_parameter("robot_config").value)
        self._queue_size = int(self.get_parameter("queue_size").value)
        self._send_rate_hz = float(self.get_parameter("send_rate_hz").value)
        self._sync_tolerance_sec = float(self.get_parameter("sync_tolerance_sec").value)
        self._prompt = self.get_parameter("prompt").value
        self._action_rate_hz = float(self.get_parameter("action_rate_hz").value)

        self._latest_images: dict[str, BufferedImage] = {}
        self._latest_odom: BufferedOdometry | None = None
        self._lock = threading.Lock()

        self._total_sent = 0
        self._total_responses = 0
        self._dropped_frames = 0
        self._parse_failures = 0
        self._last_latency_ms = 0.0
        self._first_odom_logged = False
        self._first_images_logged: set[str] = set()
        self._last_waiting_data_log_ns = 0
        self._last_sync_drop_log_ns = 0
        self._last_missing_actions_log_ns = 0
        self._start_time_ns = self.get_clock().now().nanoseconds
        self._infer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-infer")
        self._infer_future: concurrent.futures.Future | None = None
        self._infer_sent_ns: int = 0
        self._pending_actions: deque[tuple[float, float]] = deque()
        self._action_lock = threading.Lock()
        self._last_action: tuple[float, float] | None = None

        image_topic = self.get_parameter("image_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        image_topic_type = self.get_parameter("image_topic_type").value.lower().strip()

        self._odom_sub = self.create_subscription(Odometry, odom_topic, self._on_odom, self._queue_size)
        
        # Robot configuration mapping: robot_config -> {observation_key: parameter_name}
        config_map = {
            "tracer": {"observation/image": "image_topic"},
            "tracer_front_right": {
                "observation/front_image": "front_image_topic",
                "observation/right_image": "right_image_topic",
            },
            "tracer_front_left": {
                "observation/front_image": "front_image_topic",
                "observation/left_image_topic": "left_image_topic",
            },
            "tracer_side": {
                "observation/front_image": "front_image_topic",
                "observation/left_image": "left_image_topic",
                "observation/right_image": "right_image_topic",
            },
        }
        
        self._required_images = config_map.get(self._robot_config, config_map["tracer"])
        
        for obs_key, param_name in self._required_images.items():
            topic = self.get_parameter(param_name).value
            if image_topic_type == "compressed":
                self.create_subscription(
                    CompressedImage, topic, lambda msg, k=obs_key: self._on_compressed_image(msg, k), self._queue_size
                )
            else:
                self.create_subscription(
                    Image, topic, lambda msg, k=obs_key: self._on_image(msg, k), self._queue_size
                )

        inferred_cmd_topic = self.get_parameter("inferred_cmd_topic").value
        ack_topic = self.get_parameter("ack_topic").value
        raw_response_topic = self.get_parameter("raw_response_topic").value
        diagnostics_topic = self.get_parameter("diagnostics_topic").value

        inferred_cmd_qos = QoSProfile(
            history=HistoryPolicy.KEEP_ALL,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._inferred_cmd_pub = self.create_publisher(Twist, inferred_cmd_topic, inferred_cmd_qos)
        self._ack_pub = self.create_publisher(String, ack_topic, self._queue_size)
        self._raw_pub = self.create_publisher(String, raw_response_topic, self._queue_size)
        self._diag_pub = self.create_publisher(String, diagnostics_topic, self._queue_size)

        self._manager = WebsocketClientManager(
            host=self._websocket_host,
            port=self._websocket_port,
            api_key=self._api_key,
            reconnect_interval_sec=float(self.get_parameter("reconnect_interval_sec").value),
            max_reconnect_interval_sec=float(self.get_parameter("max_reconnect_interval_sec").value),
            logger=self.get_logger(),
        )
        self._manager.set_callbacks(
            on_connected=self._on_connected,
            on_metadata=self._on_metadata,
            on_disconnected=self._on_disconnected,
            on_error=self._on_error,
        )
        self._manager.start()

        send_period = 1.0 / max(self._send_rate_hz, 1e-3)
        action_period = 1.0 / max(self._action_rate_hz, 1e-3)
        self._send_timer = self.create_timer(send_period, self._on_send_timer)
        self._cmd_timer = self.create_timer(action_period, self._on_cmd_timer)
        self._diag_timer = self.create_timer(1.0, self._publish_diagnostics)

        self.get_logger().info(
            "Bridge config: "
            f"image_topic={image_topic}, image_topic_type={image_topic_type}, "
            f"odom_topic={odom_topic}, inferred_cmd_topic={inferred_cmd_topic}, "
            f"send_rate_hz={self._send_rate_hz}, action_rate_hz={self._action_rate_hz}, "
            f"sync_tolerance_sec={self._sync_tolerance_sec}"
        )

    def destroy_node(self) -> bool:
        self._manager.stop()
        self._infer_executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _on_connected(self) -> None:
        self.get_logger().info("Websocket connected")

    def _on_metadata(self, metadata: dict[str, Any]) -> None:
        self.get_logger().info(f"Server metadata received: keys={sorted(metadata.keys())}")

    def _on_disconnected(self) -> None:
        self.get_logger().warning("Websocket disconnected")

    def _on_error(self, message: str) -> None:
        self._parse_failures += 1
        self.get_logger().error(message)

    def _on_odom(self, msg: Odometry) -> None:
        stamp_ns = self._stamp_to_ns(msg.header.stamp)
        with self._lock:
            self._latest_odom = BufferedOdometry(
                stamp_ns=stamp_ns,
                linear_x=float(msg.twist.twist.linear.x),
                angular_z=float(msg.twist.twist.angular.z),
            )
        if not self._first_odom_logged:
            self._first_odom_logged = True
            self.get_logger().info(
                f"First odom received: stamp_ns={stamp_ns}, "
                f"linear_x={msg.twist.twist.linear.x:.4f}, angular_z={msg.twist.twist.angular.z:.4f}"
            )

    def _on_image(self, msg: Image, key: str) -> None:
        try:
            image_array = self._image_msg_to_array(msg)
            stamp_ns = self._stamp_to_ns(msg.header.stamp)
            with self._lock:
                self._latest_images[key] = BufferedImage(
                    stamp_ns=stamp_ns,
                    frame_id=msg.header.frame_id,
                    image_array=image_array,
                )
            if key not in self._first_images_logged:
                self._first_images_logged.add(key)
                self.get_logger().info(
                    f"First image received for {key}: stamp_ns={stamp_ns}, frame_id={msg.header.frame_id}, "
                    f"shape={image_array.shape}, encoding={msg.encoding}"
                )
        except Exception as exc:
            self._parse_failures += 1
            self.get_logger().warning(f"Failed to parse Image message for {key}: {exc}")

    def _on_compressed_image(self, msg: CompressedImage, key: str) -> None:
        try:
            stamp_ns = self._stamp_to_ns(msg.header.stamp)
            # For compressed streams, we cannot reliably decode without extra deps.
            # Keep inference image as a placeholder until decode support is added.
            image_array = np.zeros((224, 224, 3), dtype=np.uint8)
            with self._lock:
                self._latest_images[key] = BufferedImage(
                    stamp_ns=stamp_ns,
                    frame_id=msg.header.frame_id,
                    image_array=image_array,
                )
            if key not in self._first_images_logged:
                self._first_images_logged.add(key)
                self.get_logger().info(
                    f"First compressed image placeholder received for {key}: stamp_ns={stamp_ns}, "
                    f"frame_id={msg.header.frame_id}, shape={image_array.shape}"
                )
        except Exception as exc:
            self._parse_failures += 1
            self.get_logger().warning(f"Failed to parse CompressedImage message for {key}: {exc}")

    def _image_msg_to_array(self, msg: Image) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)

        if height <= 0 or width <= 0 or step <= 0:
            raise ValueError("Invalid image dimensions")

        data = np.frombuffer(msg.data, dtype=np.uint8)
        if data.size != height * step:
            raise ValueError("Image data length does not match height*step")

        row_major = data.reshape((height, step))
        encoding = (msg.encoding or "").lower().strip()

        if encoding in ("rgb8", "bgr8"):
            channels = 3
            reshaped = row_major[:, : width * channels].reshape((height, width, channels))
            if encoding == "bgr8":
                reshaped = reshaped[:, :, ::-1]
            return np.ascontiguousarray(reshaped)

        if encoding in ("rgba8", "bgra8"):
            channels = 4
            reshaped = row_major[:, : width * channels].reshape((height, width, channels))
            if encoding == "bgra8":
                reshaped = reshaped[:, :, [2, 1, 0, 3]]
            return np.ascontiguousarray(reshaped[:, :, :3])

        if encoding in ("mono8", "8uc1"):
            mono = row_major[:, :width].reshape((height, width))
            return np.stack([mono, mono, mono], axis=-1)

        raise ValueError(f"Unsupported Image encoding: {msg.encoding}")

    def _on_send_timer(self) -> None:
        self._drain_infer_result()

        if not self._manager.connected:
            return

        with self._lock:
            images = self._latest_images
            odom = self._latest_odom

        if odom is None or any(key not in images for key in self._required_images):
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_waiting_data_log_ns > 2_000_000_000:
                self._last_waiting_data_log_ns = now_ns
                uptime_sec = (now_ns - self._start_time_ns) / 1e9
                if uptime_sec > 3.0:
                    self.get_logger().warning(
                        f"Waiting for synced inputs before infer: have_odom={odom is not None}, "
                        f"have_images={list(images.keys())}, uptime_sec={uptime_sec:.1f}. "
                        "Verify image topics and robot_config match the actual camera setup."
                    )
                else:
                    self.get_logger().info(
                        f"Waiting for synced inputs before infer: have_odom={odom is not None}, "
                        f"have_images={list(images.keys())}"
                    )
            return

        # Sync check: ensure all required images are reasonably close in time to odom
        for key in self._required_images:
            img = images[key]
            sync_diff_sec = abs(img.stamp_ns - odom.stamp_ns) / 1e9
            if sync_diff_sec > self._sync_tolerance_sec:
                self._dropped_frames += 1
                now_ns = self.get_clock().now().nanoseconds
                if now_ns - self._last_sync_drop_log_ns > 2_000_000_000:
                    self._last_sync_drop_log_ns = now_ns
                    self.get_logger().warning(
                        f"Dropping frame due to timestamp mismatch for {key}: diff_sec={sync_diff_sec:.4f}, "
                        f"image_stamp_ns={img.stamp_ns}, odom_stamp_ns={odom.stamp_ns}, "
                        f"tolerance_sec={self._sync_tolerance_sec}"
                    )
                return

        request = {
            "observation/state": np.asarray([odom.linear_x, odom.angular_z], dtype=np.float32),
            "prompt": self._prompt,
        }
        for key in self._required_images:
            request[key] = images[key].image_array

        if self._infer_future is not None and not self._infer_future.done():
            self._dropped_frames += 1
            return

        self._infer_sent_ns = self.get_clock().now().nanoseconds
        self._infer_future = self._infer_executor.submit(self._manager.infer, request)
        self._total_sent += 1

    def _drain_infer_result(self) -> None:
        if self._infer_future is None or not self._infer_future.done():
            return

        future = self._infer_future
        self._infer_future = None

        try:
            response = future.result()
        except Exception as exc:
            self._parse_failures += 1
            self.get_logger().warning(f"Infer future failed: {exc}")
            return

        if response is None:
            self._dropped_frames += 1
            return

        recv_ns = self.get_clock().now().nanoseconds
        self._last_latency_ms = (recv_ns - self._infer_sent_ns) / 1e6
        self._total_responses += 1
        response_keys = [str(k) for k in response.keys()]
        self.get_logger().debug(
            f"Received inference response: latency_ms={self._last_latency_ms:.2f}, "
            f"keys={sorted(response_keys)}"
        )

        self._enqueue_inferred_actions(response)
        self._publish_ack(response)
        self._publish_raw_response(response)

    def _enqueue_inferred_actions(self, response: dict[str, Any]) -> None:
        actions = self._response_get(response, "actions")
        if actions is None:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_missing_actions_log_ns > 2_000_000_000:
                self._last_missing_actions_log_ns = now_ns
                self.get_logger().warning(
                    f"Inference response has no 'actions' key. Available keys={sorted(str(k) for k in response.keys())}"
                )
            return

        try:
            arr = np.asarray(actions)
            queued_actions: list[tuple[float, float]] = []

            if arr.ndim == 2 and arr.shape[1] >= 2:
                for row in arr:
                    queued_actions.append((float(row[0]), float(row[1])))
            elif arr.ndim == 1 and arr.shape[0] >= 2:
                queued_actions.append((float(arr[0]), float(arr[1])))
            else:
                self.get_logger().warning(f"Unsupported actions shape for cmd publish: shape={arr.shape}, ndim={arr.ndim}")
                return

            if not queued_actions:
                return

            # Replace stale queued commands with the latest model rollout.
            with self._action_lock:
                self._pending_actions.clear()
                self._pending_actions.extend(queued_actions)

        except Exception as exc:
            self._parse_failures += 1
            self.get_logger().warning(f"Failed to queue inferred cmd: {exc}")

    def _on_cmd_timer(self) -> None:
        action: tuple[float, float] | None = None
        with self._action_lock:
            if self._pending_actions:
                action = self._pending_actions.popleft()
                self._last_action = action
            else:
                action = self._last_action

        if action is None:
            return

        msg = Twist()
        msg.linear.x = action[0]
        msg.angular.z = action[1]
        self._inferred_cmd_pub.publish(msg)
        self.get_logger().debug(
            f"Published inferred cmd: linear_x={action[0]:.4f}, angular_z={action[1]:.4f}"
        )

    def _publish_ack(self, response: dict[str, Any]) -> None:
        payload = {
            "ok": True,
            "latency_ms": self._last_latency_ms,
            "server_timing": self._response_get(response, "server_timing", {}),
            "policy_timing": self._response_get(response, "policy_timing", {}),
        }
        msg = String()
        msg.data = json.dumps(payload, default=str)
        self._ack_pub.publish(msg)

    def _publish_raw_response(self, response: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(self._jsonable(response), default=str)
        self._raw_pub.publish(msg)

    def _publish_diagnostics(self) -> None:
        with self._action_lock:
            pending_actions = len(self._pending_actions)

        diag = {
            "connected": self._manager.connected,
            "sent": self._total_sent,
            "responses": self._total_responses,
            "dropped_frames": self._dropped_frames,
            "parse_failures": self._parse_failures,
            "latency_ms": round(self._last_latency_ms, 2),
            "pending_actions": pending_actions,
        }
        msg = String()
        msg.data = json.dumps(diag)
        self._diag_pub.publish(msg)

    def _stamp_to_ns(self, stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _response_get(self, response: dict[str, Any], key: str, default: Any = None) -> Any:
        #TODO: The response seems to be in bytes, we can remove the first "if" after verified
        if key in response:
            return response[key]
        key_bytes = key.encode("utf-8")
        if key_bytes in response:
            return response[key_bytes]
        return default

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        if isinstance(value, dict):
            out = {}
            for key, sub_value in value.items():
                if isinstance(key, bytes):
                    key = key.decode("utf-8", errors="replace")
                else:
                    key = str(key)
                out[key] = self._jsonable(sub_value)
            return out
        if isinstance(value, list):
            return [self._jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [self._jsonable(v) for v in value]
        return value


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PiWebsocketBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
