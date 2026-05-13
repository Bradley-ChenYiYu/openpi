# Refactoring Plan: Separate Robot Bridge Classes

## Objective
Separate general bridge utilities (Websocket, synchronization, data processing) from robot-specific configurations (ROS2 topics, observation keys, action mappings) using class inheritance.

## Architecture

### 1. Base Class: `PiWebsocketBridgeBase`
Handles the "plumbing" and is agnostic to the specific robot.
- **Websocket Management**: Integration with `WebsocketClientManager`.
- **The Bridge Loop**: `_on_send_timer` and `_drain_infer_result`.
- **Data Synchronization**: Timestamp checks and ensuring all required images are present.
- **General Utilities**: `_image_msg_to_array`, `_stamp_to_ns`, `_jsonable`, and `_response_get`.
- **Diagnostics**: `_publish_diagnostics`.
- **Abstract Hooks**:
    - `setup_robot_config()`
    - `setup_subscriptions()`
    - `setup_publishers()`
    - `get_required_observations()`
    - `prepare_observation_state()`
    - `apply_action()`

### 2. Derived Robot Classes
Define the "mapping" for each specific robot configuration.
- `TracerBridge`
- `TracerSideBridge`
- `TracerFrontLeftBridge`
- `TracerFrontRightBridge`

## Implementation

```python
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
from typing import Any, Dict, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

# --- General Utilities (Agnostic to Robot) ---

def _import_openpi_client_policy():
    try:
        from openpi_client import websocket_client_policy as client_policy
        return client_policy
    except ModuleNotFoundError:
        candidate_roots = [Path(os.environ.get("OPENPI_ROOT", "")), Path("/openpi"), Path.cwd()]
        for root in candidate_roots:
            if not root: continue
            candidate = root / "packages" / "openpi-client" / "src"
            if candidate.is_dir():
                candidate_str = str(candidate)
                if candidate_str not in sys.path: sys.path.append(candidate_str)
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
    def __init__(self, *, host, port, api_key, reconnect_interval_sec, max_reconnect_interval_sec, logger):
        self._host, self._port, self._api_key = host, port, api_key
        self._reconnect_interval_sec, self._max_reconnect_interval_sec = reconnect_interval_sec, max_reconnect_interval_sec
        self._logger = logger
        self._stop_event, self._disconnect_event = threading.Event(), threading.Event()
        self._client_lock = threading.Lock()
        self._thread, self._client = None, None
        self._on_connected = self._on_metadata = self._on_disconnected = self._on_error = None

    def set_callbacks(self, *, on_connected, on_metadata, on_disconnected, on_error):
        self._on_connected, self._on_metadata, self._on_disconnected, self._on_error = on_connected, on_metadata, on_disconnected, on_error

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ws-manager", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set(); self._disconnect_event.set()
        if self._thread: self._thread.join(timeout=5.0)

    @property
    def connected(self):
        with self._client_lock: return self._client is not None and not self._disconnect_event.is_set()

    def infer(self, request):
        with self._client_lock: client = self._client
        if client is None: return None
        try: return client.infer(request)
        except Exception as exc:
            if self._on_error: self._on_error(f"Infer request failed: {exc}")
            self._disconnect_event.set(); return None

    def _run(self):
        backoff = self._reconnect_interval_sec
        while not self._stop_event.is_set():
            self._disconnect_event.clear()
            try:
                self._logger.info(f"Connecting websocket: {self._host}:{self._port}")
                client = _websocket_client_policy.WebsocketClientPolicy(host=self._host, port=self._port, api_key=self._api_key)
                with self._client_lock: self._client = client
                metadata = client.get_server_metadata()
                backoff = self._reconnect_interval_sec
                if self._on_connected: self._on_connected()
                if self._on_metadata: self._on_metadata(metadata)
                while not self._stop_event.is_set() and not self._disconnect_event.is_set(): time.sleep(0.05)
            except Exception as exc:
                if self._on_error: self._on_error(f"Websocket connect/loop error: {exc}")
            finally:
                with self._client_lock: self._client = None
                if self._on_disconnected: self._on_disconnected()
            if not self._stop_event.is_set():
                self._logger.warning(f"Websocket disconnected. Reconnecting in {backoff:.2f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, self._max_reconnect_interval_sec)

# --- Base Bridge Class ---

class PiWebsocketBridgeBase(Node):
    """Base class handling the 'plumbing' of the bridge."""
    def __init__(self) -> None:
        super().__init__("pi_websocket_bridge_base")
        
        # Common Parameters
        self.declare_parameter("websocket_host", "127.0.0.1")
        self.declare_parameter("websocket_port", 8000)
        self.declare_parameter("api_key", "")
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("send_rate_hz", 10.0)
        self.declare_parameter("reconnect_interval_sec", 1.0)
        self.declare_parameter("max_reconnect_interval_sec", 8.0)
        self.declare_parameter("sync_tolerance_sec", 0.25)
        self.declare_parameter("prompt", "do something")
        self.declare_parameter("action_rate_hz", 20.0)

        # Initialize common state
        self._websocket_host = str(self.get_parameter("websocket_host").value)
        self._websocket_port = int(self.get_parameter("websocket_port").value)
        api_key = str(self.get_parameter("api_key").value)
        self._api_key = api_key if api_key else None
        self._queue_size = int(self.get_parameter("queue_size").value)
        self._send_rate_hz = float(self.get_parameter("send_rate_hz").value)
        self._sync_tolerance_sec = float(self.get_parameter("sync_tolerance_sec").value)
        self._prompt = self.get_parameter("prompt").value
        self._action_rate_hz = float(self.get_parameter("action_rate_hz").value)

        self._latest_images: Dict[str, BufferedImage] = {}
        self._latest_odom: BufferedOdometry | None = None
        self._lock = threading.Lock()

        # Metrics
        self._total_sent = 0
        self._total_responses = 0
        self._dropped_frames = 0
        self._parse_failures = 0
        self._last_latency_ms = 0.0
        self._start_time_ns = self.get_clock().now().nanoseconds
        
        # Execution
        self._infer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-infer")
        self._infer_future: concurrent.futures.Future | None = None
        self._infer_sent_ns: int = 0
        self._pending_actions: deque = deque()
        self._action_lock = threading.Lock()
        self._last_action = None

        # Robot-specific setup (implemented by children)
        self.setup_robot_config()
        self.setup_subscriptions()
        self.setup_publishers()

        # Websocket Manager
        self._manager = WebsocketClientManager(
            host=self._websocket_host, port=self._websocket_port, api_key=self._api_key,
            reconnect_interval_sec=float(self.get_parameter("reconnect_interval_sec").value),
            max_reconnect_interval_sec=float(self.get_parameter("max_reconnect_interval_sec").value),
            logger=self.get_logger(),
        )
        self._manager.set_callbacks(
            on_connected=lambda: self.get_logger().info("Websocket connected"),
            on_metadata=lambda m: self.get_logger().info(f"Server metadata: {sorted(m.keys())}"),
            on_disconnected=lambda: self.get_logger().warning("Websocket disconnected"),
            on_error=self._on_ws_error,
        )
        self._manager.start()

        # Timers
        self.create_timer(1.0 / max(self._send_rate_hz, 1e-3), self._on_send_timer)
        self.create_timer(1.0 / max(self._action_rate_hz, 1e-3), self._on_cmd_timer)
        self.create_timer(1.0, self._publish_diagnostics)

    def _on_ws_error(self, message: str):
        self._parse_failures += 1
        self.get_logger().error(message)

    # --- Abstract Methods to be implemented by Robot Classes ---
    def setup_robot_config(self): raise NotImplementedError
    def setup_subscriptions(self): raise NotImplementedError
    def setup_publishers(self): raise NotImplementedError
    def get_required_observations(self) -> Dict[str, str]: raise NotImplementedError
    def prepare_observation_state(self, odom: BufferedOdometry) -> np.ndarray: raise NotImplementedError
    def apply_action(self, action: Tuple[float, float]): raise NotImplementedError

    # --- General Utilities ---
    def _on_odom(self, msg: Odometry):
        stamp_ns = self._stamp_to_ns(msg.header.stamp)
        with self._lock:
            self._latest_odom = BufferedOdometry(stamp_ns=stamp_ns, linear_x=float(msg.twist.twist.linear.x), angular_z=float(msg.twist.twist.angular.z))

    def _on_image(self, msg: Image, key: str):
        try:
            image_array = self._image_msg_to_array(msg)
            with self._lock:
                self._latest_images[key] = BufferedImage(stamp_ns=self._stamp_to_ns(msg.header.stamp), frame_id=msg.header.frame_id, image_array=image_array)
        except Exception as exc:
            self._parse_failures += 1
            self.get_logger().warning(f"Image parse fail {key}: {exc}")

    def _image_msg_to_array(self, msg: Image) -> np.ndarray:
        data = np.frombuffer(msg.data, dtype=np.uint8).reshape((int(msg.height), int(msg.step)))
        encoding = (msg.encoding or "").lower().strip()
        if encoding in ("rgb8", "bgr8"):
            reshaped = data[:, :int(msg.width)*3].reshape((int(msg.height), int(msg.width), 3))
            return np.ascontiguousarray(reshaped[:, :, ::-1] if encoding == "bgr8" else reshaped)
        return np.ascontiguousarray(data[:, :int(msg.width)*3].reshape((int(msg.height), int(msg.width), 3)))

    def _on_send_timer(self):
        self._drain_infer_result()
        if not self._manager.connected: return
        with self._lock:
            images, odom = self._latest_images, self._latest_odom
        
        required = self.get_required_observations()
        if odom is None or any(k not in images for k in required): return

        # Sync check
        for k in required:
            if abs(images[k].stamp_ns - odom.stamp_ns) / 1e9 > self._sync_tolerance_sec:
                self._dropped_frames += 1; return

        request = {
            "observation/state": self.prepare_observation_state(odom),
            "prompt": self._prompt,
        }
        for k in required: request[k] = images[k].image_array

        if self._infer_future and not self._infer_future.done():
            self._dropped_frames += 1; return

        self._infer_sent_ns = self.get_clock().now().nanoseconds
        self._infer_future = self._infer_executor.submit(self._manager.infer, request)
        self._total_sent += 1

    def _drain_infer_result(self):
        if not self._infer_future or not self._infer_future.done(): return
        future = self._infer_future; self._infer_future = None
        try:
            response = future.result()
            if response:
                self._last_latency_ms = (self.get_clock().now().nanoseconds - self._infer_sent_ns) / 1e6
                self._total_responses += 1
                self._enqueue_actions(response)
                self._publish_ack(response)
        except Exception as exc:
            self._parse_failures += 1; self.get_logger().warning(f"Infer failed: {exc}")

    def _enqueue_actions(self, response):
        actions = self._response_get(response, "actions")
        if actions is None: return
        try:
            arr = np.asarray(actions)
            queued = []
            if arr.ndim == 2 and arr.shape[1] >= 2:
                queued = [(float(r[0]), float(r[1])) for r in arr]
            elif arr.ndim == 1 and arr.shape[0] >= 2:
                queued = [(float(arr[0]), float(arr[1]))]
            
            with self._action_lock:
                self._pending_actions.clear()
                self._pending_actions.extend(queued)
        except Exception as exc:
            self._parse_failures += 1; self.get_logger().warning(f"Action queue fail: {exc}")

    def _on_cmd_timer(self):
        with self._action_lock:
            action = self._pending_actions.popleft() if self._pending_actions else self._last_action
        if action:
            self.apply_action(action)
            self._last_action = action

    def _publish_ack(self, response):
        msg = String(); msg.data = json.dumps({"ok": True, "latency_ms": self._last_latency_ms}, default=str)
        self._ack_pub.publish(msg)

    def _publish_diagnostics(self):
        diag = {"connected": self._manager.connected, "sent": self._total_sent, "latency_ms": round(self._last_latency_ms, 2)}
        msg = String(); msg.data = json.dumps(diag)
        self._diag_pub.publish(msg)

    def _stamp_to_ns(self, stamp): return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    def _response_get(self, response, key, default=None):
        if key in response: return response[key]
        kb = key.encode("utf-8")
        return response.get(kb, default)

    def destroy_node(self):
        self._manager.stop()
        self._infer_executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

# --- Robot Specific Implementations ---

class TracerBridge(PiWebsocketBridgeBase):
    def setup_robot_config(self):
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("inferred_cmd_topic", "/pi_bridge/inferred_cmd_vel")

    def get_required_observations(self):
        return {"observation/image": "image_topic"}

    def setup_subscriptions(self):
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, self._queue_size)
        self.create_subscription(Image, self.get_parameter("image_topic").value, lambda m: self._on_image(m, "observation/image"), self._queue_size)

    def setup_publishers(self):
        self._inferred_cmd_pub = self.create_publisher(Twist, self.get_parameter("inferred_cmd_topic").value, QoSProfile(history=HistoryPolicy.KEEP_ALL, depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self._ack_pub = self.create_publisher(String, "/pi_bridge/control_ack", self._queue_size)

    def prepare_observation_state(self, odom):
        return np.asarray([odom.linear_x, odom.angular_z], dtype=np.float32)

    def apply_action(self, action):
        msg = Twist(); msg.linear.x, msg.angular.z = action
        self._inferred_cmd_pub.publish(msg)

class TracerSideBridge(PiWebsocketBridgeBase):
    def setup_robot_config(self):
        self.declare_parameter("front_image_topic", "/camera/front/color/image_raw")
        self.declare_parameter("left_image_topic", "/camera/left/color/image_raw")
        self.declare_parameter("right_image_topic", "/camera/right/color/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("inferred_cmd_topic", "/pi_bridge/inferred_cmd_vel")

    def get_required_observations(self):
        return {
            "observation/front_image": "front_image_topic",
            "observation/left_image": "left_image_topic",
            "observation/right_image": "right_image_topic",
        }

    def setup_subscriptions(self):
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, self._queue_size)
        for key, param in self.get_required_observations().items():
            self.create_subscription(Image, self.get_parameter(param).value, lambda m, k=key: self._on_image(m, k), self._queue_size)

    def setup_publishers(self):
        self._inferred_cmd_pub = self.create_publisher(Twist, self.get_parameter("inferred_cmd_topic").value, QoSProfile(history=HistoryPolicy.KEEP_ALL, depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self._ack_pub = self.create_publisher(String, "/pi_bridge/control_ack", self._queue_size)

    def prepare_observation_state(self, odom):
        return np.asarray([odom.linear_x, odom.angular_z], dtype=np.float32)

    def apply_action(self, action):
        msg = Twist(); msg.linear.x, msg.angular.z = action
        self._inferred_cmd_pub.publish(msg)

class TracerFrontLeftBridge(PiWebsocketBridgeBase):
    def setup_robot_config(self):
        self.declare_parameter("front_image_topic", "/camera/front/color/image_raw")
        self.declare_parameter("left_image_topic", "/camera/left/color/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("inferred_cmd_topic", "/pi_bridge/inferred_cmd_vel")

    def get_required_observations(self):
        return {"observation/front_image": "front_image_topic", "observation/left_image": "left_image_topic"}

    def setup_subscriptions(self):
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, self._queue_size)
        for key, param in self.get_required_observations().items():
            self.create_subscription(Image, self.get_parameter(param).value, lambda m, k=key: self._on_image(m, k), self._queue_size)

    def setup_publishers(self):
        self._inferred_cmd_pub = self.create_publisher(Twist, self.get_parameter("inferred_cmd_topic").value, QoSProfile(history=HistoryPolicy.KEEP_ALL, depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self._ack_pub = self.create_publisher(String, "/pi_bridge/control_ack", self._queue_size)

    def prepare_observation_state(self, odom):
        return np.asarray([odom.linear_x, odom.angular_z], dtype=np.float32)

    def apply_action(self, action):
        msg = Twist(); msg.linear.x, msg.angular.z = action
        self._inferred_cmd_pub.publish(msg)

class TracerFrontRightBridge(PiWebsocketBridgeBase):
    def setup_robot_config(self):
        self.declare_parameter("front_image_topic", "/camera/front/color/image_raw")
        self.declare_parameter("right_image_topic", "/camera/right/color/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("inferred_cmd_topic", "/pi_bridge/inferred_cmd_vel")

    def get_required_observations(self):
        return {"observation/front_image": "front_image_topic", "observation/right_image": "right_image_topic"}

    def setup_subscriptions(self):
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._on_odom, self._queue_size)
        for key, param in self.get_required_observations().items():
            self.create_subscription(Image, self.get_parameter(param).value, lambda m, k=key: self._on_image(m, k), self._queue_size)

    def setup_publishers(self):
        self._inferred_cmd_pub = self.create_publisher(Twist, self.get_parameter("inferred_cmd_topic").value, QoSProfile(history=HistoryPolicy.KEEP_ALL, depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self._ack_pub = self.create_publisher(String, "/pi_bridge/control_ack", self._queue_size)

    def prepare_observation_state(self, odom):
        return np.asarray([odom.linear_x, odom.angular_z], dtype=np.float32)

    def apply_action(self, action):
        msg = Twist(); msg.linear.x, msg.angular.z = action
        self._inferred_cmd_pub.publish(msg)
```