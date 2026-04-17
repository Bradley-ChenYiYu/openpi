# pi_bridge

ROS2 websocket bridge node for OpenPI policy serving.

## Features

- Subscribes to image (`sensor_msgs/msg/Image` or `sensor_msgs/msg/CompressedImage`) and odometry (`nav_msgs/msg/Odometry`).
- Bundles latest synchronized image + velocity into websocket inference requests.
- Uses OpenPI-compatible msgpack+numpy binary websocket contract.
- Handles websocket metadata frame, heartbeat ping, reconnect backoff, and clean shutdown.
- Republishes policy responses to ROS2 topics:
  - inferred command topic (`geometry_msgs/msg/Twist`)
  - ack topic (`std_msgs/msg/String` JSON)
  - raw response topic (`std_msgs/msg/String` JSON)
  - diagnostics topic (`std_msgs/msg/String` JSON)

## Parameters

- `websocket_host` (string, default: `127.0.0.1`)
- `websocket_port` (int, default: `8000`)
- `api_key` (string, optional)
- `image_topic` (string)
- `odom_topic` (string)
- `image_topic_type` (string: `image` or `compressed`)
- `inferred_cmd_topic` (string)
- `ack_topic` (string)
- `raw_response_topic` (string)
- `diagnostics_topic` (string)
- `queue_size` (int)
- `send_rate_hz` (float)
- `reconnect_interval_sec` (float)
- `max_reconnect_interval_sec` (float)
- `sync_tolerance_sec` (float)
- `prompt` (string)

## Build

From the workspace root:

```bash
cd ros2_ws
colcon build --packages-select pi_bridge
source install/setup.bash
```

## Run

### With `ros2 run`

```bash
ros2 run pi_bridge websocket_bridge --ros-args \
  -p websocket_host:=127.0.0.1 \
  -p websocket_port:=8000 \
  -p image_topic:=/camera/image_raw \
  -p odom_topic:=/odom \
  -p send_rate_hz:=10.0
```

### Run random test publisher

```bash
ros2 run pi_bridge random_test_publisher --ros-args \
  -p image_topic:=/camera/image_raw \
  -p odom_topic:=/odom \
  -p publish_rate_hz:=10.0
```

### With `ros2 launch`

```bash
ros2 launch pi_bridge websocket_bridge.launch.py \
  websocket_host:=127.0.0.1 \
  websocket_port:=8000 \
  image_topic:=/camera/image_raw \
  odom_topic:=/odom
```

Note: OpenPI websocket serving is request/response oriented. This bridge uses infer-style round trips (`send` then blocking `recv`) for each request, matching the behavior of [examples/tracer/main.py](examples/tracer/main.py).

## OpenPI server test flow

1. Start policy server:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_tracer_finetune --policy.dir=checkpoints/pi0_tracer_finetune/my_experiment_tracer/4999
```

2. Optionally run reference client:

```bash
python examples/tracer/main.py --host localhost --port 8000
```

3. Start bridge node and publish test topics.

4. Validate outputs:

```bash
ros2 topic echo /pi_bridge/diagnostics
ros2 topic echo /pi_bridge/control_ack
ros2 topic hz /pi_bridge/inferred_cmd_vel
```

5. Restart the policy server and verify reconnect behavior from bridge logs and diagnostics.
