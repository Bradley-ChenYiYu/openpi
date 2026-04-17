---
description: Build a ROS2 websocket bridge package for image/cmd_vel round-trip with PI server.
---

## Plan: ROS2 Websocket Bridge Node

Build a ROS2 Python package in ros2_ws/src requested by the user, containing a node that connects to a PI server websocket, subscribes to image and cmd_vel topics, sends requests to the PI server, receives responses, and republishes translated ROS2 topics.

**Steps**
1. Phase 1: Define package and runtime contract.
2. Create a ROS2 Python package in ros2_ws/src using `ros2 pkg create --build-type ament_python <pkg_name>` and add dependencies for rclpy, sensor_msgs, geometry_msgs, std_msgs, and websocket support library.
3. Define node parameters: websocket_url, image_topic, cmd_vel_topic, output topic names, queue sizes, send rate/throttle, timeout, reconnect interval, and frame encoding mode.
4. Define request/response schema to PI server: include timestamp, frame id, encoded image payload, cmd_vel linear/angular values, and optional robot/session metadata.
5. Phase 2: Implement websocket lifecycle and ROS2 subscriptions.
6. Implement websocket client manager with startup connect, heartbeat/ping, automatic reconnect with backoff, clean shutdown, and send/receive task separation.
7. Subscribe to image topic (sensor_msgs/msg/Image or CompressedImage if configured) and cmd_vel topic (geometry_msgs/msg/Twist), then maintain latest synchronized values using timestamp buffering policy.
8. Add serialization helpers: image conversion/compression/base64, cmd_vel extraction, request JSON packing, and robust size/error checks before send.
9. Phase 3: Response translation to ROS2 topics.
10. Implement response parser for PI server messages with validation and error handling for malformed or partial payloads.
11. Translate server responses into ROS2 outputs (for example: control/ack topic, inferred command topic, or custom message topic) and publish with correct headers/timestamps.
12. Add optional diagnostics publisher (connection status, message latency, dropped frames, parse failures).
13. Phase 4: Packaging and usability.
14. Wire setup.py entry point so node runs via `ros2 run <pkg_name> <node_executable>` and update package.xml dependencies.
15. Add launch file for configurable params and topic remaps; include README usage with sample websocket URL and ros2 run / ros2 launch commands.
16. Phase 5: Verification with real PI serving path.
17. Run an end-to-end local test against `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_tracer_finetune --policy.dir=checkpoints/pi0_tracer_finetune/my_experiment_tracer/4999` and verify the bridge connects and exchanges messages using the same websocket contract as `examples/tracer/main.py`.
18. Validate reconnect behavior by stopping/restarting the PI websocket server and confirming automatic recovery.
19. Verify topic interfaces with `ros2 topic echo` and `ros2 topic hz`, and confirm no blocking in callbacks under expected message rates while the server is actively inferring.

**Relevant files**
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/<PKG_NAME>/<node>.py — websocket bridge node implementation target.
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/package.xml — runtime/build dependencies.
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/setup.py — console_scripts entry point.
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/launch/<bridge>.launch.py — launch configuration and parameters.
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/README.md — setup and run instructions.

**Verification**
1. Start PI server with `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_tracer_finetune --policy.dir=checkpoints/pi0_tracer_finetune/my_experiment_tracer/4999` and run the ROS2 bridge with test params; confirm successful connection log and heartbeat.
2. In parallel, run `examples/tracer/main.py` as a known-good websocket reference client and confirm both clients can independently communicate with the server.
3. Publish test image and cmd_vel messages; verify outgoing PI server requests include both payloads and produce valid policy responses that republish to expected ROS2 topics.
4. Force malformed/partial responses (or schema mismatch fixtures) and verify node logs warnings/errors without crashing.
5. Disconnect/restart PI websocket server and verify reconnect attempts and eventual republish recovery.

**Decisions**
- Include scope: live ROS2 subscription bridge (not offline conversion).
- Include scope: image + cmd_vel bundled request messages to PI websocket server.
- Include scope: server response translation back into ROS2 published topics.
- Include scope: reconnect/timeout/error handling and diagnostics.
- Exclude scope: model training, dataset generation, and rosbag conversion.

**Further Considerations**
1. If image throughput is high, prefer CompressedImage or JPEG compression before websocket send.
2. For synchronization, consider using latest cmd_vel with nearest image timestamp within a bounded tolerance.
3. If response schema may evolve, use versioned message envelopes to maintain compatibility.
