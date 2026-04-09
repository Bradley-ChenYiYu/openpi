---
description: Build an offline ROS2 rosbag2 to LeRobot conversion workflow.
---

## Plan: ROS2 Rosbag to LeRobot Converter

Build an offline ROS2 rosbag2-to-LeRobot converter in a ROS2 Python package path requested by the user, reusing the LeRobot dataset creation and per-frame ingestion pattern used by existing converters while adding configurable topic mapping, approximate timestamp synchronization, and per-episode metadata task text.

**Steps**
1. Phase 1: Define converter contract and configuration schema.
2. Specify CLI arguments for input bag path, repo id, robot type, fps, output behavior, topic mapping config path, and metadata path; include validation rules for required topics and message types.
3. Define external config format (YAML or JSON) for camera topics list, state topic, action topic, optional time tolerance, and optional resize/encoding options; include defaults that let users fill mappings later. *parallel with step 2*
4. Define metadata format for one-episode-per-bag mode (task text, optional episode-level tags, optional split name), and fallback behavior when metadata is missing. *depends on 2*
5. Phase 2: Create ROS2 bag ingestion and synchronization design.
6. Implement rosbag2 reader abstraction using Python rosbag2 API/libraries and deserialize messages for configurable topics; include topic discovery and human-readable errors when configured topics are absent. *depends on 1-4*
7. Implement approximate synchronization strategy: choose a reference stream (action or highest-rate state), then select nearest neighbor messages from other streams within tolerance; drop or warn for unmatched samples per policy. *depends on 6*
8. Convert synchronized samples into frame records using selected schema: observation.images.<camera_name>, observation.state (dim 6), action (dim 2), task (from metadata).
9. Phase 3: Integrate LeRobot dataset writing pattern.
10. Add dataset factory function mirroring the pattern from existing converters (LeRobotDataset.create with explicit features, fps, writer threading/process options).
11. For each bag file (one episode), stream synchronized frames via add_frame and finalize with save_episode(task=...) using metadata task text.
12. Optionally consolidate/push dataset if requested by CLI flags, matching existing converter conventions. *depends on 10-11*
13. Phase 4: Packaging and repository integration.
14. Create the ROS2 Python package first using ros2 pkg create --build-type ament_python <pkg>, then add the converter module at ros2_ws/src/<pkg>/<pkg>/<converter>.py and required package wiring for execution via ros2 run. *depends on 1*
15. Add a short README usage section in the same ROS2 package describing config and metadata files and an end-to-end command example. *depends on 14*
16. Phase 5: Verification.
17. Validate with a small rosbag2 sample and a minimal mapping config; confirm dataset folder generation, feature shapes, and episode count.
18. Run smoke checks for topic discovery, missing-topic errors, deserialization failures, and sync drop-rate reporting.
19. Compare output structure against expected LeRobot schema by loading dataset metadata/stats and checking state/action dimensions and camera field names.

**Relevant files**
- /home/shared/openpi/examples/libero/convert_libero_data_to_lerobot.py — reuse minimal flow: clean output, create dataset, add_frame loop, save_episode, optional push.
- /home/shared/openpi/examples/aloha_real/convert_aloha_data_to_lerobot.py — reuse stronger modular pattern: create_empty_dataset, per-episode population, feature declaration style.
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/<PKG_NAME>/<CONVERTER>.py — new converter implementation target selected by user.
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/package.xml — ROS2 package dependencies for rosbag2 reading and runtime.
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/setup.py — Python entry points for converter execution.
- /home/shared/openpi/ros2_ws/src/<PKG_NAME>/README.md — converter usage, config schema, metadata schema.

**Verification**
1. Run converter on one rosbag2 file with a test mapping config and metadata file; verify one saved episode and non-empty frames.
2. Inspect created LeRobot feature manifest and ensure required keys exist: observation.images.<camera_name>, observation.state (6), action (2), task.
3. Run with intentionally wrong topic names to verify clear validation errors before conversion starts.
4. Run with tighter sync tolerance to verify warning metrics for dropped/unmatched samples.
5. (If enabled) run push-to-hub dry path disabled by default and confirm no network side effects during local validation.

**Decisions**
- Include scope: ROS2 rosbag2 offline conversion from bag path, not live topic subscription.
- Include scope: configurable topic mapping via external config, approximate synchronization, single episode per bag.
- Include scope: schema fields observation.images.<camera_name>, observation.state, action, task.
- Include scope: dimensions state=6 and action=2.
- Include scope: task text source from per-episode metadata file.
- Exclude scope: ROS1 bag support, automatic semantic inference of unknown topic meanings, multi-episode segmentation inside one bag.

**Further Considerations**
1. Config format choice recommendation: YAML is easier for multi-topic mapping; JSON is simpler for strict parsing.
2. Synchronization recommendation: use action as reference if action rate is stable; otherwise use state and resample action nearest neighbor.
3. Image normalization recommendation: normalize to a single resolution per camera in converter config to avoid heterogeneous frame shapes.