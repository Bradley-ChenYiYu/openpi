from __future__ import annotations
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

'''
This script converts ROS bag files into the LeRobot dataset format.  
Usage:  

    ```bash
    uv run scripts/rosbag-to-lerobot/convert_rosbag_to_lerobot.py \
        --input-bag-path /home/shared/rosbag2_2025_09_25-13_49_34/ \
        --repo-id brad/tracer_data --robot-type tracer \
        --fps 3 --config-path ros2_ws/src/rosbag_to_lerobot/config/tracer_topic_mapping.yaml \
        --metadata-path ros2_ws/src/rosbag_to_lerobot/config/tracer_metadata.yaml \
        --force-clean-output --log-level DEBUG
    ```

'''

DEFAULT_SYNC_TOLERANCE_SEC = 0.05
DEFAULT_DROP_POLICY = "drop"
DEFAULT_REFERENCE_STREAM = "action"
DEFAULT_TASK_TEXT = "unspecified task"
DEFAULT_TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TopicSpec:
    topic: str
    msg_type: str | None = None
    dim: int | None = None
    fields: list[str] | None = None


@dataclass(frozen=True)
class CameraSpec:
    name: str
    topic: str
    msg_type: str | None = None
    resize: tuple[int, int] | None = None


@dataclass(frozen=True)
class SyncConfig:
    reference_stream: str
    tolerance_sec: float
    drop_policy: str


@dataclass(frozen=True)
class ConverterConfig:
    cameras: list[CameraSpec]
    state: TopicSpec
    action: TopicSpec
    sync: SyncConfig


@dataclass(frozen=True)
class EpisodeMetadata:
    task: str
    tags: list[str]
    split: str | None


@dataclass
class TimedSample:
    t_sec: float
    value: Any


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YAML config requires PyYAML. Install with `pip install pyyaml`."
            ) from exc
        data = yaml.safe_load(text)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping object in {path}")
        return data
    raise ValueError(f"Unsupported config format for {path}. Use .json, .yaml, or .yml")


def _parse_resize(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("camera resize must be [height, width]")
    h = int(value[0])
    w = int(value[1])
    if h <= 0 or w <= 0:
        raise ValueError("camera resize dimensions must be > 0")
    return (h, w)


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")   # format="%(levelname)s: %(message)s"
    # Keep third-party debug logs from drowning conversion diagnostics.
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jaxlib").setLevel(logging.WARNING)


def load_converter_config(path: Path) -> ConverterConfig:
    raw = _load_json_or_yaml(path)

    camera_topics = raw.get("camera_topics", {})
    if not isinstance(camera_topics, dict) or not camera_topics:
        raise ValueError("config.camera_topics must be a non-empty object")

    cameras: list[CameraSpec] = []
    for camera_name, camera_cfg in camera_topics.items():
        if not isinstance(camera_cfg, dict):
            raise ValueError(f"camera config for {camera_name} must be an object")
        topic = camera_cfg.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"camera {camera_name} requires a non-empty topic")
        cameras.append(
            CameraSpec(
                name=str(camera_name),
                topic=topic,
                msg_type=camera_cfg.get("msg_type"),
                resize=_parse_resize(camera_cfg.get("resize")),
            )
        )

    def parse_topic_spec(section_name: str, expected_dim: int | None = None) -> TopicSpec:
        section = raw.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"config.{section_name} must be an object")
        topic = section.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"config.{section_name}.topic must be a non-empty string")
        dim_raw = section.get("dim", expected_dim)
        dim = None if dim_raw is None else int(dim_raw)
        if expected_dim is not None and dim != expected_dim:
            raise ValueError(f"config.{section_name}.dim must be {expected_dim}")
        fields_raw = section.get("fields")
        fields: list[str] | None = None
        if fields_raw is not None:
            if not isinstance(fields_raw, list) or not all(isinstance(item, str) for item in fields_raw):
                raise ValueError(f"config.{section_name}.fields must be a list of strings")
            fields = [item.strip() for item in fields_raw]
            if any(not item for item in fields):
                raise ValueError(f"config.{section_name}.fields cannot contain empty paths")
            if dim is not None and len(fields) != dim:
                raise ValueError(
                    f"config.{section_name}.fields length ({len(fields)}) must equal config.{section_name}.dim ({dim})"
                )
        return TopicSpec(topic=topic, msg_type=section.get("msg_type"), dim=dim, fields=fields)

    state_spec = parse_topic_spec("state_topic")    # expected_dim=6
    action_spec = parse_topic_spec("action_topic")  # expected_dim=2

    sync_raw = raw.get("sync", {})
    if not isinstance(sync_raw, dict):
        raise ValueError("config.sync must be an object")
    reference_stream = str(sync_raw.get("reference_stream", DEFAULT_REFERENCE_STREAM)).lower()
    if reference_stream not in {"action", "state"}:
        raise ValueError("config.sync.reference_stream must be 'action' or 'state'")
    tolerance_sec = float(sync_raw.get("tolerance_sec", DEFAULT_SYNC_TOLERANCE_SEC))
    if tolerance_sec < 0:
        raise ValueError("config.sync.tolerance_sec must be >= 0")
    drop_policy = str(sync_raw.get("drop_policy", DEFAULT_DROP_POLICY)).lower()
    if drop_policy not in {"drop", "warn"}:
        raise ValueError("config.sync.drop_policy must be 'drop' or 'warn'")

    return ConverterConfig(
        cameras=cameras,
        state=state_spec,
        action=action_spec,
        sync=SyncConfig(
            reference_stream=reference_stream,
            tolerance_sec=tolerance_sec,
            drop_policy=drop_policy,
        ),
    )


def load_metadata(metadata_path: Path | None) -> dict[str, Any]:
    if metadata_path is None:
        return {}
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    return _load_json_or_yaml(metadata_path)


def resolve_episode_metadata(
    metadata_root: dict[str, Any],
    bag_name: str,
    default_task: str,
) -> EpisodeMetadata:
    task = str(metadata_root.get("default_task", default_task))
    tags: list[str] = []
    split: str | None = None

    episodes_obj = metadata_root.get("episodes", {})
    if isinstance(episodes_obj, dict):
        episode_cfg = episodes_obj.get(bag_name)
        if isinstance(episode_cfg, dict):
            task = str(episode_cfg.get("task", task))
            raw_tags = episode_cfg.get("tags", [])
            if isinstance(raw_tags, list):
                tags = [str(tag) for tag in raw_tags]
            raw_split = episode_cfg.get("split")
            split = None if raw_split is None else str(raw_split)

    return EpisodeMetadata(task=task, tags=tags, split=split)


def list_bag_dirs(input_bag_path: Path) -> list[Path]:
    if not input_bag_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_bag_path}")

    def is_bag_dir(path: Path) -> bool:
        return path.is_dir() and (path / "metadata.yaml").exists()

    if is_bag_dir(input_bag_path):
        return [input_bag_path]

    candidates = sorted(path for path in input_bag_path.iterdir() if is_bag_dir(path))
    if not candidates:
        raise ValueError(
            "No rosbag2 directories found. Provide a bag directory containing metadata.yaml "
            "or a parent directory that contains bag directories."
        )
    return candidates


def discover_topics(bag_dir: Path) -> dict[str, str]:
    with AnyReader([bag_dir], default_typestore=DEFAULT_TYPESTORE) as reader:
        return {connection.topic: connection.msgtype for connection in reader.connections}


def validate_topic_mapping(config: ConverterConfig, discovered_topics: dict[str, str]) -> None:
    required_topics: list[tuple[str, str, str | None]] = [
        ("state_topic", config.state.topic, config.state.msg_type),
        ("action_topic", config.action.topic, config.action.msg_type),
    ] + [
        (f"camera_topics.{camera.name}", camera.topic, camera.msg_type)
        for camera in config.cameras
    ]

    missing = [key for key, topic, _ in required_topics if topic not in discovered_topics]
    if missing:
        available = "\n".join(
            f"- {name}: {msg_type}" for name, msg_type in sorted(discovered_topics.items())
        )
        raise ValueError(
            "Configured topics were not found in bag:\n"
            + "\n".join(f"- {item}" for item in missing)
            + "\n\nDiscovered topics:\n"
            + available
        )

    mismatched: list[str] = []
    for key, topic, expected_msg_type in required_topics:
        if expected_msg_type and discovered_topics[topic] != expected_msg_type:
            mismatched.append(
                f"{key}: expected {expected_msg_type}, found {discovered_topics[topic]}"
            )
    if mismatched:
        raise ValueError(
            "Message type mismatch between config and bag:\n" + "\n".join(f"- {m}" for m in mismatched)
        )


def _decode_raw_image(msg: Any) -> np.ndarray:
    encoding = str(getattr(msg, "encoding", "")).lower()
    height = int(getattr(msg, "height", 0))
    width = int(getattr(msg, "width", 0))
    data = bytes(getattr(msg, "data", b""))

    if height <= 0 or width <= 0:
        raise ValueError("sensor_msgs/Image must contain positive height and width")

    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
    elif encoding in {"mono8", "8uc1"}:
        channels = 1
    else:
        raise ValueError(f"Unsupported raw image encoding: {encoding}")

    arr = np.frombuffer(data, dtype=np.uint8)
    expected = height * width * channels
    if arr.size != expected:
        raise ValueError(
            f"Unexpected raw image size for encoding {encoding}: got {arr.size}, expected {expected}"
        )

    arr = arr.reshape((height, width, channels))
    if encoding == "bgr8":
        arr = arr[:, :, ::-1]
    elif encoding == "bgra8":
        arr = arr[:, :, [2, 1, 0, 3]]

    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)
    if channels == 4:
        arr = arr[:, :, :3]

    return arr


def _decode_compressed_image(msg: Any) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sensor_msgs/CompressedImage requires OpenCV. Install with `pip install opencv-python`."
        ) from exc

    compressed = np.frombuffer(bytes(getattr(msg, "data", b"")), dtype=np.uint8)
    image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode compressed image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_image_message(msg: Any, msg_type: str) -> np.ndarray:
    if msg_type == "sensor_msgs/msg/Image":
        return _decode_raw_image(msg)
    if msg_type == "sensor_msgs/msg/CompressedImage":
        return _decode_compressed_image(msg)
    raise ValueError(
        f"Unsupported camera message type {msg_type}. Supported types: "
        "sensor_msgs/msg/Image, sensor_msgs/msg/CompressedImage"
    )


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Image resize requires OpenCV or Pillow. Install with `pip install opencv-python` or `pip install pillow`."
            ) from exc
        pil_img = Image.fromarray(image)
        resized = pil_img.resize((width, height), resample=Image.Resampling.BILINEAR)
        return np.asarray(resized)

    interpolation = cv2.INTER_AREA if image.shape[0] > height or image.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_with_pad_numpy(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    cur_h, cur_w = image.shape[:2]
    if cur_h <= 0 or cur_w <= 0:
        raise ValueError(f"Invalid image shape for resize: {image.shape}")

    scale = min(target_w / cur_w, target_h / cur_h)
    resized_w = max(1, int(round(cur_w * scale)))
    resized_h = max(1, int(round(cur_h * scale)))
    resized = _resize_image(image, resized_w, resized_h)

    if resized.ndim != 3 or resized.shape[2] != 3:
        raise ValueError(f"Resized image must be HxWx3, got {resized.shape}")

    padded = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    y0 = (target_h - resized_h) // 2
    x0 = (target_w - resized_w) // 2
    padded[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return padded


def _resolve_field_path(message: Any, field_path: str) -> float:
    current: Any = message
    for part in field_path.split("."):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid empty segment in field path: '{field_path}'")

        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"Field path '{field_path}' failed at '{part}' (dict key not found)")
            current = current[part]
            continue

        if isinstance(current, (list, tuple)) and part.isdigit():
            idx = int(part)
            if idx < 0 or idx >= len(current):
                raise ValueError(f"Field path '{field_path}' failed at index '{part}' (out of range)")
            current = current[idx]
            continue

        if not hasattr(current, part):
            raise ValueError(
                f"Field path '{field_path}' failed at '{part}' on message type {type(current).__name__}"
            )
        current = getattr(current, part)

    if isinstance(current, (list, tuple, dict)):
        raise ValueError(f"Field path '{field_path}' resolved to non-scalar type {type(current).__name__}")
    return float(current)


def extract_numeric_vector(message: Any, field_paths: list[str] | None = None) -> np.ndarray:
    if field_paths:
        values = [_resolve_field_path(message, path) for path in field_paths]
        return np.asarray(values, dtype=np.float32)

    if hasattr(message, "data"):
        data = getattr(message, "data")
        if isinstance(data, list | tuple):
            return np.asarray(data, dtype=np.float32)

    if hasattr(message, "position"):
        pos = getattr(message, "position")
        if isinstance(pos, list | tuple):
            return np.asarray(pos, dtype=np.float32)

    if hasattr(message, "linear") and hasattr(message, "angular"):
        linear = getattr(message, "linear")
        angular = getattr(message, "angular")
        if hasattr(linear, "x") and hasattr(angular, "z"):
            return np.asarray([float(linear.x), float(angular.z)], dtype=np.float32)

    raise ValueError(
        f"Could not extract numeric vector from message type {type(message).__name__}. "
        "Supported patterns: msg.data, msg.position, or Twist(linear.x, angular.z)."
    )


def maybe_resize(image: np.ndarray, resize: tuple[int, int] | None) -> np.ndarray:
    if resize is None:
        return image
    target_h, target_w = resize
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image

    return resize_with_pad_numpy(image, target_h, target_w)


def read_streams_from_bag(
    bag_dir: Path,
    config: ConverterConfig,
    discovered_topics: dict[str, str],
) -> tuple[
    list[TimedSample],
    list[TimedSample],
    dict[str, list[TimedSample]],
]:
    state_samples: list[TimedSample] = []
    action_samples: list[TimedSample] = []
    camera_samples: dict[str, list[TimedSample]] = {camera.name: [] for camera in config.cameras}

    camera_by_topic = {camera.topic: camera for camera in config.cameras}
    selected_topics = {config.state.topic, config.action.topic, *camera_by_topic.keys()}

    message_count = 0
    error_count = 0
    topic_counts: dict[str, int] = {topic: 0 for topic in selected_topics}
    first_seen_t: dict[str, float] = {}
    last_seen_t: dict[str, float] = {}

    try:
        with AnyReader([bag_dir], default_typestore=DEFAULT_TYPESTORE) as reader:
            selected_connections = [conn for conn in reader.connections if conn.topic in selected_topics]
            logger.info("    processing %s selected connections...", len(selected_connections))
            
            for connection, timestamp_ns, raw_data in reader.messages(connections=selected_connections):
                try:
                    topic = connection.topic
                    msg = reader.deserialize(raw_data, connection.msgtype)
                    t_sec = float(timestamp_ns) / 1e9
                    message_count += 1
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    if topic not in first_seen_t:
                        first_seen_t[topic] = t_sec
                    last_seen_t[topic] = t_sec

                    if message_count % 100 == 0:
                        logger.debug("    progress: processed %s messages...", message_count)

                    if topic == config.state.topic:
                        state_samples.append(
                            TimedSample(
                                t_sec=t_sec,
                                value=extract_numeric_vector(msg, field_paths=config.state.fields),
                            )
                        )
                    elif topic == config.action.topic:
                        action_samples.append(
                            TimedSample(
                                t_sec=t_sec,
                                value=extract_numeric_vector(msg, field_paths=config.action.fields),
                            )
                        )
                    else:
                        camera_spec = camera_by_topic[topic]
                        image = decode_image_message(msg, discovered_topics[topic])
                        image = maybe_resize(image, camera_spec.resize)
                        camera_samples[camera_spec.name].append(TimedSample(t_sec=t_sec, value=image))
                
                except Exception as e:
                    error_count += 1
                    logger.exception(
                        "    error processing message %s: %s: %s",
                        message_count,
                        type(e).__name__,
                        e,
                    )
                    if error_count >= 10:
                        logger.error("    too many errors (%s), stopping stream processing", error_count)
                        raise RuntimeError(f"Too many message processing errors: {error_count}") from e
                    continue
    
    except Exception as e:
        logger.exception("    fatal error reading bag: %s: %s", type(e).__name__, e)
        raise

    logger.info("    processed %s messages total (%s errors)", message_count, error_count)
    logger.info("    per-topic ingest summary:")
    for topic in sorted(selected_topics):
        count = topic_counts.get(topic, 0)
        if count == 0:
            logger.info("      - %s: count=0", topic)
            continue
        start_t = first_seen_t[topic]
        end_t = last_seen_t[topic]
        span = end_t - start_t
        logger.info(
            "      - %s: count=%s, first_t=%.3f, last_t=%.3f, span_sec=%.3f",
            topic,
            count,
            start_t,
            end_t,
            span,
        )
    return state_samples, action_samples, camera_samples


def _nearest_within_tolerance(
    samples: list[TimedSample],
    t_ref: float,
    tolerance_sec: float,
) -> TimedSample | None:
    if not samples:
        return None

    times = [sample.t_sec for sample in samples]
    idx = int(np.searchsorted(times, t_ref))

    candidates: list[TimedSample] = []
    if 0 <= idx < len(samples):
        candidates.append(samples[idx])
    if idx - 1 >= 0:
        candidates.append(samples[idx - 1])

    if not candidates:
        return None

    best = min(candidates, key=lambda sample: abs(sample.t_sec - t_ref))
    if abs(best.t_sec - t_ref) > tolerance_sec:
        return None
    return best


def build_synchronized_frames(
    config: ConverterConfig,
    state_samples: list[TimedSample],
    action_samples: list[TimedSample],
    camera_samples: dict[str, list[TimedSample]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not state_samples:
        raise ValueError("No state samples found in bag")
    if not action_samples:
        raise ValueError("No action samples found in bag")
    for camera_name, samples in camera_samples.items():
        if not samples:
            raise ValueError(f"No samples found for camera {camera_name}")

    reference_samples = action_samples if config.sync.reference_stream == "action" else state_samples

    inferred_state_dim = int(np.asarray(state_samples[0].value).shape[0])
    inferred_action_dim = int(np.asarray(action_samples[0].value).shape[0])
    expected_state_dim = config.state.dim if config.state.dim is not None else inferred_state_dim
    expected_action_dim = config.action.dim if config.action.dim is not None else inferred_action_dim

    dropped_out_of_tolerance = 0
    dropped_shape = 0
    shape_mismatch_examples: list[str] = []
    frames: list[dict[str, Any]] = []
    dropped_missing_state = 0
    dropped_missing_action = 0
    dropped_missing_camera = 0
    dropped_missing_camera_by_name: dict[str, int] = {name: 0 for name in camera_samples}
    dropped_examples: list[str] = []

    for ref in reference_samples:
        state_match = _nearest_within_tolerance(
            state_samples, t_ref=ref.t_sec, tolerance_sec=config.sync.tolerance_sec
        )
        action_match = _nearest_within_tolerance(
            action_samples, t_ref=ref.t_sec, tolerance_sec=config.sync.tolerance_sec
        )

        camera_matches: dict[str, TimedSample] = {}
        missing_camera_name: str | None = None
        for camera_name, samples in camera_samples.items():
            match = _nearest_within_tolerance(samples, t_ref=ref.t_sec, tolerance_sec=config.sync.tolerance_sec)
            if match is None:
                missing_camera_name = camera_name
                camera_matches = {}
                break
            camera_matches[camera_name] = match

        if state_match is None or action_match is None or not camera_matches:
            dropped_out_of_tolerance += 1
            if state_match is None:
                dropped_missing_state += 1
            if action_match is None:
                dropped_missing_action += 1
            if missing_camera_name is not None:
                dropped_missing_camera += 1
                dropped_missing_camera_by_name[missing_camera_name] = (
                    dropped_missing_camera_by_name.get(missing_camera_name, 0) + 1
                )
            if len(dropped_examples) < 5:
                dropped_examples.append(
                    "t_ref={:.3f}, missing_state={}, missing_action={}, missing_camera={}".format(
                        ref.t_sec,
                        state_match is None,
                        action_match is None,
                        missing_camera_name if missing_camera_name is not None else "none",
                    )
                )
            if config.sync.drop_policy == "warn":
                continue
            continue

        state_vec = np.asarray(state_match.value, dtype=np.float32)
        action_vec = np.asarray(action_match.value, dtype=np.float32)
        state_ok = state_vec.ndim == 1 and state_vec.shape[0] == expected_state_dim
        action_ok = action_vec.ndim == 1 and action_vec.shape[0] == expected_action_dim
        if not state_ok or not action_ok:
            dropped_shape += 1
            if len(shape_mismatch_examples) < 5:
                shape_mismatch_examples.append(
                    f"state={tuple(state_vec.shape)} expected=({expected_state_dim},), "
                    f"action={tuple(action_vec.shape)} expected=({expected_action_dim},)"
                )
            continue

        frame: dict[str, Any] = {
            "observation.state": state_vec,
            "action": action_vec,
        }
        for camera_name, sample in camera_matches.items():
            frame[f"observation.images.{camera_name}"] = sample.value

        frames.append(frame)

    stats = {
        "reference_samples": len(reference_samples),
        "frames": len(frames),
        "dropped_out_of_tolerance": dropped_out_of_tolerance,
        "dropped_missing_state": dropped_missing_state,
        "dropped_missing_action": dropped_missing_action,
        "dropped_missing_camera": dropped_missing_camera,
        "dropped_missing_camera_by_name": dropped_missing_camera_by_name,
        "dropped_shape": dropped_shape,
        "expected_state_dim": expected_state_dim,
        "expected_action_dim": expected_action_dim,
        "shape_mismatch_examples": len(shape_mismatch_examples),
    }
    if dropped_examples:
        logger.warning("Out-of-tolerance examples (up to 5):")
        for example in dropped_examples:
            logger.warning("  - %s", example)
    if shape_mismatch_examples:
        logger.warning("Shape mismatch examples (up to 5):")
        for example in shape_mismatch_examples:
            logger.warning("  - %s", example)
    return frames, stats


def create_dataset(
    repo_id: str,
    robot_type: str,
    fps: int,
    cameras: list[CameraSpec],
    state_dim: int | None,
    action_dim: int | None,
    force_clean: bool,
    image_writer_threads: int,
    image_writer_processes: int,
) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / repo_id
    if force_clean and output_path.exists():
        shutil.rmtree(output_path)
    elif output_path.exists():
        return LeRobotDataset(repo_id)

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }

    for camera in cameras:
        shape = (camera.resize[0], camera.resize[1], 3) if camera.resize is not None else (480, 640, 3)
        features[f"observation.images.{camera.name}"] = {
            "dtype": "image",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def convert_bags(
    input_bag_path: Path,
    repo_id: str,
    robot_type: str,
    fps: int,
    config_path: Path,
    metadata_path: Path | None,
    default_task: str,
    force_clean: bool,
    push_to_hub: bool,
    image_writer_threads: int,
    image_writer_processes: int,
) -> None:
    config = load_converter_config(config_path)
    if config.state.dim is None:
        raise ValueError("config.state_topic.dim is required to define dataset feature shape")
    if config.action.dim is None:
        raise ValueError("config.action_topic.dim is required to define dataset feature shape")

    metadata_root = load_metadata(metadata_path)
    dataset_path = HF_LEROBOT_HOME / repo_id

    bag_dirs = list_bag_dirs(input_bag_path)
    if not bag_dirs:
        raise ValueError(f"No bag directories to convert under {input_bag_path}")

    logger.info("Starting conversion")
    logger.info("  input path: %s", input_bag_path)
    logger.info("  discovered bag directories: %s", len(bag_dirs))
    logger.info("  output dataset path: %s", dataset_path)
    logger.info("  sync reference stream: %s", config.sync.reference_stream)
    logger.info("  sync tolerance_sec: %s", config.sync.tolerance_sec)
    logger.info("  state topic dim (config): %s", config.state.dim)
    logger.info("  action topic dim (config): %s", config.action.dim)

    dataset = create_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        cameras=config.cameras,
        state_dim=config.state.dim,
        action_dim=config.action.dim,
        force_clean=force_clean,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    episodes_saved = 0
    total_frames_saved = 0

    for bag_dir in bag_dirs:
        logger.info("\nConverting bag: %s", bag_dir)
        logger.info("==== Starting bag: %s ====", bag_dir.name)
        
        discovered_topics = discover_topics(bag_dir)
        logger.info("  discovered topics: %s", len(discovered_topics))

        validate_topic_mapping(config, discovered_topics)

        # Show only state, action, and camera topics for brevity
        relevant_topics = {config.state.topic, config.action.topic}
        for camera in config.cameras:
            relevant_topics.add(camera.topic)
        for topic_name in sorted(discovered_topics.keys()):
            if topic_name in relevant_topics:
                logger.info("    - %s: %s", topic_name, discovered_topics[topic_name])

        logger.info("  reading streams from bag...")
        state_samples, action_samples, camera_samples = read_streams_from_bag(
            bag_dir=bag_dir,
            config=config,
            discovered_topics=discovered_topics,
        )
        state_count = len(state_samples)
        action_count = len(action_samples)
        camera_counts = {name: len(samples) for name, samples in camera_samples.items()}
        
        logger.info("  streams loaded successfully")
        logger.info(
            "  sample counts: state=%s, action=%s, cameras=%s",
            state_count,
            action_count,
            camera_counts,
        )
        
        # Additional diagnostics before sync
        if state_count == 0:
            logger.error("  No state samples! Config state topic: %s", config.state.topic)
        if action_count == 0:
            logger.error("  No action samples! Config action topic: %s", config.action.topic)
        for camera_name, count in camera_counts.items():
            if count == 0:
                camera_topic = next(c.topic for c in config.cameras if c.name == camera_name)
                logger.error("  No samples for camera '%s'! Topic: %s", camera_name, camera_topic)

        logger.info(
            "  starting frame synchronization (reference: %s, tolerance: %ss)...",
            config.sync.reference_stream,
            config.sync.tolerance_sec,
        )
        frames, stats = build_synchronized_frames(
            config=config,
            state_samples=state_samples,
            action_samples=action_samples,
            camera_samples=camera_samples,
        )
        logger.info("  synchronization complete. Result: %s synchronized frames", stats["frames"])
        logger.info("    - reference_samples: %s", stats["reference_samples"])
        logger.info("    - out_of_tolerance: %s", stats["dropped_out_of_tolerance"])
        logger.info("      - missing_state: %s", stats["dropped_missing_state"])
        logger.info("      - missing_action: %s", stats["dropped_missing_action"])
        logger.info("      - missing_camera_total: %s", stats["dropped_missing_camera"])
        logger.info("      - missing_camera_by_name: %s", stats["dropped_missing_camera_by_name"])
        logger.info("    - shape_mismatches: %s", stats["dropped_shape"])

        episode_meta = resolve_episode_metadata(
            metadata_root=metadata_root,
            bag_name=bag_dir.name,
            default_task=default_task,
        )

        if not frames:
            logger.warning(
                "Warning: no synchronized frames were produced for "
                f"{bag_dir.name}. Stats: {stats}"
            )
            logger.warning(
                "  Hint: try increasing sync.tolerance_sec and verify state/action dims and configured fields."
            )
            logger.warning(
                "  Tip: compare per-topic first/last timestamps and spans above; "
                "large gaps typically indicate a sync tolerance mismatch."
            )
            continue

        for frame in frames:
            frame["task"] = episode_meta.task
            dataset.add_frame(frame)

        dataset.save_episode()
        episodes_saved += 1
        total_frames_saved += len(frames)
        logger.info(
            f"Saved episode for {bag_dir.name} with {len(frames)} frames. "
            f"Sync stats: {stats}"
        )

    if push_to_hub:
        dataset.push_to_hub()

    logger.info("Conversion finished")
    logger.info("  episodes saved: %s", episodes_saved)
    logger.info("  total frames saved: %s", total_frames_saved)
    logger.info("  dataset path: %s", dataset_path)

    if episodes_saved == 0:
        raise RuntimeError(
            "No episodes were saved. Check sync tolerance, topic-field mapping, and state/action dimensions. "
            f"Dataset path: {dataset_path}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ROS2 rosbag2 recordings into LeRobot format (one episode per bag)."
    )
    parser.add_argument(
        "--input-bag-path",
        required=True,
        type=Path,
        help="Bag directory containing metadata.yaml or a parent directory containing bag directories.",
    )
    parser.add_argument("--repo-id", required=True, help="Target LeRobot repo id, e.g. org/dataset")
    parser.add_argument("--robot-type", required=True, help="LeRobot robot type string")
    parser.add_argument("--fps", type=int, default=10, help="Output dataset frames per second")
    parser.add_argument(
        "--config-path",
        required=True,
        type=Path,
        help="Path to topic mapping config (.yaml/.yml/.json)",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional metadata file with task text and episode tags",
    )
    parser.add_argument(
        "--default-task",
        default=DEFAULT_TASK_TEXT,
        help="Fallback task text when metadata is missing",
    )
    parser.add_argument(
        "--force-clean-output",
        action="store_true",
        help="Delete existing output dataset directory before conversion",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to Hub after conversion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="LeRobot image writer threads",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=5,
        help="LeRobot image writer processes",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)
    logger.info("Starting conversion with arguments:")
    for arg, value in vars(args).items():
        logger.info("  %s: %s", arg, value)
    
    convert_bags(
        input_bag_path=args.input_bag_path,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        config_path=args.config_path,
        metadata_path=args.metadata_path,
        default_task=args.default_task,
        force_clean=args.force_clean_output,
        push_to_hub=args.push_to_hub,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


if __name__ == "__main__":
    main()
