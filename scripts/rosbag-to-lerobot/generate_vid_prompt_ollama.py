import cv2
import base64
import requests
import os
import argparse
import sys
from pathlib import Path
'''
This script processes a video file and sends them to an Ollama model for action labeling.

Usage:
1. Connect to ollama server:
    ssh -R 11434:localhost:11434 your_ollama_server_user@your_ollama_server_ip
2. Run this script:
    ```
    uv run scripts/rosbag-to-lerobot/generate_vid_prompt_ollama.py \
    --metadata-path /home/shared/openpi/scripts/rosbag-to-lerobot/config/tracer_metadata.yaml  \
    --parent-dir /home/shared/openpi/rosbag_dir/
    ```
'''

# Configuration
VIDEO_PATH = "/home/shared/openpi/rosbag_dir/rosbag2_2026_04_17-13_41_55/_camera_camera_color_image_raw.mp4"
MODEL = "gemma4:31b"    #"kimi-k2.5:cloud"
OLLAMA_URL = "http://localhost:11434/api/generate"
TARGET_FPS = 3  # Extract 3 frames per second
PROMPT = """
You are an AI assistant specialized in robot navigation. Watch this video from a mobile robot's perspective and identify the specific target destination.

Your task is to generate a direct natural language command that is specific enough to distinguish the target from other similar objects in the room.

### Instructions:
1. Identify the primary object the robot is approaching.
2. Use distinguishing attributes to disambiguate it from others (e.g., "the wooden table," "the table on the far left," or "the table with the laptop").
3. Format the output as a command for the robot to execute.

### Examples:
- Input: [Robot moves toward one of three tables; this one has a red bag]
  Output: Navigate to the table with the red bag.
- Input: [Robot turns toward the second desk in a row]
  Output: Move to the second desk on the right side.

### Constraints:
- Output ONLY the command.
- No preamble, no "Here is the instruction," and no description of camera movement.
""".strip()


def _load_yaml_mapping(yaml_path: Path) -> dict[str, object]:
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedMap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ruamel.yaml is required for comment-preserving metadata read/write. Install with `uv sync` or `pip install ruamel.yaml`."
        ) from exc

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)

    if not yaml_path.exists():
        return CommentedMap({"episodes": CommentedMap()})

    raw = yaml.load(yaml_path.read_text(encoding="utf-8"))
    if raw is None:
        return CommentedMap({"episodes": CommentedMap()})
    if not isinstance(raw, dict):
        raise ValueError(f"Metadata YAML must be a mapping at root: {yaml_path}")
    if "episodes" not in raw or raw["episodes"] is None:
        raw["episodes"] = CommentedMap()
    if not isinstance(raw["episodes"], dict):
        raise ValueError(f"'episodes' must be a mapping in: {yaml_path}")
    return raw


def _write_yaml_mapping(yaml_path: Path, data: dict[str, object]) -> None:
    try:
        from ruamel.yaml import YAML
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ruamel.yaml is required for comment-preserving metadata read/write. Install with `uv sync` or `pip install ruamel.yaml`."
        ) from exc

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)


def append_episode_metadata(
    metadata_path: str,
    episode_name: str,
    task: str,
    tags: list[str] | None = None,
    split: str | None = None,
    overwrite: bool = False,
) -> bool:
    metadata_file = Path(metadata_path)
    metadata = _load_yaml_mapping(metadata_file)
    episodes = metadata["episodes"]

    if episode_name in episodes and not overwrite:
        print(
            f"Episode '{episode_name}' already exists in {metadata_file}. "
            "Use --overwrite-episode to update it."
        )
        return False

    episode_entry = {"task": task}
    if tags:
        try:
            from ruamel.yaml.comments import CommentedSeq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ruamel.yaml is required for comment-preserving metadata read/write. Install with `uv sync` or `pip install ruamel.yaml`."
            ) from exc

        episode_tags = CommentedSeq(tags)
        episode_tags.fa.set_flow_style()
        episode_entry["tags"] = episode_tags
    if split:
        episode_entry["split"] = split

    episodes[episode_name] = episode_entry
    _write_yaml_mapping(metadata_file, metadata)
    print(f"Updated metadata: {metadata_file} (episode: {episode_name})")
    return True


def infer_episode_name(video_path: str) -> str:
    parent_name = Path(video_path).resolve().parent.name
    if parent_name:
        return parent_name
    return Path(video_path).stem


def find_video_paths(parent_dir: str) -> list[Path]:
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_path}")
    if not parent_path.is_dir():
        raise ValueError(f"Parent path must be a directory: {parent_path}")

    video_paths: list[Path] = []
    for child_dir in sorted(path for path in parent_path.iterdir() if path.is_dir()):
        video_paths.extend(
            sorted(
                path
                for path in child_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
            )
        )

    if not video_paths:
        raise ValueError(f"No video files found under: {parent_path}")
    return video_paths


def process_and_maybe_append_video(
    video_path: str,
    target_fps: int,
    model_name: str,
    api_url: str,
    prompt: str,
    metadata_path: str | None,
    task_override: str | None,
    tags: list[str] | None,
    split: str | None,
    overwrite: bool,
    episode_name: str | None = None,
) -> str | None:
    generated_task = process_video(video_path, target_fps, model_name, api_url, prompt)
    task_to_store = task_override or generated_task

    if metadata_path is not None:
        if not task_to_store:
            raise ValueError(
                "No task text available to append. Provide --task or ensure Ollama returns text."
            )
        resolved_episode_name = episode_name or infer_episode_name(video_path)
        append_episode_metadata(
            metadata_path=metadata_path,
            episode_name=resolved_episode_name,
            task=task_to_store,
            tags=tags,
            split=split,
            overwrite=overwrite,
        )

    return task_to_store

def process_video(video_path, target_fps, model_name, api_url, prompt):
    print(f"Processing video: {video_path}")
    
    # 1. Extract Frames directly to memory (no temp dir needed)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None

    base64_images = []
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame interval to match target FPS
    # e.g., if video is 30fps and we want 3fps, we grab every 10th frame
    frame_interval = int(original_fps / target_fps)
    if frame_interval == 0: frame_interval = 1 # Handle low fps videos

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process frames at the specific interval
        if frame_count % frame_interval == 0:
            # Encode frame to JPG in memory
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64 string
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            base64_images.append(jpg_as_text)
            
        frame_count += 1

    cap.release()
    print(f"Extracted {len(base64_images)} frames.")

    # 2. Send to Ollama
    print("Sending payload to Ollama...")
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "images": base64_images,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status() # Raise error for bad status codes
        
        result = response.json()
        model_response = result.get("response", "").strip()
        print("\n--- Response ---")
        print(model_response or "No response found")
        return model_response
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


def parse_tags(raw_tags: str | None) -> list[str]:
    if not raw_tags:
        return []
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Process a video with Ollama and optionally append the generated task "
            "to tracer metadata YAML under episodes."
        )
    )
    parser.add_argument("--video-path", default=VIDEO_PATH)
    parser.add_argument(
        "--parent-dir",
        default=None,
        help="Grandparent folder containing one subfolder per episode, each with one or more videos.",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--target-fps", type=int, default=TARGET_FPS)
    parser.add_argument("--prompt", default=PROMPT)

    parser.add_argument(
        "--metadata-path",
        default=os.path.join(os.path.dirname(__file__), "config", "tracer_metadata.yaml"),
        help="Path to metadata YAML file to update.",
    )
    parser.add_argument(
        "--append-metadata",
        action="store_true",
        help="Append generated (or provided) task text to metadata episodes.",
    )
    parser.add_argument(
        "--episode-name",
        default=None,
        help="Episode key under metadata. Defaults to parent directory name of the video file.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional explicit task text. If omitted, model response is used.",
    )
    parser.add_argument(
        "--tags",
        default=None,
        help="Comma-separated tags for the episode, e.g. 'demo,train'.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split for the episode, e.g. train/val/test.",
    )
    parser.add_argument(
        "--overwrite-episode",
        action="store_true",
        help="Overwrite existing episode metadata if the episode key already exists.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    append_requested = args.append_metadata or "--metadata-path" in sys.argv
    metadata_path = args.metadata_path if append_requested else None

    if args.parent_dir:
        if args.episode_name is not None:
            raise ValueError("--episode-name cannot be used with --parent-dir batch mode.")

        video_paths = find_video_paths(args.parent_dir)
        print(f"Found {len(video_paths)} video files under {args.parent_dir}")
        for video_path in video_paths:
            print(f"\n=== Processing {video_path} ===")
            process_and_maybe_append_video(
                video_path=str(video_path),
                target_fps=args.target_fps,
                model_name=args.model,
                api_url=args.ollama_url,
                prompt=args.prompt,
                metadata_path=metadata_path,
                task_override=args.task,
                tags=parse_tags(args.tags),
                split=args.split,
                overwrite=args.overwrite_episode,
            )
    else:
        process_and_maybe_append_video(
            video_path=args.video_path,
            target_fps=args.target_fps,
            model_name=args.model,
            api_url=args.ollama_url,
            prompt=args.prompt,
            metadata_path=metadata_path,
            task_override=args.task,
            tags=parse_tags(args.tags),
            split=args.split,
            overwrite=args.overwrite_episode,
            episode_name=args.episode_name,
        )
