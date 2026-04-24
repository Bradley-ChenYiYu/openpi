#!/usr/bin/env python3
"""
Sum rosbag durations from YAML files under a parent directory.

This script looks specifically for the ROS 2 bag info field and extracts
`rosbag2_bagfile_information.duration.nanoseconds`, converts to seconds, and
reports the total across all YAML files found under the provided parent dir.

Usage:
    python scripts/rosbag-to-lerobot/count_yaml_total_duration.py --parent-dir /path/to/rosbag_dir
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, List, Optional

try:
    import yaml  # PyYAML
    _HAS_PYYAML = True
except Exception:
    _HAS_PYYAML = False

try:
    from ruamel.yaml import YAML as _RuamelYAML
    _HAS_RUAMEL = True
except Exception:
    _HAS_RUAMEL = False

# No date parsing required — we only extract nanoseconds values from rosbag2 info.


def find_yaml_paths(parent_dir: str) -> list[Path]:
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_path}")
    if not parent_path.is_dir():
        raise ValueError(f"Parent path must be a directory: {parent_path}")

    yaml_paths: list[Path] = []
    for child_dir in sorted(path for path in parent_path.iterdir() if path.is_dir()):
        yaml_paths.extend(
            sorted(
                path
                for path in child_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            )
        )

    if not yaml_paths:
        raise ValueError(f"No YAML files found under: {parent_path}")
    return yaml_paths


def _load_yaml_structured(path: Path) -> Optional[Any]:
    text = path.read_text(encoding="utf-8")
    if _HAS_PYYAML:
        try:
            return yaml.safe_load(text)
        except Exception:
            pass
    if _HAS_RUAMEL:
        try:
            ry = _RuamelYAML(typ="safe")
            return ry.load(text)
        except Exception:
            pass
    return None


# Removed generic duration/timestamp parsing — not required for rosbag2 nanoseconds.


def _collect_durations(obj: Any) -> List[float]:
    durations: List[float] = []
    if obj is None:
        return durations

    # Only look for the rosbag2 bagfile information duration nanoseconds field.
    # Example YAML structure handled:
    # rosbag2_bagfile_information:
    #   duration:
    #     nanoseconds: 60282141341
    if isinstance(obj, dict):
        if "rosbag2_bagfile_information" in obj:
            info = obj.get("rosbag2_bagfile_information")
            if isinstance(info, dict):
                dur = info.get("duration")
                if isinstance(dur, dict) and "nanoseconds" in dur:
                    try:
                        nanos = int(dur["nanoseconds"])
                        durations.append(nanos / 1e9)
                    except Exception:
                        pass
            # If we've found the rosbag2 info at this level, return what we have
            return durations

        # Recurse into nested structures to find the rosbag2 key.
        for v in obj.values():
            durations.extend(_collect_durations(v))

    elif isinstance(obj, list):
        for item in obj:
            durations.extend(_collect_durations(item))

    return durations


def parse_yaml_for_total_seconds(path: Path) -> float:
    # Try structured parse first
    data = _load_yaml_structured(path)
    durations: List[float] = []
    if data is not None:
        durations = _collect_durations(data)
    # Fallback: look for nanoseconds fields in raw text (one or more entries)
    if not durations:
        text = path.read_text(encoding="utf-8")
        nanos = [int(m.group(1)) for m in re.finditer(r"(?m)^\s*nanoseconds\s*:\s*([0-9]+)", text)]
        if nanos:
            durations = [n / 1e9 for n in nanos]

    return sum(durations)


def format_hms(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Count total duration in YAML files under a parent dir.")
    default_parent = Path(__file__).resolve().parents[2] / "rosbag_dir"
    parser.add_argument("--parent-dir", default=str(default_parent), help="Parent directory containing episode subfolders (default: repo's rosbag_dir)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    parent = Path(args.parent_dir)
    try:
        yaml_paths = find_yaml_paths(str(parent))
    except Exception as e:
        print(f"Error finding YAML files: {e}")
        return 2

    grand_total = 0.0
    for p in yaml_paths:
        try:
            secs = parse_yaml_for_total_seconds(p)
        except Exception as e:
            if args.verbose:
                print(f"Failed to parse {p}: {e}")
            secs = 0.0
        grand_total += secs
        if args.verbose:
            print(f"{p}: {secs:.2f} s ({format_hms(secs)})")

    print(f"Files scanned: {len(yaml_paths)}")
    print(f"Total duration: {grand_total:.2f} s ({format_hms(grand_total)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
