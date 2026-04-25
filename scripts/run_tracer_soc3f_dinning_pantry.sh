#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")

cd "$repo_root"

# uv run scripts/rosbag-to-lerobot/rosbag2video/rosbag2video.py -r 50 rosbag_dir/rosbag_dir_20260424/ && \
uv run scripts/rosbag-to-lerobot/generate_vid_prompt_ollama.py \
    --metadata-path scripts/rosbag-to-lerobot/config/tracer_metadata.yaml \
    --parent-dir rosbag_dir/rosbag_dir_20260424/ && \
uv run scripts/rosbag-to-lerobot/convert_rosbag_to_lerobot.py --input-bag-path rosbag_dir/rosbag_dir_20260424/ --repo-id brad/tracer_data_Soc_3F_dinning_pantry --robot-type tracer --fps 50 --config-path scripts/rosbag-to-lerobot/config/tracer_topic_mapping.yaml --metadata-path scripts/rosbag-to-lerobot/config/tracer_metadata.yaml --force-clean-output && \
uv run wandb login && \
uv run scripts/compute_norm_stats.py --config-name pi0_tracer_finetune && \
(XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 nohup uv run scripts/train.py pi0_tracer_finetune --exp-name=tracer_soc3f_dinning_pantry --overwrite > train_output.log 2>&1 &)