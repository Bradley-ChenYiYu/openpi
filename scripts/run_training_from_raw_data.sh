#!/bin/bash
set -Eeuo pipefail

on_error() {
    local exit_code="$?"
    echo "Error: pipeline failed at line ${BASH_LINENO[0]} (exit code: ${exit_code})."
    exit "$exit_code"
}

trap on_error ERR

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname "$script_dir")

cd "$repo_root"

# ===== Pipeline Step Configuration =====
# Available steps: 1=rosbag2video, 2=generate_vid, 3=convert_rosbag, 4=compute_stats, 5=wandb_login_and_train
# Example: START_STEP=5 to skip to wandb login and training
START_STEP="${START_STEP:-1}"

# ===== Path Variables =====
CONFIG_NAME="pi05_tracer_finetune"
METADATA_CONFIG="scripts/rosbag-to-lerobot/config/tracer_metadata.yaml"
TOPIC_MAPPING_CONFIG="scripts/rosbag-to-lerobot/config/tracer_topic_mapping.yaml"
ROSBAG_DIR="rosbag_dir/rosbag_dir_20260417/"
ROSBAG2VIDEO_RATE="50"

# ===== generate_vid_prompt_ollama.py Variables =====
VID_PROMPT_METADATA_PATH="$METADATA_CONFIG"
VID_PROMPT_PARENT_DIR="$ROSBAG_DIR"

# ===== convert_rosbag_to_lerobot.py Variables =====
CONVERT_INPUT_BAG_PATH="$ROSBAG_DIR"
CONVERT_REPO_ID="brad/tracer_data"
CONVERT_ROBOT_TYPE="tracer"
CONVERT_FPS="50"
CONVERT_CONFIG_PATH="$TOPIC_MAPPING_CONFIG"
CONVERT_METADATA_PATH="$METADATA_CONFIG"
CONVERT_FORCE_CLEAN_OUTPUT="--force-clean-output"

# ===== compute_norm_stats.py Variables =====
COMPUTE_NORM_CONFIG_NAME="$CONFIG_NAME"

# ===== train.py Variables =====
TRAIN_CONFIG_NAME="$CONFIG_NAME"
TRAIN_EXP_NAME="pi05_tracer_soc3f_cafe_PromptStop"
TRAIN_OVERWRITE_FLAG="--overwrite"
TRAIN_XLA_MEM_FRACTION="0.9"
TRAIN_OUTPUT_LOG="train_output.log"

# ===== Environment Variable Checks =====
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    echo "Please set it before running this script:"
    echo "  export WANDB_API_KEY='your-api-key-here'"
    exit 1
fi

echo "Starting pipeline from step: $START_STEP"
echo "Available steps:"
echo "  1 = Convert rosbag to videos (ROSBAG_DIR)"
echo "  2 = Generate video prompts (selected bag dir)"
echo "  3 = Convert rosbag to LeRobot"
echo "  4 = Compute normalization statistics"
echo "  5 = Weights & Biases login + Train model"
echo ""

# ===== Execute Commands =====

# Step 1: Convert rosbag data to videos
if [[ $START_STEP -le 1 ]]; then
    echo "Step 1/5: Converting rosbag data to videos..."
    uv run scripts/rosbag-to-lerobot/rosbag2video/rosbag2video.py \
        -r "$ROSBAG2VIDEO_RATE" "$ROSBAG_DIR"
    echo "✓ Step 1 completed"
else
    echo "⊘ Step 1 skipped"
fi

# Step 2: Generate video prompts using Ollama
if [[ $START_STEP -le 2 ]]; then
    echo "Step 2/5: Generating video prompts..."
    uv run scripts/rosbag-to-lerobot/generate_vid_prompt_ollama.py \
        --metadata-path "$VID_PROMPT_METADATA_PATH" \
        --parent-dir "$VID_PROMPT_PARENT_DIR"
    echo "✓ Step 2 completed"
else
    echo "⊘ Step 2 skipped"
fi

# Step 3: Convert rosbag to LeRobot format
if [[ $START_STEP -le 3 ]]; then
    echo "Step 3/5: Converting rosbag to LeRobot format..."
    uv run scripts/rosbag-to-lerobot/convert_rosbag_to_lerobot.py \
        --input-bag-path "$CONVERT_INPUT_BAG_PATH" \
        --repo-id "$CONVERT_REPO_ID" \
        --robot-type "$CONVERT_ROBOT_TYPE" \
        --fps "$CONVERT_FPS" \
        --config-path "$CONVERT_CONFIG_PATH" \
        --metadata-path "$CONVERT_METADATA_PATH" \
        "$CONVERT_FORCE_CLEAN_OUTPUT"
    echo "✓ Step 3 completed"
else
    echo "⊘ Step 3 skipped"
fi

# Step 4: Compute normalization statistics
if [[ $START_STEP -le 4 ]]; then
    echo "Step 4/5: Computing normalization statistics..."
    uv run scripts/compute_norm_stats.py \
        --config-name "$COMPUTE_NORM_CONFIG_NAME"
    echo "✓ Step 4 completed"
else
    echo "⊘ Step 4 skipped"
fi

# Step 5: Login to Weights & Biases and Train the model
if [[ $START_STEP -le 5 ]]; then
    echo "Step 5/5: Logging into Weights & Biases..."
    uv run wandb login
    echo "Step 5/5: Starting model training..."
    (XLA_PYTHON_CLIENT_MEM_FRACTION="$TRAIN_XLA_MEM_FRACTION" nohup uv run scripts/train.py "$TRAIN_CONFIG_NAME" \
        --exp-name="$TRAIN_EXP_NAME" \
        "$TRAIN_OVERWRITE_FLAG" \
        > "$TRAIN_OUTPUT_LOG" 2>&1 &)
    echo "✓ Step 5 completed (training running in background)"
else
    echo "⊘ Step 5 skipped"
fi