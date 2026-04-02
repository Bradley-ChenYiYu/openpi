#!/bin/bash
# /home/lilclaw/.local/bin/uv run python -m lerobot.scripts.visualize_dataset --mode local --repo-id your_hf_username/libero --root /home/lilclaw/.cache/huggingface/lerobot/your_hf_username/libero/  --episode-index 0

uv run python examples/libero/visualize_dataset.py --mode distant --repo-id your_hf_username/libero --root /home/shared/huggingface/lerobot/your_hf_username/libero/  --episode-index 0