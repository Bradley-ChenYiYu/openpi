#!/bin/bash
export PATH=/home/lilclaw/.local/bin:$PATH
cd /home/shared/openpi
GIT_LFS_SKIP_SMUDGE=1 nohup /home/lilclaw/.local/bin/uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /home/shared/modified_libero_rlds/ > /home/shared/openpi/convert_libero.log 2>&1 &
