#!/bin/bash
cd ~/Pi0.5_yam
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    memmelma_block_bowl_weighted \
      --exp-name debug_run_1 \
      --num-train-steps 15000 \
      --overwrite
      # --resume
exec bash
