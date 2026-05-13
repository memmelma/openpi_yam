#!/bin/bash

CONFIG="weighted_bc_local"

# uv run scripts/compute_norm_stats.py --config-name=$CONFIG

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    $CONFIG \
      --exp-name block_bowl \
      --num-train-steps 25000 \
      --overwrite
      # --resume
exec bash
