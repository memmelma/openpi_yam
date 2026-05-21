#!/bin/bash

CONFIG="success_only_bc_local_star_wars"

uv run scripts/compute_norm_stats.py --config-name=$CONFIG

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    $CONFIG \
      --exp-name star_wars \
      --num-train-steps 25000 \
      --overwrite
      #--resume
exec bash
