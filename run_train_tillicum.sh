#!/bin/bash

export OPENPI_DATA_HOME="/gpfs/scrubbed/memmelma/projects/openpi_yam/openpi"
CONFIG="weighted_bc_tillicum"

# uv run scripts/compute_norm_stats.py --config-name=$CONFIG

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    $CONFIG \
      --exp-name black_bowl \
      --num-train-steps 25000 \
      --overwrite
      # --resume
exec bash
