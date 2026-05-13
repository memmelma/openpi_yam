#!/bin/bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    weighted_bc_tillicum \
      --exp-name debug_run_1 \
      --num-train-steps 15000 \
      --overwrite
      # --resume
exec bash
