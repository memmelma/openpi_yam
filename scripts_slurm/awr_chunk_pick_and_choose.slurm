#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=awr_chunk_pick_and_choose
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=logs_slurm/slurm-%x-%j.out
#SBATCH --error=logs_slurm/slurm-%x-%j.err

set -euo pipefail

echo "START TIME: $(date)"

export OPENPI_HOME="/gpfs/home/memmelma/projects/openpi_yam"
export OPENPI_DATA_HOME="/gpfs/scrubbed/memmelma/projects/openpi_yam/openpi"

CONFIG="awr_chunk_pick_and_choose"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
cd $OPENPI_HOME && uv run scripts/compute_norm_stats.py --config-name=$CONFIG && srun uv run scripts/train.py \
    $CONFIG \
      --exp-name $CONFIG \
      --num-train-steps 50000 \
      --overwrite

echo "END TIME: $(date)"
