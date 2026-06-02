# restart later once scheduled
cfgrl_candy_05_26_adv_rvlm
cfgrl_candy_05_26_adv_rbm
cfgrl_candy_05_26_adv_topreward

# not running
awr_chunk_candy_05_26_rvlm
awr_chunk_candy_05_26_rbm
awr_chunk_candy_05_26_topreward

# restart later once scheduled
cfgrl_ethernet_05_26_adv_rvlm
cfgrl_ethernet_05_26_adv_rbm
cfgrl_ethernet_05_26_adv_topreward


CONFIG="cfgrl_ethernet_05_26_adv_topreward"
cd /gpfs/home/memmelma/projects/openpi_yam
export OPENPI_HOME="/gpfs/home/memmelma/projects/openpi_yam"
export OPENPI_DATA_HOME="/gpfs/scrubbed/memmelma/projects/openpi_yam/openpi"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py \
    $CONFIG \
      --exp-name $CONFIG \
      --num-train-steps 50000 \
      --resume

uv run scripts/compute_norm_stats.py --config-name=$CONFIG && uv run scripts/train.py \
