uv run scripts/convert_raiden_to_lerobot_joint.py \
    --manifest datasets_manifest_05_20.yaml \
    --repo-id memmelma/swb_joint_05_20

uv run scripts/annotate_rewards_joint.py \
    --repo-id memmelma/swb_joint_05_20 \
    --reward-model rvlm \
    --reference-instruction "move the star wars book from the book shelf to the gray box" \
    --gamma 0.99 --beta 2.0 --compute-delta
# beta is only used to precompute the AWR weights for vis purposes

uv run scripts/annotate_rewards_joint.py \
    --repo-id memmelma/swb_joint_05_20 \
    --reward-model success \
    --reference-instruction "move the star wars book from the book shelf to the gray box" \
    --gamma 0.99 --beta 2.0 --compute-delta


## 
uv run scripts/convert_raiden_to_lerobot_joint.py \
    --manifest datasets_bimanual_05_20.yaml \
    --repo-id memmelma/swb_bimanual_05_20

uv run scripts/annotate_rewards_joint.py \
    --repo-id memmelma/swb_bimanual_05_20 \
    --reward-model success \
    --reference-instruction "move the star wars book from the left gray box to the right gray box" \
    --gamma 0.99 --beta 2.0 --compute-delta

## CFGRL training (π*0.6 advantage-conditioned policy extraction)

Enable in DataConfig:
```python
reward_name = "<prefix>_reward"   # must contain "reward"; stores V(s_t) estimates
cfgrl_enabled = True
cfgrl_positive_quantile = 0.30    # paper: 0.30 pre-train, 0.40 fine-tune
cfgrl_dropout_prob = 0.30
cfgrl_force_positive = False      # set True for SFT / pure-demo fine-tune phase
```

At **inference**, append `"\nAdvantage: positive"` to the task prompt to sample
from the improved policy π̂ (β=1). Example:
```python
prompt = "move the star wars book from the book shelf to the gray box\nAdvantage: positive"
```
Without the suffix the policy samples unconditionally (behavior cloning baseline).

Smoke test (500 steps):
```bash
CONFIG=cfgrl_swb_local_smoke
uv run scripts/compute_norm_stats.py --config-name=$CONFIG
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG --exp-name $CONFIG --overwrite
```
Look for log line `CFGRL: threshold=...` at dataset init.