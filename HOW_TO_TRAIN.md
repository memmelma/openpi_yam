# LoRA fine-tuning pi05 policies w/ CFGRL

## conversion

### convert dataset raiden (raiden raw -> processed)
```
rd convert
```

### convert dataset raiden (raiden -> huggingface)
```
USER=your huggingface user
DATASET=set datset name of processed raiden dataset

uv run scripts/convert_raiden_to_lerobot_joint.py \
    --manifest manifest.yaml \
    --repo-id $USER/$DATASET
```

above requires a ```manifest.yaml``` containing dataset path and optimality, for example:

```
datasets:
    - path: /home/reward/Projects/raiden/data/processed/candy_bowl_optimal
      optimality: optimal
    - path: /home/reward/Projects/raiden/data/processed/candy_bowl_suboptimal
      optimality: suboptimal
```

### annotate dataset

```
DATASET=dataset name on huggingface, e.g., memmelma/block_stacking or  $USER/$DATASET from above
LANGUAGE_INSTRUCTION=language instruction used for reward annotation
    
# annotate w/ our method
uv run scripts/annotate_rewards_joint.py \
    --repo-id $DATASET \
    --reward-model rvlm \
    --subsample_factor 30 \
    --video_logging \
    --camera head \
    --reference-instruction "$LANGUAGE_INSTRUCTION"

# activate envs for baselines
deactivate
cd ~/Projects/annotate_rewards
source .venv/bin/activate
cd ~/Projects/openpi_yam

# annotate w/ robometer
python scripts/annotate_rewards_joint.py \
    --repo-id $DATASET \
    --reward-model rbm \
    --model-path aliangdw/Robometer-4B \
    --camera head \
    --subsample_factor 30 \
    --overwrite \
    --reference-instruction "$LANGUAGE_INSTRUCTION"

# annotate w/ topreward
python scripts/annotate_rewards_joint.py \
    --repo-id $DATASET \
    --reward-model topreward \
    --camera head \
    --subsample_factor 30 \
    --reference-instruction "$LANGUAGE_INSTRUCTION" \
    --overwrite \
    --batched


# annotate w/ binary success (for BC)
uv run scripts/annotate_rewards_joint.py \
    --repo-id $DATASET \
    --reward-model success \
    --reference-instruction "$LANGUAGE_INSTRUCTION"
```

### uploading huggingface/lerobot datasets

#### login
```
cd openpi_yam
source .venv/bin/activate
huggingface-cli login
```

#### upload dataset and tag w/ version
```
DATASET="memmelma/ood_05_27_high_res"

huggingface-cli upload \
  $DATASET \
  $HF_LEROBOT_HOME/$DATASET \
  --repo-type dataset

huggingface-cli tag $DATASET v2.1 --repo-type dataset --delete
huggingface-cli tag $DATASET v2.1 --repo-type dataset
```

## training

### generate a config
```
    TrainConfig(
        ### CHANGE THIS TO RUN NAME ###
        name="CONFIG",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora"),
        data=LeRobotYAMDataConfig(
            ### CHANGE THIS TO HF DATASET ###
            repo_id="memmelma/ethernet_05_26",
            base_config=DataConfig(
                prompt_from_task=True,
                ### CHANGE THIS TO REWARD NAME FROM ANNOTATION ###
                reward_name="place_the_computer_in_the_shelf_and_unplug_the_ethernet_cable_rvlm_reward",
                override_prompt_from_reward=True,
                cfgrl_enabled=True,
                cfgrl_positive_quantile=0.30,
                cfgrl_dropout_prob=0.30,
                cfgrl_force_positive=False,
                weight_scheme="adv_chunk",
                weight_quantile=None,
                weight_cutoff=None,
                ### CHANGE THIS OR REMOVE (ON YAM MACHINE) ###
                lerobot_home="/gpfs/scrubbed/memmelma/projects/openpi_yam/data",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        ### CHANGE THIS OR REMOVE (ON YAM MACHINE) ###
        checkpoint_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints",
        ### CHANGE THIS OR REMOVE (ON YAM MACHINE) ###
        assets_base_dir="/gpfs/scrubbed/memmelma/projects/openpi_yam/assets",
        num_train_steps=50_000,
        batch_size=32,
        num_workers=8,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
```

add the above to ```openpi_yam/src/openpi/training/config.py``` for training AND inference

### compute normalization states (per config) 
```
CONFIG= # as set above in the config
uv run scripts/compute_norm_stats.py --config-name=$CONFIG
```

### start training
```
CONFIG= # as set above in the config

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    $CONFIG \
      --exp-name star_wars \
      --num-train-steps 25000 \
      --overwrite
      #--resume
```