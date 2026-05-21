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