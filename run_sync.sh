#!/bin/bash
rsync -avzP yam:/home/reward/Projects/openpi_yam/checkpoints/ /gpfs/scrubbed/memmelma/projects/openpi_yam/checkpoints/
rsync -avzP yam:/home/reward/.cache/huggingface/lerobot/ /gpfs/scrubbed/memmelma/projects/openpi_yam/data/
rsync -avzP yam:/home/reward/Projects/openpi_yam/assets/ /gpfs/scrubbed/memmelma/projects/openpi_yam/assets/