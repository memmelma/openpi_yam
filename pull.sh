#!/bin/bash

# pull data
rsync -avzP yam:/home/reward/.cache/huggingface/lerobot/ /gpfs/scrubbed/memmelma/projects/openpi_yam/data/
# pull assets
rsync -avzP yam:/home/reward/Projects/openpi_yam/assets/ /gpfs/scrubbed/memmelma/projects/openpi_yam/assets/