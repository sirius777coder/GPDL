#!/bin/bash

python3 train_poc.py --output_folder parameters/ \
    --project_name "poc-esm-inpaint_v2" \
    --epochs 2 \
    --batch_size 4 \
    --max_length 300 \
    --lr 0.001 \
