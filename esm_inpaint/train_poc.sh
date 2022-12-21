#!/bin/bash

python3 train_poc.py --output_folder parameters/ \
    --epochs 2 \
    --batch_size 16 \
    --max_length 300 \
    --lr 0.0001 \
    --project_entity poc_train_v1.0
 