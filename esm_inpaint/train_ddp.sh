#!/bin/bash

# python3 -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="192.168.0.1" --master_port=12345 train_poc.py 
python3 -m torch.distributed.launch --nproc_per_node=4 train_poc.py
