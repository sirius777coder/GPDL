# v1.0 
# date   : 12.19
# Author : Bo Zhang
# Content: 
#  - Single V100 GPU for poc (5 epochs, batch_size = 1)
#  - except multi GPU part


import json,time
import customize_data
import utils

import json
import time
import os
import sys
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


data_file = "/root/ESM-Inpainting/esm_inpaint/data/chain_set.jsonl"
dataset = customize_data.StructureDataset(data_file)

# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
with open(args.file_splits) as f:
    dataset_splits = json.load(f)
train_set, validation_set, test_set = [
    Subset(dataset, [
        dataset_indices[chain_name] for chain_name in dataset_splits[key] 
        if chain_name in dataset_indices
    ])
    for key in ['train', 'validation', 'test']
]
loader_train, loader_validation, loader_test = [data.StructureLoader(
    d, batch_size=args.batch_tokens
) for d in [train_set, validation_set, test_set]]
print('Training:{}, Validation:{}, Test:{}'.format(len(train_set),len(validation_set),len(test_set)))
dataLoader = customize_data.StructureDataloader(dataset,64,num_workers=2)
output = next(iter(dataLoader))