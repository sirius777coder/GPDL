import torch
import torch.nn as nn
import json,time
import customize_data
import utils

data_file = "/root/ESM-Inpainting/esm_inpaint/data/chain_set.jsonl"
dataset = customize_data.StructureDataset(data_file)
print(f"bad seq:{dataset.discard['bad_chars']}\ntoo long:{dataset.discard['too_long']}")
dataLoader = customize_data.StructureDataloader(dataset,64,num_workers=2)
output = next(iter(dataLoader))