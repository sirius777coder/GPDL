import torch
import torch.nn as nn
import json,time
import customize_data
import utils
import numpy as np
import esm.esmfold.v1.esmfold as ESM
import modules 
import os
from openfold.utils.rigid_utils import Rigid


import json,time,os,sys,shutil,wandb
import utils
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import customize_data
import modules

# model = esm.pretrained.esmfold_v1()

# Reading the data file and initialize the esm-inpainting class
model_path = "/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"
model_data = torch.load(str(model_path), map_location="cuda:0") #读取一个pickle文件为一个dict
cfg = model_data["cfg"]["model"]
model = modules.esm_inpaint(cfg, chunk_size=64)  # make an instance
model_state = model_data["model"]
model.esmfold.load_state_dict(model_state, strict=False)




model = model.eval().cuda()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
# Multimer prediction can be done with chains separated by ':'

# with torch.no_grad():
#     output = model.esmfold.infer_pdb(sequence)

# with open("result.pdb", "w") as f:
#     f.write(output)

# import biotite.structure.io as bsio
# struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
# print(struct.b_factor.mean())  # this will be the pLDDT

seq = torch.tensor([utils.restype_order[i] for i in sequence]).unsqueeze(0).to("cuda:0")



dataset = customize_data.StructureDataset("/data/users/zb/data/chain_set.jsonl",max_length=200)
    # Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
with open("/data/users/zb/data/splits.json") as f:
    dataset_splits = json.load(f)
train_set, validation_set, test_set = [
    Subset(dataset, [
        dataset_indices[chain_name] for chain_name in dataset_splits[key] 
        if chain_name in dataset_indices
    ])
    for key in ['train', 'validation', 'test']
]
loader_train, loader_validation, loader_test = [customize_data.StructureDataloader(
        d, batch_size=1
    ) for d in [train_set, validation_set, test_set]]
batch = next(iter(loader_train))
print(batch['seq'])
print(batch['mask_seq'])
output1=model(batch['coord'].to("cuda:0"),batch['seq'].to("cuda:0"),bert_mask_structure=batch['bert_mask_structure'])
utils.output_to_pdb(output1['positions'],output1['aatype'],output1['plddt'],file_path="poc_native_0.pdb")
output2=model(batch['coord'].to("cuda:0"),batch['mask_seq'].to("cuda:0"),bert_mask_structure=batch['bert_mask_structure'])
utils.output_to_pdb(output2['positions'],output2['aatype'],output2['plddt'],file_path="poc_mask_0.pdb")
