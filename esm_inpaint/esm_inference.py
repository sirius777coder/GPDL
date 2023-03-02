import torch
import torch.nn as nn
import json
import time
import customize_data
import utils
import numpy as np
import esm.esmfold.v1.esmfold as ESM
import modules
import os
from openfold.utils.rigid_utils import Rigid


import json
import time
import os
import sys
import shutil
import wandb
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
import biotite
import biotite.structure as struc
from Bio.PDB import *

import customize_data
import modules


parser = ArgumentParser(description='ESM inpainting argparse')

parser.add_argument('--inpaint_seq', type=str,
                    help='Inpainting sequences format')
parser.add_argument('-i', '--input', type=str, help='input pdb path')
parser.add_argument('-o', '--output_prefix', type=str, help='output prefix')
parser.add_argument('--mask_aa', type=str, default="A",
                    help="input special mask aa to run esm_ip")
parser.add_argument('-n', '--num_design', type=int,
                    default=5, help="num of designs")
parser.add_argument('-T', '--sample steps', type=int,
                    default=1, help="num of designs")
args = parser.parse_args()


# model = esm.pretrained.esmfold_v1()

# Reading the data file and initialize the esm-inpainting class
model_path = "/root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"
# 读取一个pickle文件为一个dict
model_data = torch.load(str(model_path), map_location="cuda:0")
cfg = model_data["cfg"]["model"]
model = modules.esm_inpaint(cfg, chunk_size=64)  # make an instance
model_state = model_data["model"]
model.esmfold.load_state_dict(model_state, strict=False)
model.load_inpaint_dict("./inpaint_weight_0.pt")
model = model.eval().cuda()


# input : --inpaint_seq A1-3,A4,A6,B10-100

# structure = utils.load_structure(args.input)

inapint_info = []
motif_mask = ""  # 1 unmasked, 0 masked

# parsing the inpaint_seq
segment = (args.inpaint_seq).split(",")
for i in range(len(segment)):
    if segment[i][0] not in [chr(ord('a')+_) for _ in range(26)] and segment[i][0] not in [chr(ord('A')+_) for _ in range(26)]:
        motif_mask += "0" * int(segment[i])
        inapint_info.append({"mask": int(segment[i])})
        # Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
    else:
        chain = segment[i][0]
        start, end = (segment[i][1:]).split("-")
        start = int(start)
        end = int(end)
        length = end-start+1
        motif_mask += "1" * length
        inapint_info.append({f"{chain}": [start, end]})


structure = utils.load_structure(args.input)
coords, seq = utils.extract_coords_from_structure(structure, pattern="max")

# parsing the pdb files
inpaint_seq = ""
inpaint_coord = np.zeros((len(motif_mask), 4, 3))

parser = PDBParser()
structure = parser.get_structure("esm_inpiant", args.input)
structure = structure[0]
location = 0
print(inapint_info)
for item in inapint_info:
    if list(item.keys())[0] == "mask":
        inpaint_seq += args.mask_aa * item['mask']
        location += item['mask']
    else:
        chain_name = list(item.keys())[0]
        for res_id in range(item[chain_name][0], item[chain_name][1]+1):
            res = structure[chain_name][res_id]
            res_name = utils.alphabet[res.get_resname()]
            inpaint_seq += res_name
            N = list(res["N"].get_vector())
            CA = list(res["CA"].get_vector())
            C = list(res["C"].get_vector())
            O = list(res["O"].get_vector())
            inpaint_coord[location][0] = np.array(N)
            inpaint_coord[location][1] = np.array(CA)
            inpaint_coord[location][2] = np.array(C)
            inpaint_coord[location][3] = np.array(O)
            location += 1

print(inpaint_seq)
seq = torch.tensor([utils.restype_order[i] for i in inpaint_seq],
                   dtype=torch.long).unsqueeze(0).to("cuda:0")
coord = (torch.from_numpy(inpaint_coord).to(
    torch.float)).unsqueeze(0).to("cuda:0")

output = model.infer(coord, seq, T=args.T, motif_mask="")
for i in range(len(motif_mask)):
    if motif_mask[i] == "1":
        output['aatype'][0][i] = seq[0][i]
utils.output_to_pdb(output['positions'], output['aatype'],
                    output['plddt'], file_path=f"{args.output_prefix}.pdb")


# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)

# sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
# Multimer prediction can be done with chains separated by ':'

# with torch.no_grad():
#     output = model.esmfold.infer_pdb(sequence)

# with open("result.pdb", "w") as f:
#     f.write(output)

# import biotite.structure.io as bsio
# struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
# print(struct.b_factor.mean())  # this will be the pLDDT

# seq = torch.tensor([utils.restype_order[i] for i in sequence]).unsqueeze(0).to("cuda:0")


# dataset = customize_data.StructureDataset("/data/users/zb/data/chain_set.jsonl",max_length=200)
#     # Split the dataset
# dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
# with open("/data/users/zb/data/splits.json") as f:
#     dataset_splits = json.load(f)
# train_set, validation_set, test_set = [
#     Subset(dataset, [
#         dataset_indices[chain_name] for chain_name in dataset_splits[key]
#         if chain_name in dataset_indices
#     ])
#     for key in ['train', 'validation', 'test']
# ]
# loader_train, loader_validation, loader_test = [customize_data.StructureDataloader(
#         d, batch_size=1
#     ) for d in [train_set, validation_set, test_set]]
# batch = next(iter(loader_train))
# print(batch['seq'])
# print(batch['motif_mask'])
# output1=model(batch['coord'].to("cuda:0"),batch['seq'].to("cuda:0"),bert_mask_structure=batch['bert_mask_structure'])
# utils.output_to_pdb(output1['positions'],output1['aatype'],output1['plddt'],file_path="poc_native_0.pdb")
# output2=model(batch['coord'].to("cuda:0"),batch['motif_mask'].to("cuda:0"),bert_mask_structure=batch['bert_mask_structure'])
# utils.output_to_pdb(output2['positions'],output2['aatype'],output2['plddt'],file_path="poc_mask_0.pdb")
