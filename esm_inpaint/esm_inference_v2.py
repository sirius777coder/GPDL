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
import biotite.structure.io as strucio
from biotite.structure.residues import get_residues
from biotite.sequence import ProteinSequence

import customize_data
import modules

# 3.08 rewright the num of design function

parser = ArgumentParser(description='ESM inpainting argparse')

parser.add_argument('--inpaint_seq', type=str,
                    help='Inpainting sequences format')
parser.add_argument('-i', '--input', type=str, help='input pdb path')
parser.add_argument('-o', '--output_prefix', type=str, help='output prefix')
parser.add_argument('--mask_aa', type=str, default="A",
                    help="input special mask aa to run esm_ip")
parser.add_argument('--fpath', type=str, default="./loc.txt",
                    help="fpath to note the sequence location of motif in each design")
parser.add_argument('-n', '--num_design', type=int,
                    default=5, help="num of designs")
parser.add_argument('-T', '--sample_steps', type=int,
                    default=1, help="num of designs")
args = parser.parse_args()

print(f"{torch.cuda.get_device_name(torch.cuda.current_device())}")

# model = esm.pretrained.esmfold_v1()

# Reading the data file and initialize the esm-inpainting class
model_path = "/lustre/home/acct-stu/stu005/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt"
# 读取一个pickle文件为一个dict
model_data = torch.load(str(model_path), map_location="cuda:0")
cfg = model_data["cfg"]["model"]
model = modules.esm_inpaint(cfg, chunk_size=64)  # make an instance
model_state = model_data["model"]
model.esmfold.load_state_dict(model_state, strict=False)
model.load_inpaint_dict("/lustre/home/acct-stu/stu005/ESM-Inpainting/esm_inpaint/checkpoints/inpaint_weight_11.pt")
model = model.eval().cuda()


# input : --inpaint_seq A1-3,A4,A6,B10-100
for design in range(args.num_design):

    inapint_info = []
    motif_mask = ""  # ["1111"] 1 unmasked, 0 masked

    # parsing the inpaint_seq
    segment = (args.inpaint_seq).split(",")

    for i in range(len(segment)):
        # scaffold region
        if segment[i][0] not in [chr(ord('a')+_) for _ in range(26)] and segment[i][0] not in [chr(ord('A')+_) for _ in range(26)]:
            if "-" in segment[i]:
                a, b = segment[i].split("-")
                a, b = int(a), int(b)
                if a == 0:
                    a = 1
                scaffold = np.random.randint(a, b+1)
            else:
                scaffold = int(segment[i])
            motif_mask += "0" * scaffold
            inapint_info.append({"mask": scaffold})
            # 1 meaning position is unmasked motif and 0 meaning position is masked scaffold.
        else:  # motif region
            chain = segment[i][0]
            if "-" in segment[i]:
                start, end = (segment[i][1:]).split("-")
                start = int(start)
                end = int(end)
                length = end-start+1
            else:
                start = end = int(segment[i][1:])
                length = 1
            motif_mask += "1" * length
            inapint_info.append({f"{chain}": [start, end]})

    # load the input file by biotite (only standard aa will in this AtomArray)
    structure = utils.load_structure(args.input)
    print(structure[structure.res_id == 92])
    # preliminaries
    inpaint_seq = ""
    inpaint_coord = np.zeros((len(motif_mask), 4, 3))
    location = 0
    print(inapint_info)

    # inpaint_info : [{'mask': 9}, {'A': [119, 140]}, {'mask': 18}, {'A': [63, 82]}, {'mask': 28}]
    for item in inapint_info:
        if list(item.keys())[0] == "mask":  # mask region (scaffold region)
            inpaint_seq += args.mask_aa * item['mask']
            location += item['mask']
        else:  # motif region (fix to some coordinates)
            chain_name = list(item.keys())[0]
            start, end = int(item[chain_name][0]), int(item[chain_name][1])
            for res_id in range(start, end+1):
                res_atom_array = structure[(structure.chain_id == chain_name) & (
                    structure.res_id == res_id)]
                res_name = ProteinSequence.convert_letter_3to1(
                    get_residues(res_atom_array)[1][0])
                inpaint_seq += res_name
                print(res_name)
                print(res_atom_array)
                inpaint_coord[location][0] = res_atom_array[res_atom_array.atom_name == "N"].coord[0]
                inpaint_coord[location][1] = res_atom_array[res_atom_array.atom_name == "CA"].coord[0]
                inpaint_coord[location][2] = res_atom_array[res_atom_array.atom_name == "C"].coord[0]
                inpaint_coord[location][3] = res_atom_array[res_atom_array.atom_name == "O"].coord[0]
                location += 1

    seq = torch.tensor([utils.restype_order[i] for i in inpaint_seq],
                       dtype=torch.long).unsqueeze(0).to("cuda:0")
    coord = (torch.from_numpy(inpaint_coord).to(
        torch.float)).unsqueeze(0).to("cuda:0")

    print(f"seq:{len(seq)},coord:{len(coord)},mask:{len(motif_mask)}")
    with torch.no_grad():
        output = model.infer(coord, seq, T=args.sample_steps, motif_mask=torch.tensor(
            [int(i) for i in list(motif_mask)], device=coord.device).unsqueeze(0))

    print(f"{design}\n{motif_mask}\n")
    with open(f"{args.output_prefix}_{design}.pdb", "w") as f:
        f.write(output[0])

    # simultaneously note the sequence location of each design
    with open(f"{args.fpath}", "a") as f:
        f.write(f"{design}\n{inapint_info}\n")
