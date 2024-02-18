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

# model = esm.pretrained.esmfold_v1()

import subprocess

def get_nvidia_driver_version():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"]).decode("utf-8")
        return output.strip()
    except Exception as e:
        print("Error:", e)
        return
# driver_version = get_nvidia_driver_version()

# Check if the driver_version is not None, which means the function executed successfully
# if driver_version:
#     print(f"NVIDIA GPU driver version: {driver_version}")
# else:
#     print("Unable to retrieve the driver version.")
print(f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
# Reading the data file and initialize the esm-inpainting class
model_path =  os.path.expanduser("~/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())
# 读取一个pickle文件为一个dict
model_data = torch.load(str(model_path), map_location=device)
cfg = model_data["cfg"]["model"]
model = modules.esm_inpaint(cfg, chunk_size=64)  # make an instance
model_state = model_data["model"]
model.esmfold.load_state_dict(model_state, strict=False)
model.load_inpaint_dict("checkpoints/inpaint_weight_11.pt")
model = model.to(device)
model = model.eval()


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
    # preliminaries
    inpaint_seq = ""
    inpaint_coord = np.zeros((len(motif_mask), 4, 3))
    location = 0

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
                inpaint_coord[location][0] = res_atom_array[res_atom_array.atom_name == "N"].coord[0]
                inpaint_coord[location][1] = res_atom_array[res_atom_array.atom_name == "CA"].coord[0]
                inpaint_coord[location][2] = res_atom_array[res_atom_array.atom_name == "C"].coord[0]
                inpaint_coord[location][3] = res_atom_array[res_atom_array.atom_name == "O"].coord[0]
                location += 1

    seq = torch.tensor([utils.restype_order[i] for i in inpaint_seq],
                       dtype=torch.long).unsqueeze(0).to(device)
    coord = (torch.from_numpy(inpaint_coord).to(
        torch.float)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.infer(coord, seq, T=args.sample_steps, motif_mask=torch.tensor(
            [int(i) for i in list(motif_mask)], device=coord.device).unsqueeze(0))
    with open(f"{args.output_prefix}_{design}.pdb", "w") as f:
        f.write(output[0])

    # simultaneously note the sequence location of each design
    with open(f"{args.fpath}", "a") as f:
        f.write(f"{design}\n{inapint_info}\n")
