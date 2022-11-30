from Bio.PDB import *
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description='ESM inpainting argparse')

parser.add_argument('--inpaint_seq', type=str,
                    help='Inpainting sequences format')
parser.add_argument('-i', '--input', type=str, help='input pdb path')
parser.add_argument('-o', '--output_prefix', type=str, help='output prefix')
parser.add_argument('--mask_aa', type=str, default="G",
                    help="input special mask aa to run esm_ip")
parser.add_argument('-n', '--num_design', type=int,
                    default=5, help="num of designs")
args = parser.parse_args()

alphabet = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
    'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
    'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
}

# input : --inpaint_seq A1-3,A4,A6,B10-100

inapint_info = []
mask_seq = ""  # 1 unmasked, 0 masked

# parsing the inpaint_seq
segment = (args.inpaint_seq).split(",")
for i in segment:
    if segment[i][0] not in [chr(ord('a')+_) for _ in range(26)] and segment[i][0] not in [chr(ord('A')+_) for _ in range(26)]:
        mask_seq += "0" * int(segment[i])
        inapint_info.append({"mask": int(segment[i])})
        # Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
    else:
        chain = segment[i][0]
        start, end = (segment[i][1:]).split("-")
        start = int(start)
        end = int(end)
        length = end-start+1
        mask_seq += "1" * length
        inapint_info.append({f"{chain}": [start, end]})


# parsing the pdb files
inpaint_seq = ""
inpaint_coord = np.zeros((len(mask_seq), 4, 3))

parser = PDBParser()
structure = parser.get_structure("esm_inpiant", args.input)
location = 0
for item in inapint_info:
    if item.keys()[0] == "mask":
        inpaint_seq += args.mask_aa * item['mask']
        location += item['mask'][0]
    else:
        chain_name = item.keys()[0]
        for res_id in range(start=item[chain_name][0], end=item[chain_name][1]+1):
            res = structure[chain_name][res_id]
            res_name = alphabet[res.get_resname()]
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
