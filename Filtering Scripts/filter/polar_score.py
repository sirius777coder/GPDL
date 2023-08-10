#Surface polar score filtering script.
#Originally from @weiting and rewritten by @Immortals on July 24, 2023.

import sys
import os
import logging
import time
import re 
import argparse
import shutil
import typing as T
from typing import List, Tuple
from pathlib import Path

import torch
import math
import mdtraj as md

'''
Calculate surface polar score of PDBs, compare them with reference and filter them through surface polar score threshold.
You can customize your reference either from a reference PDB or an exact value.

Usage:
[Required]
"-d", "--design_path": Folder containing the PDBs of which you want to compute surface polar score.
[Options]
"-r", "--reference_pdb": The PDB you want to compare calculated surface polar score with.
"-p", "--polar-score": The value you want to compare calculated surface polar score with. 
                       PDBs with scores lower than this value will be kept, otherwise would be filtered out.
"--root-path": Root path of operation. Default = My own surface polar score filtering folder.
"-c", "--calculate": Calculate the surface polar score of a single PDB file.

Caution:
1. You should input "r" or "--polar-score", at least and only one of them when you do the filtering work.
2. When modifying this script, the root_path should contain a "/" as prefix.
Otherwise, it might create an absoulte path under your executing folder, which might be annoying.
'''

PathLike = T.Union[str, Path]

parser = argparse.ArgumentParser(description= "Calculating surface polar score.")
parser.add_argument("-d",
                    "--design_path", 
                    type=str,  
                    help="Path to the folder containing designed PDBs")
parser.add_argument("-r", 
                    "--reference_pdb", 
                    type=str, 
                    help="Path to the reference PDB file")
parser.add_argument("-p",
                    "--polar-score",
                    type=float,
                    help="Value of surface polar score for filtering threshold.")
parser.add_argument("-c",
                    "--calculate",
                    type=str,
                    help="Calculate surface polar score for a single PDB")
parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/dssg/home/acct-clschf/clschf/zzq/ADS/filter/polar_score"),
        help= "Root path for file operations"
)

args = parser.parse_args()

if args.reference_pdb and args.polar_score:
    parser.error("Cannot input both -r and --polar-score simultaneously! Please just choose one of them instead.")
elif args.reference_pdb is None and args.polar_score is None and args.calculate is None:
    parser.error("Please input the surface polar score threshold you want.")

def load_pdb(pdb_file: Path) -> tuple[md.Topology, md.Trajectory]:
    top = md.load(pdb_file).topology
    t = md.load(pdb_file, top)
    return top, t

def calculate_backbone_neighbors(pdb_file: Path) -> torch.Tensor: 
    top, t = load_pdb(pdb_file)
    CBs_index = top.select("backbone and name C")
    CBs_coords = t.xyz[0, CBs_index,]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CBs_coords = torch.from_numpy(CBs_coords).to(device)
    CBs_dis = torch.norm(CBs_coords[:, None] - CBs_coords, dim=2, p=2)
    return CBs_dis

def polar_nopolar(pdb_file: Path) -> tuple[torch.Tensor, torch.Tensor]:
    top, t = load_pdb(pdb_file)
    residues = [str(residue) for residue in t.topology.residues]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    non_polar = torch.zeros(len(residues))
    non_polar = non_polar.to(device)
    polar = torch.zeros(len(residues))
    polar = polar.to(device)
    for i in range(len(residues)):
        if residues[i][:3] in ['ILE', 'LEU', 'MET', 'TRP', 'PHE', 'VAL']:
            non_polar[i] = 1
        if residues[i][:3] in ['SER', 'THR', 'TYR', 'ASN', 'GLN']:
            polar[i] = 1
    return non_polar, polar

def angle(pdb_file: Path) -> torch.Tensor:
    top, t = load_pdb(pdb_file)
    CBs_index = top.select("backbone and name C")
    CBs_coords = t.xyz[0, CBs_index,]
    CAs_index = top.select("backbone and name CA")
    CAs_coords = t.xyz[0, CAs_index,]
    CAs_CBs_coords = CBs_coords - CAs_coords
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CAs_CBs_coords = torch.from_numpy(CAs_CBs_coords).to(device)
    norm = torch.sqrt(torch.sum(torch.mul(CAs_CBs_coords, CAs_CBs_coords), dim=1))
    phi_ij = torch.zeros((CAs_CBs_coords.shape[0], CAs_CBs_coords.shape[0]))  # 求得夹角（弧度制）：
    phi_ij = phi_ij.to(device)
    for i in range(CAs_CBs_coords.shape[0]):
        for j in range(CAs_CBs_coords.shape[0]):
            if j > i:
                cos = torch.sum(torch.mul(CAs_CBs_coords[i,], CAs_CBs_coords[j,])) / (norm[i] * norm[j])
                if cos > 1:
                    cos = torch.tensor(([1]))
                if cos < -1:
                    cos = torch.tensor(([-1]))
                phi_ij[i, j] = torch.arccos(cos)
                phi_ij[j, i] = phi_ij[i, j]
    return phi_ij

def non_polar_polar(pdb_file: Path) -> float:
    CBs_dis = calculate_backbone_neighbors(pdb_file)
    non_polar, polar = polar_nopolar(pdb_file)
    phi_ij = angle(pdb_file)
    m = 1
    a = 0.5
    b = 2
    distance = 1 / (1 + torch.exp(CBs_dis - m))
    degree = ((torch.cos(math.pi - phi_ij) + a) / (1 + a)) ** b
    distance_degree = torch.mul(distance, degree)
    n_i = torch.sum(distance_degree - torch.diag_embed(torch.diag(distance_degree)), dim=1)
    polar_score = torch.sum(non_polar * (1-torch.sigmoid(n_i - torch.median(n_i)))) / torch.sum(
        1-torch.sigmoid(n_i - torch.median(n_i)))
    return polar_score

def natural_sort_key(s: Path) -> List:
    # Helper function for natural sorting of file names
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":
    
    if args.calculate is not None:
        polorScore = non_polar_polar(args.calculate)
        print(f"The polar score of {os.path.basename(os.path.normpath(args.calculate))} is {polorScore}.")
        logging.info(f"The polar score of {os.path.basename(os.path.normpath(args.calculate))} is {polorScore}.")
    else:
        design_path = os.path.normpath(args.design_path)
        root_path = args.root_path.resolve()
        native = args.reference_pdb
        polarScore_threshold = args.polar_score

        out_dir = os.path.join(root_path, 'results')
        os.makedirs(out_dir, exist_ok=True)
        basename = os.path.basename(design_path)
        out_file = os.path.join(out_dir, f"polarScore_{basename}.txt")

        logging.basicConfig(level=logging.INFO, format = "%(asctime)s | %(levelname)s | %(message)s", datefmt="%d/%m/%y %H:%M:%S")
        logger = logging.getLogger(__name__)
        logger.info(f"Calculating surface polar score from {design_path}......")

        filtered_dir = os.path.join(root_path, f"polarScore_{basename}")
        os.makedirs(filtered_dir, exist_ok=True)

        sample_numbers = []

        with open(out_file, 'w') as out:
            out.write('Design\tSurface Polar Score\n')
            logger.info(f"Loaded {len(os.listdir(design_path))} designed PDBs from {design_path}")
            logger.info("Calculating......")
            if native:
                polarScore_natural = non_polar_polar(args.reference_pdb)
                logger.info(f"The surface polar score of native PDB is {polarScore_natural}.")
            for pdb_file in sorted(os.listdir(design_path), key = natural_sort_key):
                if pdb_file.endswith(".pdb"):
                    pdb = pdb_file.strip()
                    des_pdb = os.path.join(design_path, pdb)
                    polarScore_designed = non_polar_polar(des_pdb)
                    out.write('%s\t%.3f\n' % (pdb, polarScore_designed))
                    logger.info(f"Sample: {pdb}, Surface Polar Score = {polarScore_designed}")
                    
                    if polarScore_threshold is not None:
                        if polarScore_designed <= polarScore_threshold:
                            logger.info("Kept.")
                            shutil.copy(des_pdb, filtered_dir)
                            sample_match = re.search(r'(\d+)\.pdb', pdb_file)
                            if sample_match:
                                sample_number = int(re.search(r'(\d+)\.pdb', pdb).group(1))
                                sample_numbers.append(sample_number)
                            else:
                                logger.warning(f"Could not find the sample number of pdb file. You might need to double check the file format.")
                        else:
                            logger.info("Filtered out.")    
                    elif native:                
                        if polarScore_designed <= polarScore_natural:
                            logger.info("Kept.")
                            shutil.copy(des_pdb, filtered_dir)
                            sample_match = re.search(r'(\d+)\.pdb', pdb_file)
                            if sample_match:
                                sample_number = int(re.search(r'(\d+)\.pdb', pdb).group(1))
                                sample_numbers.append(sample_number)
                            else:
                                logger.warning(f"Could not find the sample number of pdb file. You might need to double check the file format.")
                        else:
                            logger.info("Filtered out.")
        out.close()

        logger.info(f"Number of PDB files kept after filtering: {len(sample_numbers)}.")

        filter_id_file = os.path.join(root_path, f"polarScore_filterID_{basename}.txt")
        with open(filter_id_file, 'w') as f:
            f.write('\n'.join(map(str, sorted(sample_numbers))))
        f.close() 
