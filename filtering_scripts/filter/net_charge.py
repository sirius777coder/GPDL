#Net charge filtering script.
#Originally from @weiting and rewritten by @Immortals on July 23, 2023.

import sys
import os
import logging
import time
import re 
import argparse
import shutil
import typing as T
from typing import List
from pathlib import Path

import mdtraj as md

'''
Calculate net charge of PDBs, and filter them out using specified threshold.

Usage:
[Required]
"-d", "--design_path": Folder containing the PDBs of which you want to calculate net charge.
[Options]
"-n", "--net-charge": The net charge filtering threshold, deafult = -1. PDBs with net charge lower than this value will be kept, otherwise would be filtered out.
"--root-path": Root path of operation. Default = My own net charge filtering folder.  
'''

PathLike = T.Union[str, Path]

def netcharge(pdbfile):
    t = md.load(pdbfile)
    arg, lys, asp, glu = 0, 0, 0, 0
    for residue in t.topology.residues:
        residue = str(residue)
        if residue[:3] == 'ARG':
            arg += 1
        if residue[:3] == 'LYS':
            lys += 1
        if residue[:3] == 'ASP':
            asp += 1
        if residue[:3] == 'GLU':
            glu += 1
    return arg + lys - asp - glu

def natural_sort_key(s: Path) -> List:
    # Helper function for natural sorting of file names
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculating net charge.")
    parser.add_argument("-d",
                        "--design_path", 
                        type=str, 
                        required=True, 
                        help="Path to the folder containing designed PDBs")
    parser.add_argument("-n",
                        "--net-charge",
                        type=float,
                        default= -1,
                        help="Value of net charge for filtering threshold.")
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/dssg/home/acct-clschf/clschf/zzq/ADS/filter/net_charge"),
        help= "Root path for file operations"
)
        
    args = parser.parse_args()
    
    design_path = os.path.normpath(args.design_path)
    root_path = args.root_path.resolve()
    nc_threshold = args.net_charge
    
    out_dir = os.path.join(root_path, 'results')
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(design_path)
    out_file = os.path.join(out_dir, f"netCharge_{basename}.txt")
    
    logging.basicConfig(level=logging.INFO, format = "%(asctime)s | %(levelname)s | %(message)s", datefmt="%d/%m/%y %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating net charge from {design_path}......")
    
    filtered_dir = os.path.join(root_path, f"netCharge_{basename}")
    os.makedirs(filtered_dir, exist_ok=True)
    
    sample_numbers = []
    
    with open(out_file, 'w') as out:
        out.write('Design\tNet Charge\n')
        logger.info(f"Loaded {len(os.listdir(design_path))} designed PDBs from {design_path}")
        logger.info("Calculating......")
        nc_threshold = args.net_charge
        for pdb_file in sorted(os.listdir(design_path), key = natural_sort_key):
            if pdb_file.endswith(".pdb"):
                pdb = pdb_file.strip()
                des_pdb = os.path.join(design_path, pdb)
                netcharge_designed = netcharge(des_pdb)
                out.write('%s\t%.3f\n' % (pdb, netcharge_designed))
                logger.info(f"Sample: {pdb}, Net charge = {netcharge_designed}")
                
            #if sap_threshold is not None:
                if netcharge_designed <= nc_threshold:
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
    
    filter_id_file = os.path.join(root_path, f"netCharge_filterID_{basename}.txt")
    with open(filter_id_file, 'w') as f:
        f.write('\n'.join(map(str, sorted(sample_numbers))))
    f.close()