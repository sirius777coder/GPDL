#Filter PDBs based on global-RMSD with a reference one.
#Written by @Immortals on August 2nd, 2023.

import sys
import os
import logging
import time
import re 
import argparse
import shutil
import warnings
import numpy as np
import typing as T
from typing import List
from pathlib import Path

from Bio.PDB import *
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBParser import PDBConstructionWarning as BiopythonWarning

'''
Calculating global-RMSD and filter them out.
[Required]
"-d", "--design_path": Path to the folder containing designed PDBs.
"-r", "--reference_pdb": Path to the reference PDB.
[Optional]
"--rmsd": "--rmsd": The filtering threshold of global-RMSD, default = 1.5
"--root-path": Root path of operation. Default = My own RMSD filtering folder.  
'''

PathLike = T.Union[str, Path]
warnings.filterwarnings("ignore", category=BiopythonWarning)

parser = argparse.ArgumentParser(description= "Calculating global-RMSD between a reference PDB and a set of given structures.")
parser.add_argument("-d",
                    "--design_path", 
                    type=str, 
                    required=True, 
                    help="Path to the folder containing designed PDBs")
parser.add_argument("-r", 
                    "--reference_pdb", 
                    type=str, 
                    required=True,
                    help="Path to the reference PDB file")
parser.add_argument("--rmsd",
                    type=float,
                    default = 1.5,
                    help="Value of global-RMSD for filtering threshold.")
parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/dssg/home/acct-clschf/clschf/zzq/ADS/filter/RMSD"),
        help= "Root path for file operations"
)

args = parser.parse_args()

def coarse_rmsd(tgt_pdb, ref_pdb, option="bb"):
    """
    注意这个脚本没能解决不同的原子顺序产生的影响,理论上来说pdb里面每个氨基酸的原子摆放顺序是有规律的即N,CA,C,O
    """
    paser = PDBParser()
    structure_ref = paser.get_structure("ref", ref_pdb)
    structure_tgt = paser.get_structure("des", tgt_pdb)
    coord_ref = []
    coord_tgt = []
    if option == "full":
        for atom_ref in structure_ref.get_atoms():
            coord_ref.append(atom_ref.get_coord())
        for atom_tgt in structure_tgt.get_atoms():
            coord_tgt.append(atom_tgt.get_coord())
    elif option == "CA":
        for atom_ref in structure_ref.get_atoms():
            if atom_ref.get_name() == "CA":
                coord_ref.append(atom_ref.get_coord())
        for atom_tgt in structure_tgt.get_atoms():
            if atom_tgt.get_name() == "CA":
                coord_tgt.append(atom_tgt.get_coord())
    elif option == "bb":
        for atom_ref in structure_ref.get_atoms():
            if atom_ref.get_name() == "CA" or atom_ref.get_name() == "N" or atom_ref.get_name() == "C" or atom_ref.get_name() == "O":
                coord_ref.append(atom_ref.get_coord())
        for atom_tgt in structure_tgt.get_atoms():
            if atom_tgt.get_name() == "CA" or atom_tgt.get_name() == "N" or atom_tgt.get_name() == "C" or atom_tgt.get_name() == "O":
                coord_tgt.append(atom_tgt.get_coord())
    coord_ref = np.array(coord_ref)
    coord_tgt = np.array(coord_tgt)
    print(coord_ref.shape)
    print(coord_tgt.shape)
    sup = SVDSuperimposer()
    sup.set(coord_ref, coord_tgt)
    sup.run()
    rmsd = sup.get_rms()
    return rmsd

def natural_sort_key(s: Path) -> list:
    # Helper function for natural sorting of file names
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":   
    design_path = os.path.normpath(args.design_path)
    root_path = args.root_path.resolve()
    native = args.reference_pdb
    rmsd_threshold = args.rmsd
    
    out_dir = os.path.join(root_path, 'results')
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(design_path)
    out_file = os.path.join(out_dir, f"globalRMSD_{basename}.txt")
    
    logging.basicConfig(level=logging.INFO, format = "%(asctime)s | %(levelname)s | %(message)s", datefmt="%d/%m/%y %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    logger.info(f"Calculating global-RMSD from {design_path}")
    
    filtered_dir = os.path.join(root_path, f"globalRMSD_{basename}")
    os.makedirs(filtered_dir, exist_ok= True)
    
    sample_numbers = []

    with open(out_file, 'w') as out:
        out.write('design\tESM_RMSD\n')
        logger.info(f"Loaded {len(os.listdir(design_path))} designed PDBs from {design_path}")
        logger.info(f"The reference PDB is {native}.")
        logger.info("Calculating......")
        for pdb_file in sorted(os.listdir(design_path), key=natural_sort_key):
            if pdb_file.endswith(".pdb"):
                pdb = pdb_file.strip()
                des_pdb = os.path.join(design_path, pdb)
                rmsd = coarse_rmsd(des_pdb, native)
                out.write('%s\t%.3f\n' % (pdb, rmsd))
                logger.info(f"Sample: {pdb}, global-RMSD = {rmsd}")
                
                if rmsd <= rmsd_threshold:
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

    logger.info(f"Number of PDB files kept after filtering: {len(sample_numbers)}")

    # Write the sample numbers to a new text file
    filter_id_file = os.path.join(root_path, f"filterID_{basename}.txt")
    with open(filter_id_file, 'w') as f:
        f.write('\n'.join(map(str, sorted(sample_numbers))))
    f.close()