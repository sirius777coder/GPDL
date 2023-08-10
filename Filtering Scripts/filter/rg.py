#Rg filtering script.
#Originally from @weiting and rewritten by @Immortals on July 22, 2023.
import os
import time
import re
import argparse
import logging
import time
import shutil
import typing as T
from pathlib import Path
from typing import Tuple, List

import mdtraj as md

'''
Calculate Radius of Gyration of PDBs, compare them with reference and filter them through Rg threshold.
You can customize your reference either from a reference PDB or an exact value.

Usage:
[Required]
"-d", "--design_path": Folder containing the PDBs of which you want to compute Rg.
[Options]
"-r", "--reference_pdb": The PDB you want to compare calculated Rg with.
"--rg": The value you want to compare calculated Rg with. PDBs with Rg lower than this value will be kept, otherwise would be filtered out.
"--root-path": Root path of operation. Default = My own Rg filtering folder.  

Caution:
1. You should input "r" or "--rg", at least and only one of them.
2. Specifically, this script filters Rg of CA. If you want to use all atoms, just modify the script a little bit. 
3. When modifying this script, the root_path should contain a "/" as prefix.
Otherwise, it might create an absoulte path under your executing folder, which might be annoying.
'''

PathLike = T.Union[str, Path]

def radius_of_gyration(pdbfile:Path) -> tuple[float, float]:
    t = md.load(pdbfile)
    rg_Atoms = md.compute_rg(t)[0]
    top = md.load(pdbfile).topology
    CAs_index = top.select("backbone and name CA") #26
    t.restrict_atoms(CAs_index)  # this acts inplace on the trajectory
    rg_CA = md.compute_rg(t)[0]
    return rg_Atoms, rg_CA

def natural_sort_key(s: Path) -> List:
    # Helper function for natural sorting of file names
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculating Radius of Gyration")
    parser.add_argument("-d",
                        "--design_path", 
                        type=str, 
                        required=True, 
                        help="Path to the folder containing designed PDBs")
    parser.add_argument("-r", 
                        "--reference_pdb", 
                        type=str, 
                        help="Path to the reference PDB file")
    parser.add_argument("--rg",
                        type=float,
                        help="Value of Rg for filtering threshold.")
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/dssg/home/acct-clschf/clschf/zzq/ADS/filter/rg"),
        help= "Root path for file operations"
    )
        
    args = parser.parse_args()
    if args.reference_pdb and args.rg:
        parser.error("Cannot input both -r and --rg simultaneously! Please just choose one of them instead.")
    elif args.reference_pdb is None and args.rg is None:
        parser.error("Please input the Rg threshold you want.")
    
    design_path = os.path.normpath(args.design_path)
    root_path = args.root_path.resolve()
    native = args.reference_pdb
    rg_threshold = args.rg
    
    out_dir = os.path.join(root_path, 'results')
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(design_path)
    out_file = os.path.join(out_dir, f"rg_{basename}.txt")
    
    logging.basicConfig(level=logging.INFO, format = "%(asctime)s | %(levelname)s | %(message)s", datefmt="%d/%m/%y %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating Radius of Gyration from {design_path}......")
    
    filtered_dir = os.path.join(root_path, f"Rg_{basename}")
    os.makedirs(filtered_dir, exist_ok=True)
    
    sample_numbers = []
    
    with open(out_file, 'w') as out:
        out.write('Design\tRg_Atoms\tRg_CA\n')
        logger.info(f"Loaded {len(os.listdir(design_path))} designed PDBs from {design_path}")
        logger.info("Calculating......")
        if native:
            Rg_natural_Atoms, Rg_natural_CA = radius_of_gyration(args.reference_pdb)
            logger.info(f"The Rg value (all atoms) of native PDB is {Rg_natural_Atoms}, and the Rg value (CA) of native PDB is {Rg_natural_CA}.")
        for pdb_file in sorted(os.listdir(design_path), key = natural_sort_key):
            if pdb_file.endswith(".pdb"):
                pdb = pdb_file.strip()
                des_pdb = os.path.join(design_path, pdb)
                Rg_Atoms, Rg_CA = radius_of_gyration(des_pdb)
                out.write('%s\t%.3f\t%.3f\n' % (pdb, Rg_Atoms, Rg_CA))
                logger.info(f"Sample: {pdb}, Rg_CA = {Rg_CA}")
                
                if rg_threshold is not None:
                    if Rg_CA < rg_threshold:
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
                    if Rg_CA < Rg_natural_CA:
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
    
    filter_id_file = os.path.join(root_path, f"Rg_filterID_{basename}.txt")
    with open(filter_id_file, 'w') as f:
        f.write('\n'.join(map(str, sorted(sample_numbers))))
    f.close()    

