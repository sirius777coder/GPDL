#SAP filtering script.
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

from pyrosetta import *
from pyrosetta.rosetta import *
pyrosetta.init()

'''
Calculate SAP score of PDBs, and filter them out using specified threshold.

Usage:
[Required]
"-d", "--design_path": Folder containing the PDBs of which you want to compute SAP score.
[Options]
"-r", "--reference_pdb": The PDB you want to compare calculated SAP score with.
"--sap": The SAP filtering threshold. PDBs with SAP score lower than this value will be kept, otherwise would be filtered out.
"--root-path": Root path of operation. Default = My own SAP filtering folder.  
'''

PathLike = T.Union[str, Path]

def calculation_sap(pdb):
    pose = pyrosetta.pose_from_file(pdb)
    calc_sap = pyrosetta.rosetta.core.pack.guidance_scoreterms.sap.SapScoreMetric()
    sap = calc_sap.calculate(pose)
    return sap

def natural_sort_key(s: Path) -> List:
    # Helper function for natural sorting of file names
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculating SAP score.")
    parser.add_argument("-d",
                        "--design_path", 
                        type=str, 
                        required=True, 
                        help="Path to the folder containing designed PDBs")
    parser.add_argument("-r", 
                        "--reference_pdb", 
                        type=str, 
                        help="Path to the reference PDB file")
    parser.add_argument("--sap",
                        type=float,
                        help="Value of SAP for filtering threshold.")
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/dssg/home/acct-clschf/clschf/zzq/ADS/filter/sap"),
        help= "Root path for file operations"
)
        
    args = parser.parse_args()
    if args.reference_pdb and args.sap:
        parser.error("Cannot input both -r and --sap simultaneously! Please just choose one of them instead.")
    elif args.reference_pdb is None and args.sap is None:
        parser.error("Please input the SAP threshold you want.")
    
    design_path = os.path.normpath(args.design_path)
    root_path = args.root_path.resolve()
    native = args.reference_pdb
    sap_threshold = args.sap
    
    out_dir = os.path.join(root_path, 'results')
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(design_path)
    out_file = os.path.join(out_dir, f"sap_{basename}.txt")
    
    logging.basicConfig(level=logging.INFO, format = "%(asctime)s | %(levelname)s | %(message)s", datefmt="%d/%m/%y %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating SAP score from {design_path}......")
    
    filtered_dir = os.path.join(root_path, f"SAP_{basename}")
    os.makedirs(filtered_dir, exist_ok=True)
    
    sample_numbers = []
    
    with open(out_file, 'w') as out:
        out.write('Design\tSAP Score\n')
        logger.info(f"Loaded {len(os.listdir(design_path))} designed PDBs from {design_path}")
        logger.info("Calculating......")
        if native:
            sap_natural = calculation_sap(args.reference_pdb)
            logger.info(f"The SAP score of natural PDB is {sap_natural}.")
        for pdb_file in sorted(os.listdir(design_path), key = natural_sort_key):
            if pdb_file.endswith(".pdb"):
                pdb = pdb_file.strip()
                des_pdb = os.path.join(design_path, pdb)
                sap_designed = calculation_sap(des_pdb)
                out.write('%s\t%.3f\n' % (pdb, sap_designed))
                logger.info(f"Sample: {pdb}, SAP score = {sap_designed}")
                
                if sap_threshold is not None:
                    if sap_designed < sap_threshold:
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
                    if sap_designed < sap_natural:
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
    
    filter_id_file = os.path.join(root_path, f"sap_filterID_{basename}.txt")
    with open(filter_id_file, 'w') as f:
        f.write('\n'.join(map(str, sorted(sample_numbers))))
    f.close()    
    
    


