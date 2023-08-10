#self-consistency RMSD calculating script
#Written by @Immortals on August 10th, 2023
#A first try on MDAnalysis package (for )
 
import argparse
import logging
import re
import typing as T
from pathlib import Path
from typing import List

import torch
import MDAnalysis as mda
from MDAnalysis.analysis import rms

'''
Calculating self-consistency global-CA-RMSD between two ESMFold and AlphaFold2 predictied structures.
This is based on the propose that low sc-RMSD raises better folding ability or solubility.
Based on MDAnalysis package.
The script do following things:
1. Calculate global-CA-RMSD between PDBs with the same {sample_number} as profix;
2. Write the results to a .txt file;
3. If "--threshold" is provided, the filtering IDs would be writeen into another .txt file.
This is a single step in filtering pipeline and should be used combined with other wrappers in this pipeline.

Usage:
[Required]
"-i", "--input": Folder containing Group 1 PDBs (In this case, ESM-PDBs)
"-r", "--reference": Folder containing Group 2 PDBs (In this case, AF2-PDBs)
[Optional]
"-o", "--output": (Strongly recommend to use this flag) The file path of result txt file. 
"-t", "--threshold": Filtering threshold. PDBs below this value would be kept, otherwise filtered out.
"--root-path": Root path of operation. Default = My own Rg filtering folder.  
'''

def CA_rmsd(esm:Path, af2:Path) -> float:
    u1 = mda.Universe(esm)
    u2 = mda.Universe(af2)
    rmsd = rms.rmsd(u1.select_atoms('backbone').positions,
                    u2. select_atoms('backbone').positions,
                    center=True,
                    superposition=True)
    return rmsd

def natural_sort_key(s: Path) -> List:
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s.name)
    ]
    
def create_parser():
    parser = argparse.ArgumentParser(description="Calculating Sequence Recovery")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input PDBs' folder",
    )
    parser.add_argument(
        "-o", "--output", type=Path, 
        default= 'self_consistency_RMSD.txt',
        help="Output file name"
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=True,
        help="Path to the reference PDBs' folder",
    )
    parser.add_argument(
        "-t", 
        "--threshold", 
        type=float, 
        help="sc-RMSD threshold"
    )
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/dssg/home/acct-clschf/clschf/zzq/ADS/filter/scRMSD"),
        help= "Root path for file operations",
    )
    return parser

def run(args):
    root_path = args.root_path.resolve()
    input_path = args.input.resolve()
    reference_path = args.reference.resolve()
    output_file = args.output.resolve()
    scRMSD_threshold = args.threshold
    basename = input_path.name
    
    results = []
    esm_dict = {}
    af2_dict = {}
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating self-consistency RMSD between {input_path} and {reference_path}......")
    
    esm_paths = sorted(
        [esm_path for esm_path in input_path.glob("*.pdb") if esm_path.is_file()],
        key=natural_sort_key
    )
    af2_paths = sorted(
        [af2_path for af2_path in reference_path.glob("*.pdb") if af2_path.is_file()],
        key=natural_sort_key
    )   
    
    for esm_path in esm_paths:
        sample_match = re.search(r'(\d+)$', esm_path.stem)
        if sample_match:
            sample_number = int(sample_match.group(1))
            esm_dict[sample_number] = esm_path
    
    for af2_path in af2_paths:
        sample_match = re.search(r'(\d+)$', af2_path.stem)
        if sample_match:
            sample_number = int(sample_match.group(1))
            af2_dict[sample_number] = af2_path
            
    for sample_number, esm_pdb in esm_dict.items():
        if sample_number in af2_dict:
            af2_pdb = af2_dict[sample_number]
            ca_rmsd = CA_rmsd(esm_pdb, af2_pdb)
            logger.info(f"Sample: {sample_number}, Cα-RMSD = {ca_rmsd}.")
            results.append((sample_number, ca_rmsd))
    
    with output_file.open('w') as out:
        for sample_number, ca_rmsd in results:
            out.write(f"{sample_number}\t{ca_rmsd:.3f}\n")
        logging.info(f'Voilà!')
            
        if args.threshold:
            filtered_numbers = [
                sample_number
                for sample_number, ca_rmsd in results
                if ca_rmsd <= scRMSD_threshold
            ]
            logger.info(f"Number of PDBs kept after filtering: {len(filtered_numbers)}.")
            filter_id_file = root_path / f"filterID_{basename}.txt"
            with filter_id_file.open("w") as f:
                f.write("\n".join(map(str, sorted(filtered_numbers))))
                
def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)
    
if __name__ == "__main__":
    main()
            
            
    