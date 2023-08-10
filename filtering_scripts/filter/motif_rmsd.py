#Motif-RMSD and pLDDT filtering script.
#The original script is from @weiting and is rewritten by @immortals on July 21, 2023.
import os
import numpy as np
import re
import warnings
import argparse
import logging
import time
import shutil
import typing as T
from typing import List
from pathlib import Path

from pymol import cmd
from Bio.PDB import *
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBParser import PDBConstructionWarning as BiopythonWarning

warnings.filterwarnings("ignore", category=BiopythonWarning)
'''
This script is for the following purpose:
1. Calculate motif-RMSD based on a given motif txt file between a set of designed PDBs and a reference PDB structure.
2. Extract pLDDT, which is written into a txt file along with motif-RMSD for each PDBs.
3. Filter PDBs using the threshold of motif-RMSD and pLDDT, e.g. motif-RMSD < 1.5 and pLDDT >80.
4. Save the PDBs after filtering to a new folder, and return a list of IDs of kept PDBs, saved into a new file.

Several arguments could be input as a command-line usage:
[Required]
"-d", "--design_path": Path to the folder containing designed PDBs.
"-r", "--reference_pdb": Path to the reference PDB.
"--motif_d": Path to the txt containing motif information of designed PDBs.
"--motif_r": Path to the txt containing motif information of reference PDB.
[Options]
"--rmsd": The filtering threshold of motif-RMSD, default = 1.0
"--plddt", The filtering threshold of pLDDT, default = 80
"--root-path": Root path of operation. Default = My own motif-RMSD filtering folder.
'''


parser = argparse.ArgumentParser(description="Caculate motif RMSD based on a given set of PDBs and a reference PDB")

parser.add_argument("-d", "--design_path", type=str, required=True, help="Path to the folder containing the designing PDBs.")
parser.add_argument("-r", "--reference_pdb", type=str, required=True, help="Path to the reference PDB file.")
parser.add_argument("--motif_d", type=str, required=True, help="Path to the motif positions of designed PDBs.")
parser.add_argument("--motif_r", type=str, required=True, help="Path to the motif positions of the reference PDB.")
parser.add_argument("--rmsd", type=float, default=1.0, help="Filtering threshold of Motif-RMSD")
parser.add_argument("--plddt", type=float, default=80.0, help="Filtering threshold of pLDDT")
parser.add_argument("--root-path", 
                    type=Path,
                    default=Path("/dssg/home/acct-clschf/clschf/zzq/ADS/filter/motif_RMSD"),
                    help= "Root path for file operations")

args = parser.parse_args() 

def motifRMSD(ref_pdb, des_pdb, ref_motif_txt, des_motif_txt=None, segment="\t", option="CA"):
    """
    txt should be like this: chain,sequence identifier
    E.g. A,100
    """
    ref_motif = {}
    des_motif = {}
    with open(ref_motif_txt, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            chain, id = line.split(segment)
            if chain in ref_motif.keys():
                ref_motif[chain].append(int(id))
            else:
                ref_motif[chain] = [int(id)]
    if des_motif_txt is not None:
        with open(des_motif_txt, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                chain, id = line.split(segment)
                if chain in des_motif.keys():
                    des_motif[chain].append(int(id))
                else:
                    des_motif[chain] = [int(id)]
    else:
        des_motif_txt = ref_motif_txt
    paser = PDBParser()
    structure_ref = paser.get_structure("ref", ref_pdb)
    structure_des = paser.get_structure("des", des_pdb)
    if option == "CA":
        ref_coord = []
        des_coord = []
        for chain in ref_motif.keys():
            model_ref = structure_ref[0]
            for id in ref_motif[chain]:
                res_ref = model_ref[chain][id]
                ca_ref = res_ref["CA"]
                ref_coord.append(ca_ref.get_coord())

        for chain in des_motif.keys():
            model_des = structure_des[0]
            for id in des_motif[chain]:
                res_des = model_des[chain][id]
                ca_des = res_des["CA"]
                des_coord.append(ca_des.get_coord())
        ref_coord = np.array(ref_coord)
        des_coord = np.array(des_coord)
        sup = SVDSuperimposer()
        sup.set(des_coord, ref_coord)
        sup.run()
        motif_rmsd = sup.get_rms()
        return motif_rmsd


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

def bfactor(OF_pdb):
    cmd.load(OF_pdb)
    name = OF_pdb.split('/')[-1].split('.')[0]
    myspace = {'bfactors': []}
    cmd.iterate(name, 'bfactors.append(b)', space=myspace)
    cmd.delete('all')
    return np.mean(myspace['bfactors'])

def natural_sort_key(s: Path) -> List:
    # Helper function for natural sorting of file names
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

if __name__ == "__main__":   
    design_path = os.path.normpath(args.design_path)
    root_path = args.root_path.resolve()
    des_motif_txt = args.motif_d
    native_motif_txt = args.motif_r
    native = args.reference_pdb
    rmsd_threshold = args.rmsd
    pLDDT_threshold = args.plddt
    print(des_motif_txt)
    print(native_motif_txt)
    
    out_dir = os.path.join(root_path, 'results')
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.basename(design_path)
    out_file = os.path.join(out_dir, f"motifRMSD_{basename}.txt")
    
    logging.basicConfig(level=logging.INFO, format = "%(asctime)s | %(levelname)s | %(message)s", datefmt="%d/%m/%y %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    logger.info(f"Calculating motif-RMSD and pLDDT from {design_path}")
    
    filtered_dir = os.path.join(root_path, f"motifRMSD_{basename}")
    os.makedirs(filtered_dir, exist_ok= True)
    
    sample_numbers = []

    with open(out_file, 'w') as out:
        out.write('design\tESM_RMSD\tESM_pLDDT\n')
        logger.info(f"Loaded {len(os.listdir(design_path))} designed PDBs from {design_path}")
        logger.info("Calculating......")
        for pdb_file in sorted(os.listdir(design_path), key=natural_sort_key):
            if pdb_file.endswith(".pdb"):
                pdb = pdb_file.strip()
                des_pdb = os.path.join(design_path, pdb)
                rmsd = motifRMSD(des_pdb, native, des_motif_txt, native_motif_txt)
                pLDDT = bfactor(des_pdb)
                out.write('%s\t%.3f\t%.3f\n' % (pdb, rmsd, pLDDT))
                logger.info(f"Sample: {pdb}, motif-RMSD = {rmsd}, pLDDT = {pLDDT}")
                
                if rmsd <= rmsd_threshold and pLDDT >= pLDDT_threshold:
                    #out.write('%s\t%.3f\t%.3f\n' % (pdb, rmsd, pLDDT))
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