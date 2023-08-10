import os
import re
import argparse
from pyrosetta import *
from rosetta.protocols.relax import *
from pyrosetta.rosetta.protocols.relax import FastRelax

def extract_sample_number(file_name):
    sample_match = re.search(r'(\d+)\.pdb', file_name)
    if sample_match:
        sample_number = int(sample_match.group(1))
        return sample_number
    else:
        raise ValueError(f"Could not find the sample number in file name: {file_name}")

def relax_pdb(input_pdb, output_folder, output_prefix):
    # Initialize PyRosetta
    init()

    # Load the protein structure
    pose = pose_from_pdb(input_pdb)

    # Create a FastRelax object and use it to refine the structure
    scorefxn = get_fa_scorefxn()
    fast_relax = FastRelax()
    fast_relax.set_scorefxn(scorefxn)
    fast_relax.apply(pose)

    # Extract the sample number from the input_pdb file name
    sample_number = extract_sample_number(os.path.basename(input_pdb))

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the relaxed PDB to the output folder with the specified prefix
    output_pdb = os.path.join(output_folder, f"{output_prefix}_{sample_number}.pdb")
    pose.dump_pdb(output_pdb)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Perform relaxation using PyRosetta")
    parser.add_argument("-d", "--design", type=str, required=True, help="Path to the folder containing input PDBs")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to the output folder (default: the first part of the file name + 'relaxed')")
    parser.add_argument("-p", "--prefix", type=str, default="relaxAF_apo_relaxed", help="Output file name prefix (default: 'relaxAF_apo_relaxed')")
    args = parser.parse_args()

    # Get the input and output paths
    input_folder = os.path.normpath(args.design)
    output_folder = args.output if args.output is not None else os.path.join(os.path.dirname(input_folder), args.prefix)

    # Relax each PDB in the input folder
    for pdb_file in os.listdir(input_folder):
        if pdb_file.endswith(".pdb"):
            input_pdb = os.path.join(input_folder, pdb_file)
            relax_pdb(input_pdb, output_folder, args.prefix)
