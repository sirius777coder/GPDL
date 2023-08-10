#AlphaFold predictions collecting script.
#Written by @Immortals on Augest 3rd, 2023.

'''
For those filtered sequences from ESMFold, we run AlphaFold2 and carry out the second filtering pipeline.
For my own preference, the AF2 predictions are sorted in the following style:
1. output_r{i}: The {i}th design process, i.e. for one specific backbone.
   2. r{i}_{j}: Since running AF2 is not fast enough, I break all the fastas in {j} pieces. 
                {j} can also be set and personalized in another script, see a detailed description in
                `/sequences_extract/folder_seperate.py`.
        3. {design_prefix}_{sample_number}: This is the folder containing predictions from each seperate fasta. 
                                            The {sample_number} is samples filtered from ESMFold pipeline.
                                            
Under each fasta folder contains many files, but I'm interested in "relaxed_model_2_pred_0.pdb" 
(I just run one model in this pipeline). So this script is for:
1. Get all the "relaxed_model_2_pred_0.pdb" in AF2 prediction results.
2. Copy them to a new folder.
3. Rename them by their design prefix and sample number.

This script designs for more personal usage so it's a little bit hard to generalize.
One can read this script manually and adjust them for adapted usage.
''' 

import os
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Extract and rename pdb files.")
    parser.add_argument("-i", "--input", default=".", help="Input folder containing the r2_* subfolders. Default is the current directory.")
    parser.add_argument("-o", "--output", default="~/zzq/ADS/af2_predictions/output_r2", help="Output folder to store the renamed pdb files.")
    parser.add_argument("-n", "--name", help="New prefix for renaming pdb files.")

    return parser.parse_args()

def main():
    args = parse_args()
    input_folder = os.path.expanduser(args.input)
    output_folder = os.path.expanduser(args.output)
    new_prefix = args.name

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for r2_folder in os.listdir(input_folder):
        r2_path = os.path.join(input_folder, r2_folder)
        if not os.path.isdir(r2_path):
            continue

        for prefix_sample_folder in os.listdir(r2_path):
            prefix_sample_path = os.path.join(r2_path, prefix_sample_folder)
            if not os.path.isdir(prefix_sample_path):
                continue

            pdb_file = os.path.join(prefix_sample_path, "relaxed_model_2_pred_0.pdb")
            if not os.path.exists(pdb_file):
                continue

            # Determine the new prefix
            if new_prefix:
                new_name = f"af2_{new_prefix}_{prefix_sample_folder}"
            else:
                new_name = f"af2_{prefix_sample_folder}"

            # Copy and rename the pdb file
            new_pdb_file = os.path.join(output_folder, f"{new_name}.pdb")
            shutil.copy(pdb_file, new_pdb_file)

if __name__ == "__main__":
    main()
