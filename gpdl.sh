#!/bin/bash
#SBATCH --job-name=gpdl_hal
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=hal.out
#SBATCH --error=hal.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

module load miniconda3
source activate esmfold

# pipeline for GPDL
temp_dir = "./gpdl_inpainting/design/1bcf/1BCF"
python3 gpdl_inpainting/esm_inference_v2.py  \
    --inpaint_seq "8-15,A92-99,16-30,A123-130,16-30,A47-54,16-30,A18-25,8-15" \
    --input ./gpdl_inpainting/benchmark_set/1BCF.pdb \
    --output_prefix $temp_dir \
    --fpath "./gpdl_inpainting/design/1bcf/1BCF.txt"

# inpainting sequence
output=$(python3 gpdl_inpainting/utils_seq.py ${temp_dir}_1.pdb)

# get the protein structure from hallucination
python3 gpdl_hallucination/hal_esm-v1.py \
    --pre_sequence  $output \
    --reference /path_of_reference.pdb \
    --output_dir /output_folder \
    --step 1500 \
    --loss 10 \
    --t1 1 \
    --t2 500 \
    --n_mut_max 20 \
    --number 100 \
    --mask_len 10,20,20,20,10 \
    --motif_id A92-99,A123-130,A47-54,A18-25 \
    --atom N,CA,C,O

