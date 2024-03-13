#!/bin/bash
#SBATCH --job-name=gpdl_inp
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=inp.out
#SBATCH --error=inp.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

module load anaconda
source activate gpdl

python3 ./gpdl_inpainting/esm_inference_v2.py  \
    --inpaint_seq "8-15,A92-99,16-30,A123-130,16-30,A47-54,16-30,A18-25,8-15" \
    --sample_steps 1 \
    --num_design 100 \
    --input ./gpdl_inpainting/benchmark_set/1BCF.pdb \
    --output_prefix "./gpdl_inpainting/design/1bcf/1BCF" \
    --fpath "./gpdl_inpainting/design/1bcf/1BCF.txt"
