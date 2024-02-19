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

protein_name="1BCF"
temp_dir="./experiment/${protein_name}"
temp_dir_inapint="${temp_dir}/inpaint"
temp_dir_hal="${temp_dir}/hal"
reference="./gpdl_inpainting/benchmark_set/${protein_name}.pdb"
if [ ! -d $temp_dir_inapint ]; then
    mkdir -p $temp_dir_inapint
fi
python3 gpdl_inpainting/esm_inference_v2.py  \
    --inpaint_seq "10,A92-99,20,A123-130,20,A47-54,20,A18-25,10" \
    --input "${reference}" \
    --output_prefix "${temp_dir_inapint}/${protein_name}" \
    --fpath "${temp_dir_inapint}/${protein_name}.txt" \
# inpainting sequence
output=$(python3 gpdl_inpainting/utils_seq.py ${temp_dir_inapint}/${protein_name}_0.pdb)
echo $output

# get the protein structure from hallucination
python3 gpdl_hallucination/hallucination_v1.py \
    --pre_sequence $output \
    --reference $reference \
    --output_dir $temp_dir_hal \
    --step 1500 \
    --loss 10 \
    --t1 1 \
    --t2 500 \
    --n_mut_max 20 \
    --number 100 \
    --mask_len 10,20,20,20,10 \
    --motif_id A92-99,A123-130,A47-54,A18-25 \
    --atom N,CA,C,O

