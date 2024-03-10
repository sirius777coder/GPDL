#!/bin/bash
#SBATCH --job-name=gpdl
#SBATCH -p carbon
#SBATCH --gres=gpu:A100:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
conda activate esmfold

protein_name="1BCF"
inpaint_seq="10,A92-99,20,A123-130,20,A47-54,20,A18-25,10"
mask_len="10,20,20,20,10"
motif_id="A92-99,A123-130,A47-54,A18-25"
max_mut=15

temp_dir="./temp/${protein_name}"
temp_dir_inapint="${temp_dir}/inpaint"
temp_dir_hal="${temp_dir}/hal"
reference="./gpdl_inpainting/benchmark_set/${protein_name}.pdb"
number=100
if [ ! -d $temp_dir_inapint ]; then
    mkdir -p $temp_dir_inapint
fi
if [ ! -d $temp_dir_hal ]; then
    mkdir -p $temp_dir_hal
fi
python3 gpdl_inpainting/esm_inference_v2.py  \
    --inpaint_seq "${inpaint_seq}" \
    --input "${reference}" \
    --output_prefix "${temp_dir_inapint}/${protein_name}" \
    --fpath "${temp_dir_inapint}/${protein_name}.txt" \
# inpainting sequence
# output=$(python3 gpdl_inpainting/utils_seq.py ${temp_dir_inapint}/${protein_name}_0.pdb)
# echo $output

# fixed-bb design for inpainted sequence
conda activate esm_if
python ./sample_sequences.py ${temp_dir_inapint}/${protein_name}_0.pdb \
    --chain A --temperature 1 --num-samples $number \
    --outpath "${temp_dir_inapint}/${protein_name}_esmif.fasta" \

# get the protein structure from hallucination
conda activate esmfold
python3 gpdl_hallucination/hallucination_v1.py \
    --pre_sequence "${temp_dir_inapint}/${protein_name}.fasta" \
    --reference $reference \
    --output_dir $temp_dir_hal \
    --step 1500 \
    --loss 10 \
    --t1 1 \
    --t2 500 \
    --max_mut $max_mut \
    --number $number \
    --mask_len $mask_len \
    --motif_id $motif_id \
    --atom N,CA,C,O

