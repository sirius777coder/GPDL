#!/bin/bash
#SBATCH --job-name=7MRX_60_240312
#SBATCH --partition=dgx2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

module load miniconda3
source activate esmfold

protein_name="7MRX_60"
inpaint_seq="19,B25-46,19"
mask_len="19,19"
motif_id="B25-46"
max_mut=15
step=800

temp_dir="/lustre/home/acct-stu/stu006/GPDL/temp/${protein_name}"
temp_dir_inpaint="${temp_dir}/inpaint"
temp_dir_hal="${temp_dir}/hal"
final_des_dir="/lustre/home/acct-stu/stu006/GPDL/240312/${protein_name}"
reference="/lustre/home/acct-stu/stu006/protein_ref/${protein_name}.pdb"
inp_num=10
if_num=1
hal_num=1

if [ ! -d $temp_dir_inpaint ]; then
    mkdir -p $temp_dir_inpaint
fi
if [ ! -d $temp_dir_hal ]; then
    mkdir -p $temp_dir_hal
fi

python3 /lustre/home/acct-stu/stu006/GPDL/GPDL/gpdl_inpainting/esm_inference_v2.py  \
    --inpaint_seq "${inpaint_seq}" \
    --input "${reference}" \
    --output_prefix "${temp_dir_inpaint}/${protein_name}" \
    --fpath "${temp_dir_inpaint}/${protein_name}.txt" \
    --num_design $inp_num \
    --weight_path "GPDL/gpdl_inpainting/checkpoints/inpaint_weight_11.pt" \
# inpainting sequence
# output=$(python3 gpdl_inpainting/utils_seq.py ${temp_dirpaint}/${protein_name}_0.pdb)
# echo $output

for i in $(seq 0 $((inp_num-1)))
do
    echo $i
    # fixed-bb design for inpainted sequence
    conda activate /lustre/home/acct-stu/stu006/miniconda3/envs/esm_if
    python /lustre/home/acct-stu/stu006/GPDL/GPDL/sample_sequences.py ${temp_dir_inpaint}/${protein_name}_${i}.pdb \
        --chain A --temperature 1 --num-samples $if_num \
        --outpath "${temp_dir_inpaint}/${protein_name}_${i}_esmif.fasta" \
        --inpaint_file "${temp_dir_inpaint}/${protein_name}.txt"

    # get the protein structure from hallucination
    source activate esmfold
    python3 /lustre/home/acct-stu/stu006/GPDL/GPDL/gpdl_hallucination/hallucination_v1.py \
        --pre_sequence "${temp_dir_inpaint}/${protein_name}_${i}_esmif.fasta" \
        --reference $reference \
        --output_dir $temp_dir_hal \
        --final_des_dir $final_des_dir \
        --bb_suffix $i \
        --step $step \
        --loss 10 \
        --t1 1 \
        --t2 500 \
        --max_mut $max_mut \
        --number $hal_num \
        --mask_len $mask_len \
        --motif_id $motif_id \
        --atom N,CA,C,O

done