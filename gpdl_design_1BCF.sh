#!/bin/bash

#SBATCH --job-name=1BCF
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1           
#SBATCH --output=%j.out
#SBATCH --error=%j.err

start=$(date +%s.%N)

protein_name="1BCF"
dir_name="1BCF" # for different length scale
inpaint_seq="10,A92-99,20,A123-130,20,A47-54,20,A18-25,10"
mask_len="10,20,20,20,10"
motif_id="A92-99,A123-130,A47-54,A18-25"
max_mut=15
step=1500

temp_dir="/dssg/home/acct-clschf/clschf/lkx/test_early_stop/${protein_name}"
temp_dir_inpaint="${temp_dir}/inp"
temp_dir_hal="${temp_dir}/hal"
final_des_dir="/dssg/home/acct-clschf/clschf/lkx/test_early_stop/final/${protein_name}"
reference="/dssg/home/acct-clschf/clschf/lkx/protein_ref/${protein_name}.pdb"
inp_num=2
if_num=1
hal_num=1

if [ ! -d $temp_dir_inpaint ]; then
    mkdir -p $temp_dir_inpaint
fi
if [ ! -d $temp_dir_hal ]; then
    mkdir -p $temp_dir_hal
fi
if [ ! -d $final_des_dir ]; then
    mkdir -p $final_des_dir
fi

source activate ESMFold

python3 /dssg/home/acct-clschf/clschf/lkx/GPDL/gpdl_inpainting/esm_inference_v2.py  \
    --inpaint_seq "${inpaint_seq}" \
    --input "${reference}" \
    --output_prefix "${temp_dir_inpaint}/${protein_name}" \
    --fpath "${temp_dir_inpaint}/${protein_name}.txt" \
    --num_design $inp_num \
    --weight_path "/dssg/home/acct-clschf/clschf/lkx/GPDL/gpdl_inpainting/checkpoints/inpaint_weight_11.pt" \

for i in $(seq 0 $((inp_num-1)))
do
    echo $i
    # fixed-bb design for inpainted sequence
    source activate esm-if1
    export PYTHONNOUSERSITE=1
    python /dssg/home/acct-clschf/clschf/lkx/GPDL/sample_sequences.py ${temp_dir_inpaint}/${protein_name}_${i}.pdb \
        --chain A --temperature 1 --num-samples $if_num \
        --outpath "${temp_dir_inpaint}/${protein_name}_${i}_esmif.fasta" \
        --inpaint_file "${temp_dir_inpaint}/${protein_name}.txt"

    # get the protein structure from hallucination
    conda activate ESMFold
    python3 /dssg/home/acct-clschf/clschf/lkx/GPDL/gpdl_hallucination/hallucination_v1.py \
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

end=$(date +%s.%N)

runtime=$(echo "$end - $start" | bc)
echo "total time consumption: $runtime seconds" >> "${temp_dir}/time_consumption.txt"