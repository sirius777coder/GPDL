
source activate
conda activate gpdl

# parameters and paths 
protein_name="2FYD"
dir_name="2FYD" # for different length scale
inpaint_seq="50,D297-311,50"
mask_len="50,50"
motif_id="D297-311"
max_mut=15
step=1500


temp_dir="temp/${protein_name}"
temp_dir_inpaint="${temp_dir}/inpaint"
temp_dir_hal="${temp_dir}/hal"
final_des_dir="./final_design/${dir_name}"
reference="./${protein_name}.pdb"
inp_num=100
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
python3 ./gpdl_inpainting/esm_inference_v2.py  \
    --inpaint_seq "${inpaint_seq}" \
    --input "${reference}" \
    --output_prefix "${temp_dir_inpaint}/${protein_name}" \
    --fpath "${temp_dir_inpaint}/${protein_name}.txt" \
    --num_design $inp_num \
    --weight_path "./gpdl_inpainting/checkpoints/inpaint_weight_11.pt" \

for i in $(seq 0 $((inp_num-1)))
do
    echo "optimization numbers : $i"
    # fixed-bb design for inpainted sequence
    conda activate esm_if
    python ./sample_sequences.py ${temp_dir_inpaint}/${protein_name}_${i}.pdb \
        --chain A --temperature 1 --num-samples $if_num \
        --outpath "${temp_dir_inpaint}/${protein_name}_${i}_esmif.fasta" \
        --inpaint_file "${temp_dir_inpaint}/${protein_name}.txt"

    # get the protein structure from hallucination
    conda activate gpdl
    python3 ./gpdl_hallucination/hallucination_v1.py \
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


