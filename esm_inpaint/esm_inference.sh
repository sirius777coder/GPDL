# python3 ./esm_inference.py  \
#     --inpaint_seq "A1-7,10,A28-79" \
#     --input ./benchmark_set/2KL8.pdb \
#     --output_prefix "2kl8_scaffold_10_1"

# python3 ./esm_inference.py  \
#     --inpaint_seq "A1-7,5,A28-79" \
#     --input ./benchmark_set/2KL8.pdb \
#     --output_prefix "2kl8_scaffold_5_1"

python3 ./esm_inference.py  \
    --inpaint_seq "10,A119-140,10,A63-82, 15" \
    --input ./benchmark_set/5IUS.pdb \
    --output_prefix "5IUS_1"