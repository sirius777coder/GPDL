#! /bin/bash

#PBS -S /bin/bash
#PBS -N esm_test
#PBS -o ${PBS_JOBNAME}.out
#PBS -e ${PBS_JOBNAME}.err
#PBS -q gpuq
#PBS -l nodes=gpu01:ppn=8
#PBS -W x=GRES:gpu@1
#PBS -r y
cd ${PBS_O_WORKDIR}

module load anaconda
source activate esmfold

echo "STARTING testing"

# python3 ./esm_inference.py  \
#     --inpaint_seq "A1-7,10,A28-79" \
#     --input ./benchmark_set/2KL8.pdb \
#     --output_prefix "2kl8_scaffold_10_1"

# python3 ./esm_inference.py  \
#     --inpaint_seq "A1-7,5,A28-79" \
#     --input ./benchmark_set/2KL8.pdb \
#     --output_prefix "2kl8_scaffold_5_1"


for ((i=1;i<4;i++))
do
    for ((j=0;j<500;j++))
    do
    echo "step $i, design $j"
    python3 ./esm_inference.py  \
        --inpaint_seq "0-30,A119-140,15-40,A63-82,0-30" \
        --sample_steps $i \
        --input ./benchmark_set/5IUS.pdb \
        --output_prefix "./design/5ius/T$i/5IUS_$j"
    done
done



echo "ENDING testing"
