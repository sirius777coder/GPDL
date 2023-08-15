#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --partition=dgx2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=out_file_name.out
#SBATCH --error=err_file_name.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

cd ${PBS_O_WORKDIR}
module load miniconda3
source activate esmfold


echo "STARTING testing"

python3 /hal_esm-III.py \
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

echo 'ENDING testing'
