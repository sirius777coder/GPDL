#! /bin/bash

#PBS -S /bin/bash
#PBS -N esm_batch
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


python3 ./esm_inference_v2.py  \
    --inpaint_seq "A1-7,20,A28-79" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/2KL8.pdb \
    --output_prefix "./design/2kl8/2KL8" \
    --fpath "./design/2kl8/2KL8.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "5-20,A16-35,10-25,A52-71,5-20" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/1PRW.pdb \
    --output_prefix "./design/1prw/1PRW" \
    --fpath "./design/1prw/1PRW.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "8-15,A92-99,16-30,A123-130,16-30,A47-54,16-30,A18-25,8-15" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/1BCF.pdb \
    --output_prefix "./design/1bcf/1BCF" \
    --fpath "./design/1bcf/1BCF.txt"


python3 ./esm_inference_v2.py  \
    --inpaint_seq "10-40,P254-277,10-40" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/3IXT.pdb \
    --output_prefix "./design/3ixt/3IXT" \
    --fpath "./design/3ixt/3IXT.txt"


python3 ./esm_inference_v2.py  \
    --inpaint_seq "10-20,A38,15-30,A14,15-30,A99,10-20" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/1QJG.pdb \
    --output_prefix "./design/1qjg/1QJG" \
    --fpath "./design/1qjg/1QJG.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "10-40,B19-27,10-40" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/1YCR.pdb \
    --output_prefix "./design/1ycr/1YCR" \
    --fpath "./design/1ycr/1YCR.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-38,B25-46,0-38" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/7MRX.pdb \
    --output_prefix "./design/7mrx_60/7MRX_60" \
    --fpath "./design/7mrx_60/7MRX_60.txt"


python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-63,B25-46,0-63" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/7MRX.pdb \
    --output_prefix "./design/7mrx_85/7MRX_85" \
    --fpath "./design/7mrx_85/7MRX_85.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-122,B25-46,0-122" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/7MRX.pdb \
    --output_prefix "./design/7mrx_128/7MRX_128" \
    --fpath "./design/7mrx_128/7MRX_128.txt"



python3 ./esm_inference_v2.py  \
    --inpaint_seq "10-25,F196-212,15-30,F63-69,10-25" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/4JHW.pdb \
    --output_prefix "./design/4jhw/4JHW" \
    --fpath "./design/4jhw/4JHW.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "10-40,A422-436,10-40" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/4ZYP.pdb \
    --output_prefix "./design/4zyp/4ZYP" \
    --fpath "./design/4zyp/4ZYP.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "10-40,A170-189,10-40" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/5WN9.pdb \
    --output_prefix "./design/5wn9/5WN9" \
    --fpath "./design/5wn9/5WN9.txt"

# python3 ./esm_inference_v2.py  \
#     --inpaint_seq "10-40,A170-189,10-40" \
#     --sample_steps 1 \
#     --num_design 1000 \
#     --input ./benchmark_set/6VW1.pdb \
#     --output_prefix "./design/6vw1/6VW1" \
#     --fpath "./design/6vw1/6VW1.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-35,A45-65,0-35" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/5TRV.pdb \
    --output_prefix "./design/5trv_short/5TRV_short" \
    --fpath "./design/5trv_short/5TRV_short.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-65,A45-65,0-65" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/5TRV.pdb \
    --output_prefix "./design/5trv_med/5TRV_med" \
    --fpath "./design/5trv_med/5TRV_med.txt"


python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-95,A45-65,0-95" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/5TRV.pdb \
    --output_prefix "./design/5trv_long/5TRV_long" \
    --fpath "./design/5trv_long/5TRV_long.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-35,A23-35,0-35" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/6E6R.pdb \
    --output_prefix "./design/6e6r_short/6E6R_short" \
    --fpath "./design/6e6r_short/6E6R_short.txt"


python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-65,A23-35,0-65" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/6E6R.pdb \
    --output_prefix "./design/6e6r_med/6E6R_med" \
    --fpath "./design/6e6r_med/6E6R_med.txt"


python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-95,A23-35,0-95" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/6E6R.pdb \
    --output_prefix "./design/6e6r_long/6E6R_long" \
    --fpath "./design/6e6r_long/6E6R_long.txt"

# 530 is the first residue A28-42 to A528-572
python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-35,A28-42,0-35" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/6EXZ.pdb \
    --output_prefix "./design/6exz_short/6EXZ_short" \
    --fpath "./design/6exz_short/6EXZ_short.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-65,A28-42,0-65" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/6EXZ.pdb \
    --output_prefix "./design/6exz_med/6EXZ_med" \
    --fpath "./design/6exz_med/6EXZ_med.txt"

python3 ./esm_inference_v2.py  \
    --inpaint_seq "0-95,A28-42,0-95" \
    --sample_steps 1 \
    --num_design 1000 \
    --input ./benchmark_set/6EXZ.pdb \
    --output_prefix "./design/6exz_long/6EXZ_long" \
    --fpath "./design/6exz_long/6EXZ_long.txt"

echo "ENDING testing"
