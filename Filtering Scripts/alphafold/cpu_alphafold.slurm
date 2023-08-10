#!/bin/bash
#SBATCH --job-name=cpu_Alphafold_2ndRound
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
##SBATCH --cpus-per-task=64
##SBATCH --gres=gpu:1
#SBATCH --output=cpu_%j.out
#SBATCH --error=cpu_%j.err

echo 'Start'

for i in {1..20}; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -J cpu_AF2_${i}
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
##SBATCH --cpus-per-task=16
##SBATCH --gres=gpu:1
#SBATCH --output=cpu_AF2_${i}.out
#SBATCH --error=cpu_AF2_${i}.err

directory="../sequences_extract/fasta_r2_seperate/fasta_r2_${i}"
files=\$(ls \$directory)
for file in \$files
do
    singularity run --nv /dssg/share/imgs/parafold/2.0.sif \
    /app/ParallelFold/run_alphafold.sh \
    -d /dssg/share/data/alphafold \
    -o ./cpu_output_r2/r2_${i} \
    -p monomer_ptm \
    -i \$directory/\$file \
    -t 2021-07-27 \
    -m model_2
done
EOF
done

echo 'Voila!'
