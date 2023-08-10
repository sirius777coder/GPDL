#!/bin/bash
#SBATCH --job-name=Alphafold_2ndRound
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

echo 'Start'

for i in {1..20}; do
    sbatch <<EOF
#!/bin/bash
#SBATCH -J AF2_${i}
#SBATCH --partition=a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=AF2_${i}.out
#SBATCH --error=AF2_${i}.err

directory="../sequences_extract/fasta_r2_seperate/fasta_r2_${i}"
files=\$(ls \$directory)
for file in \$files
do
    singularity run --nv /dssg/share/imgs/parafold/2.0.sif \
    /app/ParallelFold/run_alphafold.sh \
    -d /dssg/share/data/alphafold \
    -o ./output_r2/r2_${i} \
    -p monomer_ptm \
    -i \$directory/\$file \
    -t 2021-07-27 \
    -m model_2
done
EOF
done

echo 'Voila!'



