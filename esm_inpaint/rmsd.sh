#! /bin/bash

#PBS -S /bin/bash
#PBS -N rmsd_review
#PBS -o ${PBS_JOBNAME}.out
#PBS -e ${PBS_JOBNAME}.err
#PBS -q cpuq
#PBS -l nodes=1:ppn=8
#PBS -r y
cd ${PBS_O_WORKDIR}

module load anaconda
source activate esmfold

echo "STARTING testing"

python3 ./rmsd.py

echo "END testing"