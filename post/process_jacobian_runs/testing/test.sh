#!/bin/bash

#SBATCH -c 64
#SBATCH -t 0-00:30
#SBATCH --mem 256gb
#SBATCH -p seas_compute,serial_requeue,sapphire
#SBATCH -J testing
#SBATCH -o stest-%j.out

module load python
mamba activate imi_env

python -u ProcessGlobalJacobianRuns.py 201807 imi_20180701_000001 /n/holyscratch01/jacob_lab/jeast/proj/globalinv/
