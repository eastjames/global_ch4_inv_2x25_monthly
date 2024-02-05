#!/bin/bash

#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-05:00
####SBATCH -p test
#SBATCH -J testing
#SBATCH -o slurm-%j.out

echo 'foo'

