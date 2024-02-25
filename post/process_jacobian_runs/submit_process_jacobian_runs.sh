#!/bin/bash

module load python
mamba activate imi_env

TID=${SLURM_ARRAY_TASK_ID}
state_vector=$(printf "%06d" $TID)

python -u ProcessGlobalJacobianRuns.py ${1} ${2}_${state_vector} ${3} ${4}


#python -u ProcessGlobalJacobianRuns.py 201806 imi_20180601_000000

