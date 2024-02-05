#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o "rerun_imi_output.log"
#SBATCH --mem 16000
#SBATCH -c 4
#SBATCH -t 0-72:00
#SBATCH -p my_partition
#SBATCH -J rerun_my_run_dir
#SBATCH --mail-type=ALL

JacobianCPUs=8
RequestedTime="0-02:30"
SchedulerPartition=my_partition
JacobianMemory=10000
RunDir=my_run_dir
OutputPath=my_output_path
BaseProjDir=base_proj_dir

source ../../integrated_methane_inversion/envs/Harvard-Cannon/gcclassic.rocky+gnu12.minimal.env

cd ${OutputPath}/${RunDir}/jacobian_runs

#sbatch --array=0,423%2 --mem $JacobianMemory \
sbatch --array=0-3753%100 --mem $JacobianMemory \
-c $JacobianCPUs \
-t $RequestedTime \
-p $SchedulerPartition \
-W rerun_jacobian_simulations.sh


