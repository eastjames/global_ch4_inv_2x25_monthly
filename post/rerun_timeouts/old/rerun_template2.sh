#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o "rerun_imi_output.log"
#SBATCH --mem 16000
#SBATCH -c 4
#SBATCH -t 0-01:00
#SBATCH -p my_partition
#SBATCH -J rerun_my_run_dir
#SBATCH --mail-type=ALL

JacobianCPUs=8
RequestedTime="0-02:00"
SchedulerPartition=my_partition
JacobianMemory=10000
RunDir=my_run_dir
OutputPath=my_output_path
BaseProjDir=base_proj_dir

source ../../integrated_methane_inversion/envs/Harvard-Cannon/gcclassic.rocky+gnu12.minimal.env

# testing
cd ${OutputPath}/${RunDir}/jacobian_runs

# check for any time out failures
nfails=$(grep -i time\ limit slurm* | wc -l)
if [ $nfails -gt 0 ]; then
#echo Re-running ${nfails} Jacobian simulations

# Get jobs that timed out based on slurm stderr files
JOBS=''
while read -r line ; do
    jobn=$(echo $line | cut -d "_" -f 2 | cut -d "." -f 1)
    #JOBS=${JOBS},${jobn}

    echo "Re-running ${nfails} jacobian simulations" >> ${BaseProjDir}/cases/${RunDir}/rerun_imi_output.log
    
    # remove error status file if present
    rm -f .error_status_file.txt
    
    echo "sbatch --array=${jobn}%1 --mem $JacobianMemory -c $JacobianCPUs -t $RequestedTime -p $SchedulerPartition -W run_jacobian_simulations.sh" >> ${BaseProjDir}/cases/${RunDir}/rerun_imi_output.log
    
    
    sbatch --array=${jobn}%1 --mem $JacobianMemory \
    -c $JacobianCPUs \
    -t $RequestedTime \
    -p $SchedulerPartition \
    run_jacobian_simulations.sh
done < <(grep -l -i time\ limit slurm*)

else
echo "No jobs to rerun" >> ${BaseProjDir}/cases/${RunDir}/rerun_imi_output.log
fi

#popd
