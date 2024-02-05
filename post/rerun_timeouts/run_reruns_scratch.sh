#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o "rerun_prod_run_scratch.log"
#SBATCH -c 1
#SBATCH -t 0-72:00
#SBATCH -p seas_compute
#SBATCH -J main_rerun_scratch


# specify imi name, output path, and partition
RunNamePrefix="imi"
OutputPath="/n/holyscratch01/jacob_lab/jeast/proj/globalinv/prod/output"
#OutputPath="/n/holylfs06/SCRATCH/jacob_lab/jeast/proj/globalinv/prod/output"
SchedulerPartition="seas_compute"

# make output dir
# mkdir -p $OutputPath



new_sbatch() {
    sbr="$(/usr/bin/sbatch "$@")"
    
    if [[ "$sbr" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        exit 0
    else
        echo "sbatch failed"
        exit 1
    fi
}



source ../../integrated_methane_inversion/envs/Harvard-Cannon/gcclassic.rocky+gnu12.minimal.env

# get base project path
cd ../..
export base_proj_dir=$(pwd -P)

do_month () {

    # set start and end dates
    mydate1=$(date -d "${year}${month}01" '+%Y%m%d')
    LastRestartFile=$(date -d "${year}${month}01 + 3month" "+GEOSChem.Restart.%Y%m%d_0000z.nc4")

    # set unique run path
    RunName=${RunNamePrefix}_${mydate1}

    # make a folder for this monthly imi run
    # and copy in config and run script
    cd imi_${mydate1}
    if test -f rerun_imi_output.log; then
        rm rerun_imi_output.log
    fi
    
    # copy run script to here
    cp ../../post/rerun_timeouts/rerun_template.sh rerun.sh

    # copy jacobian rerun script to jacobian run dir
    JacobianDir=${OutputPath}/${RunName}/jacobian_runs
    cp ../../post/rerun_timeouts/rerun_jacobian_simulations_template.sh ${JacobianDir}/rerun_jacobian_simulations.sh

    # edit run script
    sed -i -e "s|my_run_dir|${RunName}|g" rerun.sh
    sed -i -e "s|my_partition|${SchedulerPartition}|g" rerun.sh
    sed -i -e "s|my_output_path|${OutputPath}|g" rerun.sh
    sed -i -e "s|base_proj_dir|${base_proj_dir}|g" rerun.sh

    # rerun jacobian run script
    sed -i -e "s|my_run_dir|${RunName}|g" ${JacobianDir}/rerun_jacobian_simulations.sh
    sed -i -e "s|my_partition|${SchedulerPartition}|g" ${JacobianDir}/rerun_jacobian_simulations.sh
    sed -i -e "s|my_output_path|${OutputPath}|g" ${JacobianDir}/rerun_jacobian_simulations.sh
    sed -i -e "s|base_proj_dir|${base_proj_dir}|g" ${JacobianDir}/rerun_jacobian_simulations.sh
    sed -i -e "s|last_restart_file|${LastRestartFile}|g" ${JacobianDir}/rerun_jacobian_simulations.sh
    
    # submit the test case
    if [ -n "$1" ]; then
        #echo new_sbatch --dependency=afterany:${1} ./rerun.sh
        JOBID=$(new_sbatch --dependency=afterany:${1} ./rerun.sh)
    else
        #echo new_sbatch ./rerun.sh
        JOBID=$(new_sbatch ./rerun.sh)
    fi
    cd ..
    echo $JOBID
    
}

cd cases

year=2018
month=06
prev_job=$(do_month)

for year in 2018; do
for month in 07 08 09 10 11 12; do
prev_job=$(do_month ${prev_job})
sleep 1s
echo $(date -d "${year}${month}01" '+%Y-%m-%d')
done
done

for year in 2019; do
for month in {01..08}; do
prev_job=$(do_month ${prev_job})
sleep 1s
echo $(date -d "${year}${month}01" '+%Y-%m-%d')
done
done
