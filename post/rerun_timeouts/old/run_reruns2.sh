#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o "rerun_prod_run.log"
#SBATCH -c 1
#SBATCH -t 0-72:00
#SBATCH -p sapphire
#SBATCH -J main_rerun


# specify imi name, output path, and partition
RunNamePrefix="imi"
#OutputPath="/n/holyscratch01/jacob_lab/jeast/proj/globalinv/prod/output"
OutputPath="/n/holylfs06/SCRATCH/jacob_lab/jeast/proj/globalinv/prod/output"
SchedulerPartition="sapphire"

# make output dir
# mkdir -p $OutputPath

source ../../integrated_methane_inversion/envs/Harvard-Cannon/gcclassic.rocky+gnu12.minimal.env

# get base project path
cd ../..
export base_proj_dir=$(pwd -P)

do_month () {

    # set start and end dates
    mydate1=$(date -d "${year}${month}01" '+%Y%m%d')

    # set unique run path
    RunName=${RunNamePrefix}_${mydate1}

    # make a folder for this monthly imi run
    # and copy in config and run script
    cd imi_${mydate1}
    
    # copy run script to here
    cp ../../post/rerun_timeouts/rerun_template2.sh rerun.sh

    # edit run script
    sed -i -e "s|my_run_dir|${RunName}|g" rerun.sh
    sed -i -e "s|my_partition|${SchedulerPartition}|g" rerun.sh
    sed -i -e "s|my_output_path|${OutputPath}|g" rerun.sh
    sed -i -e "s|base_proj_dir|${base_proj_dir}|g" rerun.sh
    
    # submit the test case
    sbatch ./rerun.sh
    cd ..
    
}

cd cases

for year in 2019; do
for month in 09 10; do
do_month
sleep 1s
echo $(date -d "${year}${month}01" '+%Y-%m-%d')
done
done
