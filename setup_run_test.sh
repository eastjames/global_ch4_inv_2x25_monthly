#!/bin/bash


# specify imi name, output path, and partition
RunName="imi_test"
OutputPath="/path/to/imi/output"
SchedulerPartition="sapphire"

# get base project path
export base_proj_dir=$(pwd -P)

# clone external repos that
# have modified GC code we want
pushd prep
./get_modified_gc.sh
popd

do_month () {

    # set start and end dates
    mydate1=$(date -d "${year}${month}01" '+%Y-%m-%d')
    mydate2=$(date -d "${year}${month}01 + 1month" '+%Y-%m-%d')

    # make a folder for this monthly imi run
    # and copy in config and run script
    mkdir -p imi_${mydate1}
    cd imi_${mydate1}
    cp ../../integrated_methane_inversion/config_template.yml config.yml
    cp ../../integrated_methane_inversion/run_imi_template.sh run_imi.sh
    
    # link files to mirror imi dir
    for f in docs envs GCClassic LICENSE.md README.md resources src; do
        ln -s $f .
    done

    # edit config
    sed -i -e "s|my_start_date|${mydate1}|g" config.yml
    sed -i -e "s|my_end_date|${mydate2}|g" config.yml
    sed -i -e "s|my_run_name|${RunName}|g" config.yml
    sed -i -e "s|my_output_path|${OutputPath}|g" config.yml
    sed -i -e "s|base_proj_dir|${base_proj_dir}|g" config.yml
    sed -i -e "s|my_partition|${SchedulerPartition}|g" config.yml

    # edit run dir
    sed -i -e "s|my_partition|${SchedulerPartition}|g" run_imi.sh
    
    # submit the test case
    #sbatch ./run_imi_testing.sh
    cd ..
    
}


mkdir -p cases
cd cases


for month in {01..12}; do
for year in 2018 2019; do
do_month
echo ''
done
done