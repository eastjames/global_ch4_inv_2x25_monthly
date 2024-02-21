#!/bin/bash

#SBATCH -o "process.log"
#SBATCH -c 1
#SBATCH -t 7-00:00
#SBATCH -p seas_compute
#SBATCH -J main


do_case () {


    mydate=$(date -d "${year}${month}01" '+%Y%m%d')

    for mm in 0 1 2; do

        yyyymm=$(date -d "${mydate} + ${mm}month" '+%Y%m')
        imi_case=imi_${mydate}
        
        sbatch \
        --array=0-3753%3754 \
        -c 64 \
        -t 0-00:30 \
        --mem 256gb \
        -p seas_compute,sapphire,serial_requeue \
        -J ${imi_case}-${yyyymm} \
        -o logs/slurm_${imi_case}_${yyyymm}-%j.out \
        -W ./submit_process_jacobian_runs.sh ${yyyymm} ${imi_case} ${input_dir}

    done
    
}

input_dir='/n/holyscratch01/jacob_lab/jeast/proj/globalinv'

for month in {06..12}; do
for year in 2018; do
echo $(date -d "${year}${month}01" '+%Y-%m-%d')
do_case
done
done

for month in {01..08}; do
for year in 2019; do
echo $(date -d "${year}${month}01" '+%Y-%m-%d')
do_case
done
done

input_dir='/n/holylfs06/SCRATCH/jacob_lab/jeast/proj/globalinv/'
for month in {09..12}; do
for year in 2019; do
echo $(date -d "${year}${month}01" '+%Y-%m-%d')
do_case
done
done

for month in {01..12}; do
for year in {2020..2022}; do
echo $(date -d "${year}${month}01" '+%Y-%m-%d')
do_case
done
done

for month in {01..07}; do
for year in 2023; do
echo $(date -d "${year}${month}01" '+%Y-%m-%d')
do_case
done
done
