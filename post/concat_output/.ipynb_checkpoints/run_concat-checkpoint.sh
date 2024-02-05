#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o "concat_output.log"
#SBATCH --mem 16000
#SBATCH -c 2
#SBATCH -t 0-01:00
#SBATCH -p my_partition
#SBATCH -J concat_run_imi
#####SBATCH --array=0-3753%4000

#TID=$SLURM_ARRAY_TASK_ID
TID=0
dir_num=$(printf "%06d" ${TID})

mydate=my_date
OutputDir=my_output_path
pushd ${OutputDir}/imi_${mydate}/jacobian_runs/imi_${mydate}_${dir_num}/OutputDir

#for mm in 0 1 2; do
for mm in 0; do
    yyyymm=$(date -d "${mydate} + ${mm}month" '+%Y%m')
    out_file=output_${yyyymm}01.nc

    for f in output_${yyyymm}??T??.nc; do
        echo $f
        ncecat -O $f $f
        ncpdq -O -a nobs,record ${f} ${f}
    done

    ncrcat output_${yyyymm}??T??.nc $out_file
    ncwa -O -a record $out_file $out_file
    #ncks -4 -L 1 $out_file $out_file

    echo "/usr/bin/rm output_${yyyymm}??T??.nc"
    #/usr/bin/rm output_${yyyymm}??T??.nc
done
popd
