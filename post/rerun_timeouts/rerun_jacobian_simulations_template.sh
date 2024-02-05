#!/bin/bash
#SBATCH -J my_run_dir
#SBATCH -N 1

### Run directory
RUNDIR=$(pwd -P)

### Get current task ID
x=${SLURM_ARRAY_TASK_ID}

### Add zeros to the cluster Id
if [ $x -lt 10 ]; then
    xstr="00000${x}"
elif [ $x -lt 100 ]; then
    xstr="0000${x}"
elif [ $x -lt 1000 ]; then
    xstr="000${x}"
elif [ $x -lt 10000 ]; then
    xstr="00${x}"
elif [ $x -lt 100000 ]; then
    xstr="0${x}"
else
    xstr="${x}"
fi

output_log_file=base_proj_dir/cases/my_run_dir/rerun_imi_output.log

# This checks for the presence of the error status file. If present, this indicates 
# a prior jacobian exited with an error, so this jacobian will not run
FILE=.error_status_file.txt
if test -f "$FILE"; then
    echo "$FILE exists. Exiting."
    echo "jacobian simulation: ${xstr} exited without running." >> $output_log_file
    exit 1
fi

### Run GEOS-Chem in the directory corresponding to the cluster Id
cd  ${RUNDIR}/my_run_dir_${xstr}

# check for Restart file
# rerun if it is not there
LastRestartFile=last_restart_file
cd Restarts
if test -f "$LastRestartFile"; then
    echo "Not re-running jacobian simulation: ${xstr}" >> $output_log_file
    exit 0
else
    cd ..
    echo "Re-running jacobian simulation: ${xstr}" >> $output_log_file
    ./my_run_dir_${xstr}.run
    exit 0
fi

# save the exit code of the jacobian simulation cmd
retVal=$?

# Check whether the jacobian finished successfully. If not, write to a hidden file. 
# The presence of the .error_status_file.txt indicates whether an error ocurred. 
# This is needed because scripts that set off sbatch jobs have no knowledge of 
# whether the job finished successfully.
if [ $retVal -ne 0 ]; then
    rm -f .error_status_file.txt
    echo "Error Status: $retVal" > ../.error_status_file.txt
    echo "jacobian simulation: ${xstr} exited with error code: $retVal" >> $output_log_file
    echo "Check the log file in the ${RUNDIR}/my_run_dir_${xstr} directory for more details." >> $output_log_file
    exit $retVal
fi

echo "finished jacobian simulation: ${xstr}" >> $output_log_file

exit 0
