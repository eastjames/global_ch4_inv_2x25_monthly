# global_ch4_inv_2x25_monthly
Repository for global inversion of CH4 fluxes using TROPOMI+GOSAT blended retrievals, GEOS-Chem at 2x2.5 resolution, monthly native-resolution state vector, from 2018-2023

### cloning
`git clone --recurse-submodules git@github.com:eastjames/global_ch4_inv_2x25_monthly.git`


### contents
* `integrated_methane_inversion`
    * custom imi branch used to run jacobian simulations

* `prep`
    * scripts and modified geos-chem files so that geos-chem can apply TROPOMI+GOSAT operator in-line and output at obs locations

* `setup_run_test.sh`
    * script to setup and run jacobian simulations

* `setup_submodule.sh`
    * commands to set up submodules, safe to ignore

### view outputs
`prep/testing/TestingCheck.ipynb`
* should run right out of the box for test case

### how to run
1. edit `setup_run_test.sh`
    * RunNamePrefix
    * OutputPath
    * SchedulerPartition

2. `./setup_run_test.sh`

3. script will
    * get files needed from external repo
    * setup imi directories for monthly jacobian runs
        * each month has its own imi dir and output dir
        * not using kalman filter because not doing inversion, just running jacobians
    * edit imi config and run scripts
    * submit each monthly imi to SLURM scheduler
