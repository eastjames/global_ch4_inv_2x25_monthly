#!/bin/bash
clone () {
    git clone https://github.com/geoschem/GCClassic.git
    cd GCClassic
    git checkout 14.2.3
    git submodule update --init --recursive
}


diffit () {
    diff /n/holylfs05/LABS/jacob_lab/Users/jeast/proj/global_ch4_inv_2x25_monthly/prep/gc_global_sensitivities/run-before/global_ch4_mod.F90  GCClassic/src/GEOS-Chem/GeosCore/global_ch4_mod.F90
}

#clone
diffit
