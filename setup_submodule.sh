#!/bin/bash
# https://git-scm.com/book/en/v2/Git-Tools-Submodules
git submodule add git@github.com:eastjames/integrated_methane_inversion.git
git config -f .gitmodules submodule.integrated_methane_inversion.branch feature/global_inversion
git submodule update --remote
cd integrated_methane_inversion
git checkout feature/global_inversion

# get changes with 
git submodule update --remote --merge

# push parent repo changes with
get push --recurse-submodules=check
#or do
git config push.recurseSubmodules check
