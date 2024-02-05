#!/bin/bash

script=./test.sh
test () {
    JOBID=$(./submit.sh -p seas_compute ${script})
    echo $JOBID
}
out=$(test foo)
echo $out

