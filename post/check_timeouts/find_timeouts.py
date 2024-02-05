#!/n/home06/jeast/.conda/envs/jpy01/bin/python

import subprocess
import numpy as np

cmd = [
    'sacct',
    "--format=JobID%30,State,Nodelist"
    # | grep --invert-match \.batch | grep --invert-match \.extern | grep --invert-match "\.0 " | tail'
]


ret = subprocess.run(cmd, capture_output=True)
so_rows = ret.stdout.decode().split('\n')
filter_out = [
    '.batch',
    '.extern',
    'RUNNING'
]
for fo in filter_out:
   so_rows = list(filter(lambda x: fo not in x, so_rows))
so_rows = list(map(str.split, so_rows))
so_rows = list(filter(lambda x: len(x) == 3, so_rows))

def timeout_nodes(inlist):
    return inlist[1] == 'TIMEOUT'

def completed_nodes(inlist):
    return inlist[1] == 'COMPLETED'


def get_list(fxn, sacct_out):
    timeout_rows = list(filter(fxn, sacct_out))
    timeout_rows = np.array(list(map(lambda x: x[2], timeout_rows)), dtype=str)
    all_timeout_nodes = np.unique(timeout_rows)
    return all_timeout_nodes


tn = get_list(timeout_nodes, so_rows)
cn = get_list(completed_nodes, so_rows)
itsxn = np.intersect1d(tn, cn)

np.savetxt('timeout_nodes.txt', tn, delimiter='\n', fmt='%11s')
np.savetxt('completed_nodes.txt', cn, delimiter='\n', fmt='%11s')
np.savetxt('both_nodes.txt', itsxn, delimiter='\n', fmt='%11s')

