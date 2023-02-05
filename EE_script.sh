#!/bin/bash
n_PXP='6'
n_TI='6'
h_c_max='20'
for h_c in $(seq 0 0.1 $h_c_max)
  do
    qsub -q intel_all.q -N EE_$h_c EE_py_script.sh $n_PXP $n_TI $h_c;
  done;

