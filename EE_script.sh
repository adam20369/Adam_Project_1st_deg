#!/bin/bash
n_PXP='8'
n_TI='10'
h_c_max='20'
dummy1='1'
dummy2='2'
dummy3='3'
for h_c in $(seq 0 0.1 $h_c_max)
  do
    qsub -q intel_all.q -N EE_$h_c EE_py_script.sh $n_PXP $n_TI $h_c $dummy1 $dummy2 $dummy3;
  done;

