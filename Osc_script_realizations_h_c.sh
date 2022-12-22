#!/bin/bash
seed_max='100'
n_PXP='10'
n_TI_max='13'
h_c_max='1'
Sample_no='1000'
for n_TI in $(seq 10 1 $n_TI_max)
  do
   for h_c in $(seq 0 0.1 $h_c_max)
    do
      for ((sd=1; sd<$seed_max ; sd++))
        do
          qsub -q intel_all.q -N PXP_$sd Osc_py_script.sh $n_PXP $n_TI $h_c $sd $Sample_no $seed_max;
        done
    done
  done;

