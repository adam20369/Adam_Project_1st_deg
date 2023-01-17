#!/bin/bash
seed_max='100'
n_PXP='9'
n_TI_max='12'
h_c_max='1.4'
Sample_no='1000'
for n_TI in $(seq 9 1 $n_TI_max)
  do
   for h_c in $(seq 1.1 0.1 $h_c_max)
    do
      for ((sd=1; sd<$seed_max ; sd++))
        do
          qsub -q barlev.q -N PXP_$sd Osc_py_script.sh $n_PXP $n_TI $h_c $sd $Sample_no $seed_max;
        done
    done
  done;

