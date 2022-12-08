#!/bin/bash
seed_max='100'
#seed=$(awk 'BEGIN{for(i=1;i<100;i+=1)print i}')  
n_PXP='10'
n_TI_max='10'
h_c='0.2'
Sample_no='100'

for n_TI in $(seq 0 1 $n_TI_max)
  do
    for ((sd=1; sd<$seed_max ; sd++)) 
      do
        qsub -q barlev.q -N PXP_$sd Osc_py_script.sh $n_PXP $n_TI $h_c $sd $Sample_no $seed_max;
    done
  done;

