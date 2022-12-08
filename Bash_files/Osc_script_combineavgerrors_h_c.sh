#!/bin/bash
seed_max='100'
#seed=$(awk 'BEGIN{for(i=1;i<100;i+=1)print i}')  
n_PXP='10'
n_TI='10'
h_c_max='1'
Sample_no='100'
sd='5'
for h_c in $(seq 0 0.1 $h_c_max)
  do
    qsub -q barlev.q -N PXP_$sd Osc_py_script.sh $n_PXP $n_TI $h_c $sd $Sample_no $seed_max;
  done;

