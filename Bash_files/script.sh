#!/bin/bash
seed_max='100'
seed=$(awk 'BEGIN{for(i=1;i<$seed_max;i+=1)print i}')  
n_PXP='7'
n_TI='7'
h_c='0.2'

for sd in $seed 
  do
    qsub -q barlev.q -N PXP_$sd py_script.sh $n_PXP $n_TI $h_c $sd $seed_max;
  done;

