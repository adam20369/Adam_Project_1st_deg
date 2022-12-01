#!/bin/bash
seed_max='100'
seed=$(awk 'BEGIN{for(i=1;i<100;i+=1)print i}')  
n_PXP='7'
n_TI='7'
h_c='0.2'
Sample_no='100'

for sd in $seed 
  do
    qsub -q barlev.q -N PXP_$sd py_script.sh $n_PXP $n_TI $h_c $sd $Sample_no;
  done;

