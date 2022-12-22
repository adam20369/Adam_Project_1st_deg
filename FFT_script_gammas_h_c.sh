#!/bin/bash
seed_max='100'
#seed=$(awk 'BEGIN{for(i=1;i<100;i+=1)print i}')
n_PXP='8'
n_TI_max='12'
h_c_max='1'
Sample_no='1000'
sd='5' #doesnt matter in gammas
for n_TI in $(seq 8 1 $n_TI_max)
  do
    for h_c in $(seq 0 0.1 $h_c_max)
      do
         qsub -q intel_all.q -N PXP_$h_c FFT_py_script.sh $n_PXP $n_TI $h_c $sd $Sample_no $seed_max;
      done
  done;

