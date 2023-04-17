#!/bin/bash
seed_max='100'
n_PXP='10'
n_TI_max='13'
h_c_max='2'
Sample_no='1000'
Sleep='5'
sd='5' #doesn't matter here
for n_TI in $(seq $n_PXP 1 $n_TI_max)
  do
    for h_c in $(seq 0 0.1 $h_c_max)
      do
        r=$(qstat|grep adamgit|wc -l)
        while [ $r -ge 1495 ]
          do
            echo sleep $Sleep seconds load is $r
            sleep $Sleep
            r=$(qstat|grep adamgit|wc -l)
          done
        qsub -q intel_all.q -N PXP_$n_TI FFT_py_script.sh $n_PXP $n_TI $h_c $sd $Sample_no $seed_max;
      done
  done;

