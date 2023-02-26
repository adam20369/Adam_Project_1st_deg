#!/bin/bash

n_PXP='9'
n_TI_max='11'
h_c_max='2'
Sample_no='1000'
seed_max='100'

for n_TI in $(seq $n_TI_max 1 $n_TI_max)
  do
    for h_c in $(seq 0.0 0.1 $h_c_max)
      do
        r=$(find "PXP_$n_PXP""_TI_$n_TI""_True_X_i/h_c_$h_c" -type f | wc -l)
        echo "PXP $n_PXP T_I $n_TI h_c $h_c file count is $r"
        for seed in $(seq 1 1 99)
          do
              f=$(find "PXP_$n_PXP""_TI_$n_TI""_True_X_i/h_c_$h_c""/Sparse_time_propagation_True_X_i_$n_PXP""_$n_TI""_$h_c""_sample_$seed"".npy" -type f | wc -l)
              if [ $f = 0 ]; then
                 qsub -q intel_all.q -N PXP_$seed Osc_py_script.sh $n_PXP $n_TI $h_c $seed $Sample_no $seed_max;
              fi
          done
      done
  done;
