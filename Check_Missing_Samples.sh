#!/bin/bash

n_PXP='9'
n_TI_max='12'
h_c_max='2'
for n_TI in $(seq $n_TI_max 1 $n_TI_max)
  do
    for h_c in $(seq 0.0 0.1 $h_c_max)
      do
        r=$(find "PXP_$n_PXP""_TI_$n_TI""_True_X_i/h_c_$h_c" -type f | wc -l)
        echo "PXP $n_PXP T_I $n_TI h_c $h_c file count is $r"
        for sample in $(seq 1 1 99)
          do
              f=$(find "PXP_$n_PXP""_TI_$n_TI""_True_X_i/h_c_$h_c""/Sparse_time_propagation_True_X_i_$n_PXP""_$n_TI""_$h_c""_sample_$sample"".npy" -type f | wc -l)
          done
      done
  done;
