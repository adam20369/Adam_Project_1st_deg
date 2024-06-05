#!/bin/bash

n_PXP='13'
n_TI_max='12'
h_c_max='10'
Sample_no='1000'
seed_max='30'
Sleep='60'
T_max='40'
Step='500'

for n_TI in $(seq 12 1 $n_TI_max)
  do
    for h_c in $(seq 0.0 0.1 $h_c_max)
      do
        r=$(find "PXP_$n_PXP""_TI_$n_TI""_T_max_$T_max""_Step_$Step""/h_c_$h_c" -type f | wc -l)
        echo "PXP $n_PXP T_I $n_TI h_c $h_c ZZ file count is $r"
        l=$(find "PXP_$n_PXP""_TI_$n_TI""_True_X_i""_T_max_$T_max""_Step_$Step""/h_c_$h_c" -type f | wc -l)
        echo "PXP $n_PXP T_I $n_TI h_c $h_c XX file count is $l"
        for seed in $(seq 1 1 $seed_max)
          do
              f=$(find "PXP_$n_PXP""_TI_$n_TI""_T_max_$T_max""_Step_$Step""/h_c_$h_c""/Sparse_time_propagation_$n_PXP""_$n_TI""_$h_c""_T_max_$T_max""_Step_$Step""_sample_$seed"".npy" -type f | wc -l)
              g=$(find "PXP_$n_PXP""_TI_$n_TI""_True_X_i""_T_max_$T_max""_Step_$Step""/h_c_$h_c""/Sparse_time_propagation_True_X_i_$n_PXP""_$n_TI""_$h_c""_T_max_$T_max""_Step_$Step""_sample_$seed"".npy" -type f | wc -l)

              if [ $f = 0 ] || [ $g = 0 ]; then
                r=$(qstat|grep adamgit|wc -l)
                while [ $r -ge 1485 ]
                  do
                    echo sleep $Sleep seconds load is $r
                    sleep $Sleep
                    r=$(qstat|grep adamgit|wc -l)
                  done
                qsub -q barlev.q,intel_all.q@bhn1083,intel_all.q@bhn1084,intel_all.q@bhn1085,intel_all.q@bhn1086,intel_all.q@bhn1087,intel_all.q@bhn23,intel_all.q@bhn24,intel_all.q@bhn26,intel_all.q@bhn27,intel_all.q@sge1026,intel_all.q@sge1029,intel_all.q@sge1030,intel_all.q@sge1031,intel_all.q@sge1032,intel_all.q@sge1033,intel_all.q@sge1076,intel_all.q@sge1077,intel_all.q@sge1078,intel_all.q@sge1041,intel_all.q@sge1042,intel_all.q@sge1043,intel_all.q@sge1044,intel_all.q@sge130,intel_all.q@sge131,intel_all.q@sge132,intel_all.q@sge133,intel_all.q@sge134,intel_all.q@sge15,intel_all.q@sge16,intel_all.q@sge17,intel_all.q@sge226,intel_all.q@sge232,intel_all.q@sge235,intel_all.q@sge236,intel_all.q@sge241,intel_all.q@sge243,intel_all.q@sge244,intel_all.q@sge245,intel_all.q@sge246,intel_all.q@sge247,intel_all.q@sge248,intel_all.q@sge249 -N PXP_$seed Osc_py_script.sh $n_PXP $n_TI $h_c $seed $Sample_no $seed_max;
              fi
          done
      done
  done;

#intel_all.q@bhn1083,intel_all.q@bhn1084,intel_all.q@bhn1085,intel_all.q@bhn1086,intel_all.q@bhn1087,intel_all.q@bhn23,intel_all.q@bhn24,intel_all.q@bhn26,intel_all.q@bhn27,intel_all.q@sge1026,intel_all.q@sge1029,intel_all.q@sge1030,intel_all.q@sge1031,intel_all.q@sge1032,intel_all.q@sge1033,intel_all.q@sge1076,intel_all.q@sge1077,intel_all.q@sge1078,intel_all.q@sge1041,intel_all.q@sge1042,intel_all.q@sge1043,intel_all.q@sge1044,intel_all.q@sge130,intel_all.q@sge131,intel_all.q@sge132,intel_all.q@sge133,intel_all.q@sge134,intel_all.q@sge15,intel_all.q@sge16,intel_all.q@sge17,intel_all.q@sge226,intel_all.q@sge232,intel_all.q@sge235,intel_all.q@sge236,intel_all.q@sge241,intel_all.q@sge243,intel_all.q@sge244,intel_all.q@sge245,intel_all.q@sge246,intel_all.q@sge247,intel_all.q@sge248,intel_all.q@sge249