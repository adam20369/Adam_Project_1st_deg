#!/bin/bash
n_PXP='8'
n_TI_max='8'
h_c_max='20'
dummy1='1'
dummy2='2'
dummy3='3'
Sleep='60'
for n_TI in $(seq 8 1 $n_TI_max)
  do
    r=$(find "EE_PXP_$n_PXP""_TI_$n_TI/" -type f | wc -l)
    echo "PXP $n_PXP T_I $n_TI ZZ Entanglement file count is $r"
    l=$(find "EE_PXP_$n_PXP""_TI_$n_TI""_True_X_i/" -type f | wc -l)
    echo "PXP $n_PXP T_I $n_TI XX Entanglement file count is $l"
    for h_c in $(seq 0.0 0.1 $h_c_max)
      do
        f=$(find "EE_PXP_$n_PXP""_TI_$n_TI""/Entanglement_h_c_$h_c"".npy" -type f | wc -l)
        g=$(find "EE_PXP_$n_PXP""_TI_$n_TI""_True_X_i""/Entanglement_h_c_$h_c""_True_X_i.npy" -type f | wc -l)

        if [ $f = 0 ] || [ $g = 0 ]; then
          r=$(qstat|grep adamgit|wc -l)
          while [ $r -ge 1495 ]
            do
              echo sleep $Sleep seconds load is $r
              sleep $Sleep
              r=$(qstat|grep adamgit|wc -l)
            done
          qsub -q intel_all.q@bhn1083,intel_all.q@bhn1084,intel_all.q@bhn1085,intel_all.q@bhn1086,intel_all.q@bhn1087,intel_all.q@bhn23,intel_all.q@bhn24,intel_all.q@bhn26,intel_all.q@bhn27,intel_all.q@sge1026,intel_all.q@sge1029,intel_all.q@sge1030,intel_all.q@sge1031,intel_all.q@sge1032,intel_all.q@sge1033,intel_all.q@sge1076,intel_all.q@sge1077,intel_all.q@sge1078,intel_all.q@sge1041,intel_all.q@sge1042,intel_all.q@sge1043,intel_all.q@sge1044,intel_all.q@sge130,intel_all.q@sge131,intel_all.q@sge132,intel_all.q@sge133,intel_all.q@sge134,intel_all.q@sge15,intel_all.q@sge16,intel_all.q@sge17,intel_all.q@sge226,intel_all.q@sge232,intel_all.q@sge235,intel_all.q@sge236,intel_all.q@sge241,intel_all.q@sge243,intel_all.q@sge244,intel_all.q@sge245,intel_all.q@sge246,intel_all.q@sge247,intel_all.q@sge248,intel_all.q@sge249 -N EE_$h_c EE_py_script.sh $n_PXP $n_TI $h_c $dummy1 $dummy2 $dummy3;
        fi
     done
  done;

#intel_all.q@bhn1083,intel_all.q@bhn1084,intel_all.q@bhn1085,intel_all.q@bhn1086,intel_all.q@bhn1087,intel_all.q@bhn23,intel_all.q@bhn24,intel_all.q@bhn26,intel_all.q@bhn27,intel_all.q@sge1026,intel_all.q@sge1029,intel_all.q@sge1030,intel_all.q@sge1031,intel_all.q@sge1032,intel_all.q@sge1033,intel_all.q@sge1076,intel_all.q@sge1077,intel_all.q@sge1078,intel_all.q@sge1041,intel_all.q@sge1042,intel_all.q@sge1043,intel_all.q@sge1044,intel_all.q@sge130,intel_all.q@sge131,intel_all.q@sge132,intel_all.q@sge133,intel_all.q@sge134,intel_all.q@sge15,intel_all.q@sge16,intel_all.q@sge17,intel_all.q@sge226,intel_all.q@sge232,intel_all.q@sge235,intel_all.q@sge236,intel_all.q@sge241,intel_all.q@sge243,intel_all.q@sge244,intel_all.q@sge245,intel_all.q@sge246,intel_all.q@sge247,intel_all.q@sge248,intel_all.q@sge249