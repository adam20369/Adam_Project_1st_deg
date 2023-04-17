#!/bin/bash
seed_max='100'
n_PXP='10'
n_TI_max='12'
h_c_max='3'
Sample_no='1000'
Sleep='10'

for n_TI in $(seq $n_PXP 1 $n_TI_max)
  do
   for h_c in $(seq 2.1 0.1 $h_c_max)
    do
      for ((sd=1; sd<100 ; sd++))
        do
          r=$(qstat|grep adamgit|wc -l)
          while [ $r -ge 1490 ]
            do
              echo sleep $Sleep seconds load is $r
              sleep $Sleep
              r=$(qstat|grep adamgit|wc -l)
            done
          qsub -q intel_all.q@bhn1083,intel_all.q@bhn1084,intel_all.q@bhn1085,intel_all.q@bhn1086,intel_all.q@bhn1087,intel_all.q@bhn23,intel_all.q@bhn24,intel_all.q@bhn26,intel_all.q@bhn27,intel_all.q@sge1026,intel_all.q@sge1029,intel_all.q@sge1030,intel_all.q@sge1031,intel_all.q@sge1032,intel_all.q@sge1033,intel_all.q@sge1076,intel_all.q@sge1077,intel_all.q@sge1078,intel_all.q@sge1041,intel_all.q@sge1042,intel_all.q@sge1043,intel_all.q@sge1044,intel_all.q@sge1045,intel_all.q@sge130,intel_all.q@sge131,intel_all.q@sge132,intel_all.q@sge133,intel_all.q@sge134,intel_all.q@sge15,intel_all.q@sge16,intel_all.q@sge17,intel_all.q@sge226,intel_all.q@sge232,intel_all.q@sge233,intel_all.q@sge234,intel_all.q@sge235,intel_all.q@sge236,intel_all.q@sge241,intel_all.q@sge243,intel_all.q@sge244,intel_all.q@sge245,intel_all.q@sge246,intel_all.q@sge247,intel_all.q@sge248,intel_all.q@sge249 -N PXP_$sd Osc_py_script_alternative.sh $n_PXP $n_TI $h_c $sd $Sample_no $seed_max;
        done
    done
  done;

#intel_all.q@bhn1083,intel_all.q@bhn1084,intel_all.q@bhn1085,intel_all.q@bhn1086,intel_all.q@bhn1087,intel_all.q@bhn23,intel_all.q@bhn24,intel_all.q@bhn26,intel_all.q@bhn27,intel_all.q@sge1026,intel_all.q@sge1029,intel_all.q@sge1030,intel_all.q@sge1031,intel_all.q@sge1032,intel_all.q@sge1033,intel_all.q@sge1076,intel_all.q@sge1077,intel_all.q@sge1078,intel_all.q@sge1041,intel_all.q@sge1042,intel_all.q@sge1043,intel_all.q@sge1044,intel_all.q@sge1045,intel_all.q@sge130,intel_all.q@sge131,intel_all.q@sge132,intel_all.q@sge133,intel_all.q@sge134,intel_all.q@sge15,intel_all.q@sge16,intel_all.q@sge17,intel_all.q@sge226,intel_all.q@sge232,intel_all.q@sge233,intel_all.q@sge234,intel_all.q@sge235,intel_all.q@sge236,intel_all.q@sge241,intel_all.q@sge243,intel_all.q@sge244,intel_all.q@sge245,intel_all.q@sge246,intel_all.q@sge247,intel_all.q@sge248,intel_all.q@sge249