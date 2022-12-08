#! /bin/bash
#$ -cwd
#$ -e err.txt
#$ -o out.txt
#$ -l mem_free=10G
#$ -pe shared 10
source /etc/profile
module load use.own
python3 FFT_Cluster.py $1 $2 $3 $4 $5 $6
