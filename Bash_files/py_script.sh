#! /bin/bash
#$ -cwd
#$ -e err.txt
#$ -o out.txt
#$ -l mem_free=10G
#$ -pe shared 10
source /etc/profile
module load use.own
python3 PXP_E_B_E_Sparse.py $1 $2 $3 $4

