#! /bin/bash
#$ -cwd
#$ -e err.txt
#$ -o out.txt
#$ -l mem_free=35G
#$ -pe shared 7
source /etc/profile
module load use.own
module load anaconda3
source ~/anaconda3/etc/profile.d/conda.sh
conda activate PXP_TI
python3 Cluster_Sparse_True_X_i_Osc_ave_n_error.py $1 $2 $3 $4 $5 $6

