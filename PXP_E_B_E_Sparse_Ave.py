import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PXP_E_B_E_Sparse_Para import *
import numpy.linalg as la

def Sparse_time_ave(seed_max):
    '''
    averages over Sparse time realizations
    :param seed_max: number of realizations
    :return: saves average
    '''
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    data = np.empty((seed_max,len(Time)))
    for j in range(0,seed_max):
        data[j,:]= np.load('Sparse_time_propagation_sample_{}.npy'.format(j))
    data_ave = np.sum(data,axis=0)/len(Time)
    np.save('Sparse_time_propagation_ave.npy', data_ave)


