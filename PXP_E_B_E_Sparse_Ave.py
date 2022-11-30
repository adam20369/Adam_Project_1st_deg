import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PXP_E_B_E_Sparse_Para import *
import numpy.linalg as la

def Sparse_time_combine(seed_max):
    '''
    averages over Sparse time realizations
    :param seed_max: number of realizations
    :return: saves average
    '''
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    data = np.empty((seed_max,len(Time)))
    for j in range(0,seed_max):
        data[j,:]= np.load('Sparse_time_propagation_sample_{}.npy'.format(j)) #creates
    np.save('Sparse_time_propagation_combine.npy', data)
#Sparse_time_combine(seed_max)

def Sparse_time_ave(seed_max):
    '''
    averages over Sparse time realizations
    :param seed_max: number of realizations
    :return: saves average
    '''
    data= np.load('Sparse_time_propagation_combine.npy')
    data_ave = np.mean(data,axis=0)
    np.save('Sparse_time_propagation_ave.npy', data_ave)
#Sparse_time_ave(seed_max)


def Bootstrap(n):
    '''
    Bootstrapping of time propagation samples
    :return: 95% confidence interval upper and lower bounds for each of time steps' average of random samples
    '''
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    lower_upper = np.empty((2,len(Time)))
    data = np.load('Sparse_time_propagation_combine.npy')
    for i in range(0,len(Time)):
        sample = np.random.choice(data[:,i],(seed_max, n), replace=True) # creates [(seed_max No.) x n] matrix of randomly sampled arrays (with return) from the original
        sample_ave = np.mean(sample, axis=0)  # vector of averages sampled from one row of propagation data (random)
        lower_mean = np.quantile(sample_ave, 0.025)
        upper_mean = np.quantile(sample_ave, 0.975)
        lower_upper[0,i] = lower_mean
        lower_upper[1,i] = upper_mean
    np.save('Sparse_time_propagation_errors.npy',lower_upper)
#Bootstrap(seed_max)

