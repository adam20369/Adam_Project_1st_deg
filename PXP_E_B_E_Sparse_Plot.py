import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PXP_Entry_By_Entry import *
from PXP_E_B_E_Sparse import *
import O_z_Oscillations as Ozosc
import numpy.linalg as la
from time import time
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import comb
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
T_start = 0
T_max = 200
T_step = 2000

n_PXP=7
n_TI=7
h_c=0.9

data_ave = np.load('Sparse_time_propagation_ave_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c))
data_errors = np.load('Sparse_time_propagation_errors_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c))
data_errors_fin_0 = data_ave-data_errors[0,:] #DATA ERRORS NEED TO BE +/- from Null
data_errors_fin_1 = data_errors[1,:]-data_ave #DATA ERRORS NEED TO BE +/- from Null
data_errors_fin= np.empty((2,len(data_errors_fin_0)))
data_errors_fin[0,:]=data_errors_fin_0
data_errors_fin[1,:]=data_errors_fin_1

#data_errors_fin= np.fabs(data_errors_fin)

Time = np.linspace(T_start, T_max, T_step, endpoint=True)
plt.errorbar(Time[:200], data_ave[:200], yerr= data_errors_fin[:,:200], fmt='bs-', ecolor='r',markersize=2,elinewidth=4)
#error = bootstrap(data_ave, np.std, confidence_level=0.95, random_state=1, method='percentile')
plt.show()