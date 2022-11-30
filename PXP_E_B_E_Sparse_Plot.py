import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PXP_Entry_By_Entry import *
from PXP_E_B_E_Sparse import *
from Cluster_PXP_E_B_E_Sparse_Para import *
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

data_ave = np.load('Sparse_time_propagation_ave.npy')
data_errors = np.load('Sparse_time_propagation_errors.npy')
Time = np.linspace(T_start, T_max, T_step, endpoint=True)
plt.errorbar(Time[:200], data_ave[:200], yerr= data_errors[:200], color='b', marker='o')
#error = bootstrap(data_ave, np.std, confidence_level=0.95, random_state=1, method='percentile')
