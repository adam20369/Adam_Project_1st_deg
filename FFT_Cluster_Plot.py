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
n_PXP=8
T_start = 0
T_max = 200
T_step = 2000
h_c_max=1
n_TI_max=12
n_TI=np.linspace(8,n_TI_max,5,endpoint=True)
h_c=np.linspace(0, h_c_max, 11,endpoint=True)
data_ave= np.empty((len(h_c)))
data_errors= np.empty((2,(len(h_c))))
for j in np.nditer(n_TI):
    for i in np.nditer(h_c):
        data_ave[int(i*10)]=np.load('Gamma_ave_{}_{}_{}.npy'.format(n_PXP,int(j),np.round(i,2)))
        print(data_ave)
        data_errors[:,int(i*10)]=np.load('Gamma_errors_{}_{}_{}.npy'.format(n_PXP,int(j),np.round(i,2)))
        print(data_errors)
        data_errors_fin_0 = data_ave - data_errors[0, :]  # DATA ERRORS NEED TO BE +/- from Null
        data_errors_fin_1 = data_errors[1, :] - data_ave  # DATA ERRORS NEED TO BE +/- from Null
        data_errors_fin = np.empty((2, len(data_errors_fin_0)))
        data_errors_fin[0, :] = data_errors_fin_0/data_ave[0] #Scaled errors as to gamma/gamma_0
        data_errors_fin[1, :] = data_errors_fin_1/data_ave[0] #Scaled errors as to gamma/gamma_0
        scaled_data_ave= data_ave/data_ave[0] #Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin[:, :], fmt='bs-', ecolor='b', markersize=2, capsize=3, capthick=3)
plt.title(r'Normalized damping for {} PXP atoms vs $h_c$ strength'.format(n_PXP))
plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
plt.show()

def Plot_Vs_TI_No():
    return