import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from matplotlib.lines import Line2D
from PXP_TI_Bath import *


def Plot_R_metric(n_TI,theta_i,theta_f,res,imp_str,imp_loc):
    data = np.load('R_Metric_TI_{}/_Imp_Site_{}_Str_{}_Int_{}-{}_res_{}.npy'.format(n_TI,imp_loc,imp_str,theta_i,theta_f,res))
    theta=data[0,:]
    r=data[1,:]
    plt.plot(theta, r,  marker= '.',linestyle = "", color='C4')
    plt.title('Angle dependence of r-metric, {} Atoms, {} Imp Loc, {} Imp Str'.format(n_TI,np.around(imp_loc,5),np.around(imp_str,5)))
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\langle r \rangle$")
    plt.ylim(0.34,0.55)
    return plt.show()


    #def Find_Min_Max_R():


narray = np.arange(0.0, 0.1, 0.01)
narray2 = np.arange(0.1, 0.6, 0.1)
arraytot = np.concatenate((narray, narray2), axis=0)
#for i in np.nditer(arraytot):
    #Plot_R_metric(11, 0.05, 3.25, 1500, np.around(i,2), 1)
