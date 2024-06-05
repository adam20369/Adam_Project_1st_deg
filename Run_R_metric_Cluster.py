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
from Cluster_R_Metric_Para import *

def plotRmetric(n_TI, theta_i, theta_f, res,imp_str,imp_loc): #running the r metric for a range of thetas
    theta=np.linspace(theta_i, theta_f, res)
    r=np.zeros(theta.shape[0])
    iter_count= 0
    #np.seterr(divide='ignore',invalid='ignore')
    for t in np.nditer(theta):
        r[iter_count]=RunRmetric(n_TI, np.sin(t), np.cos(t), imp_str,imp_loc,TIOBCNewImpure) #IMPURE MODEL!!
        iter_count=iter_count+1
    #print(max(r),theta[np.nonzero(r==max(r))]) #maximal value theta!!
    Result_Vec=np.zeros([2,theta.shape[0]])
    Result_Vec[0,:]=theta
    Result_Vec[1,:]=r
    try:
        os.mkdir('R_Metric_TI_{}'.format(n_TI))
    except:
        pass
    np.save(os.path.join('R_Metric_TI_{}'.format(n_TI),'_Imp_Site_{}_Str_{}_Int_{}-{}_res_{}'.format(imp_loc,imp_str,theta_i,theta_f,res)),Result_Vec)
plotRmetric(n_TI, theta_i, theta_f, res,imp_str,imp_loc)