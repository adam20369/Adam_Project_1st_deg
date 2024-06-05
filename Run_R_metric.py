import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from matplotlib.lines import Line2D
from PXP_TI_Bath import *

def plotRmetric(n_TI, theta_i, theta_f, res,imp_str,imp_loc): #running the r metric for a range of thetas
    theta=np.linspace(theta_i, theta_f, res)
    r=np.zeros(theta.shape[0])
    iter_count= 0
    #np.seterr(divide='ignore',invalid='ignore')
    for t in np.nditer(theta):
        r[iter_count]=RunRmetric(n_TI, np.sin(t), np.cos(t), imp_str,imp_loc,TIOBCNewImpure) #IMPURE MODEL!!
        iter_count=iter_count+1
    #print(max(r),theta[np.nonzero(r==max(r))]) #maximal value theta!!
    Stupidly_chosen = RunRmetric(n_TI, np.sin(0.485*np.pi), np.cos(0.485*np.pi),imp_str, imp_loc,TIOBCNewImpure)  # IMPURE MODEL!!
    #print(Stupidly_chosen)  #R of the weird value of theta I initially chose 0.485*pi!!
    plt.plot(theta, r,  marker= '.',linestyle = "", color='C4')
    plt.title('Angle dependence of r-metric, {} Atoms, {} Imp Loc, {} Imp Str'.format(n_TI,np.around(imp_loc,5),np.around(imp_str,5)))
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\langle r \rangle$")
    #plt.ylim(0.37,0.55)
    return plt.show()

