import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from matplotlib.lines import Line2D
from PXP_TI_Bath import *

def plotRmetric(n_TI, theta_i, theta_f, res):
    theta=np.linspace(theta_i, theta_f, res)
    for t in np.nditer(theta):
        r=RunRmetric(n_TI, np.sin(t), np.cos(t), TIOBCNewImpure) #IMPURE MODEL!!
        plt.plot(t, r,  marker= '.', color='C4')
    plt.title('Angle dependance of r-metric, {} Atoms'.format(n_TI))
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\langle r \rangle$")
# plotRmetric(10, 1, 2, 500)
# plt.show()
