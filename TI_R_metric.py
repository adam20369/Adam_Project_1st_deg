import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle
import itertools
import timeit as tit
from scipy import integrate
from scipy.linalg import expm
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.linalg import expm
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Coupling_To_Bath import *

###################### R-Metric Check for Tilted-Ising ################################
def RNewMeanMetricTI(eval):
    """
    mean R metric calculation
    :param eval: EigenValues (size-ordered: smallest to biggest)
    :return: r mean value - r= 0.39 poisson, r=0.536 W-D
    """
    S = np.diff(eval)  # returns  array of n-1 (Non-Negative) differences
    r = np.zeros([S.shape[0] - 1])
    for i in range(1, S.shape[0]):
        r[i - 1] = np.divide(min(S[i], S[i - 1]), max(S[i], S[i - 1]))
    R = np.around(r, 5)
    N = np.count_nonzero(R)
    Rmean = np.divide(np.sum(R), N)  # Averaging over No. of non-zero R contributions
    return Rmean

def RunRNewMeanMetricTI(n_TI, h_x, h_z, h_imp, m = 1):
    '''
    Runs the RMeanMetric function on Tilted Ising model
    :param n_TI:
    :param h_x:
    :param h_z:
    :param h_imp:
    :param m:
    :return: Runs the RMeanMetric function on Tilted Ising model
    '''
    eval, evec = Diagonalize(TIOBCNewImpure2(n_TI, 1, h_x, h_z, h_imp, m))
    return RNewMeanMetricTI(eval)

# RunRNewMeanMetricTI(9, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0.1)

def RNewMeanMetricTICUT(eval):
    """
    Cut version of 1/8 of eigenvalues on every side of the spectrum
    :param eval: EigenValues (size-ordered: smallest to biggest)
    :return: r mean value - r= 0.39 poisson, r=0.536 W-D
    """
    S = np.diff(eval)  # returns  array of n-1 (Non-Negative) differences
    r = np.zeros([S.shape[0] - 1])
    for i in range(int(np.divide(S.shape[0],8)), int(7*np.divide(S.shape[0],8))):
        r[i - 1] = np.divide(min(S[i], S[i - 1]), max(S[i], S[i - 1]))
    R = np.around(r, 5)
    N = np.count_nonzero(R)
    Rmean = np.divide(np.sum(R), N)  # Averaging over No. of non-zero R contributions
    return Rmean

def RunRNewMeanMetricTICUT(n_TI, h_x, h_z, h_imp, m = 1): #Run the RMeanMetric function on Tilted Ising model
    '''
    Runs the cut version of RMeanMetric function on Tilted Ising model
    :param n_TI:
    :param h_x:
    :param h_z:
    :param h_imp:
    :param m:
    :return: Runs the cut version of RMeanMetric function on Tilted Ising model
    '''
    eval, evec =Diagonalize(TIOBCNewImpure2(n_TI, 1, h_x, h_z, h_imp, m))
    return RNewMeanMetricTICUT(eval)
#RunRNewMeanMetricTICUT(9, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0.1)
