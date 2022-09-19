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


def Subspace_reduced_Zi(n_PXP,j,st,i):
    '''
    reducing the Zi matrix by removing rows/cols with only zeros
    :param n_PXP: No. of PXP OBC atoms (general)
    :param j: impurity site
    :param st: impurity strength
    :return: reduced Zi matrix, without rows/cols with only zeros
    '''
    full_dim_proj_PXP= Subspace_PXP(n_PXP,j,st)
    red_dim_proj_Zi=Z_generali(n_PXP,i)[:,~np.all(full_dim_proj_PXP==0,axis=1)]
    red_dim_proj_Zi=red_dim_proj_Zi[~np.all(full_dim_proj_PXP==0,axis=1),:]
    return red_dim_proj_Zi

def Subspace_reduced_O_z(n_PXP,j,st):
    '''
    reducing the Zi matrix by removing rows/cols with only zeros
    :param n_PXP: No. of PXP OBC atoms (general)
    :param j: impurity site
    :param st: impurity strength
    :return: reduced Zi matrix, without rows/cols with only zeros
    '''
    full_dim_proj_PXP= Subspace_PXP(n_PXP,j,st)
    red_dim_proj_O_z=O_znew(n_PXP)[:,~np.all(full_dim_proj_PXP==0,axis=1)]
    red_dim_proj_O_z=red_dim_proj_O_z[~np.all(full_dim_proj_PXP==0,axis=1),:]
    return red_dim_proj_O_z

def fig2ASubspc_PXP_Impure_Z_i(n_PXP,j,st,i=1):
    '''
     Figure 2 plot of Pxp TI TOTAL Ham
    :param n_PXP: number of PXP atoms
    :param j: site of impurity
    :param st: strength of impurity
    :param i: index of Z_i
    :return: plot
    '''
    Eval, Evec = Diagonalize(Subspace_reduced_PXP(n_PXP,j,st))
    ExpectationArray = np.diag(np.matmul(np.conjugate(np.transpose(Evec)),np.matmul(Subspace_reduced_Zi(n_PXP,j,st,i),Evec))) # outputs an array of <n|Z|n>'s
    plt.scatter(Eval, ExpectationArray, color='r',marker='o', s=5)
    plt.title(r"$\langle Z${}$\rangle$ Vs. Energy for {} atoms, PXP OBC impure (impurity in {}th site) ".format(i,n_PXP,j))
    plt.xlabel("Energy")
    plt.ylabel(r"$\langle Z${}$\rangle$".format(i))
    plt.show()
    return

def fig2ASubspc_PXP_Impure_O_z(n_PXP,j,st):
    '''
     Figure 2 plot of Pxp TI TOTAL Ham
    :param n_PXP: number of PXP atoms
    :param j: site of impurity
    :param st: strength of impurity
    :return: plot
    '''
    Eval, Evec = Diagonalize(Subspace_reduced_PXP(n_PXP,j,st))
    ExpectationArray = np.diag(np.matmul(np.conjugate(np.transpose(Evec)),np.matmul(Subspace_reduced_O_z(n_PXP,j,st),Evec))) # outputs an array of <n|Z|n>'s
    plt.scatter(Eval, ExpectationArray, color='r',marker='o', s=5)
    plt.title(r"$\langle O_z\rangle$ Vs. Energy for {} atoms, PXP OBC " .format(n_PXP,j))
    plt.xlabel("Energy")
    plt.ylabel(r"$\langle  O_z\rangle$")
    plt.show()
    return

def fig2APXP_TI(n_PXP, n_TI, h_c, i=1,Coupmat=Z_i, J=1,h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi) ,h_imp=0.01):
    '''
    OLD Figure 2 plot of Pxp TI TOTAL Ham, needs rebuilding!!!!!
    :param n_PXP: number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param i: index of Z_i
    :param Coupmat: Coupling nature
    :param J: Ising strength
    :param h_x: X direction strength
    :param h_z: Z direction strength
    :param h_imp: TI Impurity strength
    :return: plot
    '''
    Eval, Evec = Diagonalize(PXPBathHam2(n_PXP,n_TI,Coupmat,J,h_x,h_z,h_c,h_imp,m=1))
    ExpectationArray = np.diag(np.matmul(np.conjugate(np.transpose(Evec)),np.matmul(Z_generali(np.add(n_PXP,n_TI),i),Evec))) # outputs an array of <n|Z|n>'s
    plt.scatter(Eval,ExpectationArray, color='b',marker='o')
    plt.show()
    return

# Do I need infinite temperature average of an operator?
