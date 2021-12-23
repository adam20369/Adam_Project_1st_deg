import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from PXP_TI_Bath import *
from Coupling_To_Bath import *
#TODO don't forget to get rid of one of them
def plotFig2A(m,n,p,Hamiltonian,Op): #plotting for 3 atom sizes
    Eval1, Evec1 = EvecEval(SubspaceMat(m,Hamiltonian(m)))
    Eval2, Evec2 = EvecEval(SubspaceMat(n,Hamiltonian(n)))
    Eval3, Evec3 = EvecEval(SubspaceMat(p,Hamiltonian(p)))
    x = np.matmul(np.transpose(np.conjugate(Evec1)), np.matmul(SubspaceMat(m,Op(m)), Evec1))
    y = np.matmul(np.transpose(np.conjugate(Evec2)), np.matmul(SubspaceMat(n,Op(n)), Evec2))
    z = np.matmul(np.transpose(np.conjugate(Evec3)), np.matmul(SubspaceMat(p,Op(p)), Evec3))
    fig, (ax1, ax2, ax3)=  plt.subplots(3, sharex='all')
    for j in range(0, np.size(Eval1)):
        # print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y[j,j],3)))
        ax1.plot(Eval1[j], np.real(x[j, j]), marker='.', color='C2')
        ax1.legend(['8 Atoms'],loc=1)
        ax1.set_title('ETH Conjecture of {}'.format(r'$\hat{O}^{Z}$'))
        ax1.set(ylabel=r'$\langle n|\hat{O}^{Z}|n\rangle$')
    for i in range(0, np.size(Eval2)):
        # print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y[j,j],3)))
        ax2.plot(Eval2[i], np.real(y[i, i]), marker='.', color='C3')
        ax2.legend(['10 Atoms'],loc=1)
        ax2.set(ylabel=r'$\langle n|\hat{O}^{Z}|n\rangle$')
    for k in range(0, np.size(Eval3)):
        ax3.plot(Eval3[k], np.real(z[k, k]), marker='.', color='C4')
        ax3.legend(['12 Atoms'], loc=1)
        ax3.set(xlabel='E', ylabel=r'$\langle n|\hat{O}^{Z}|n\rangle$')
    #plt.savefig('fig1new')
    #plt.show()

# plotFig2A(8,10,12,PXPOBCNew,O_z)