import os
os.environ['OMP_NUM_THREADS'] = '1'
#from Coupling_To_Bath import *
from PXP_Entry_By_Entry import *
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from time import time
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations
import scipy.sparse as sp
from scipy.special import comb


###############################################################################
#                       definitions for TI model for Dim = 1                  #
###############################################################################
g_Sparse = csr_matrix([0, 1])  # ground state
# print("ground=\n", g)

r_Sparse = csr_matrix([1, 0])  # rydberg state
# print("rydberg=\n", r)

X_i_Sparse = csr_matrix([[0, 1], [1, 0]])  # flips spin up and spin down

Q_i_Sparse = csr_matrix([[1, 0], [0, 0]])  # projector on rydberg state (density of excited states) |r><r|
# print("Q_i = \n", Q_i)

P_i_Sparse = csr_matrix([[0, 0], [0, 1]])  # projector on ground state (density of ground states) |g><g|
# print("P_i = \n",P_i)

Z_i_Sparse = csr_matrix([[1, 0], [0, -1]])  # pauli Z

###################################################################################
###################################################################################

def O_z_PXP_Entry_Sparse(n, Subspace):
    '''
    Builds O_z in Subspace basis of PXP entry Sparsely
    :param n: number of atoms
    :return: O_z matrix sparse
    '''
    Base = Subspace(n)
    Avg_mag = np.empty((len(Base)))
    for i in range(0,len(Base)):
        posmag = np.sum(Base[i,:])
        Avg_mag[i] = np.divide(((-1)* (np.shape(Base)[1] - posmag) + posmag),np.shape(Base)[1])
    O_z_mat = sp.diags(Avg_mag)
    return O_z_mat

def Z_i_PXP_Entry_Sparse(n, i, Subspace):
    '''
    Builds Z_i in Subspace basis of PXP entry Sparsely
    :param n: number of atoms
    :param i: index of Z_i
    :return: Z_i matrix sparse
    '''
    Base = Subspace(n)
    Z_i_mag = np.empty((len(Base)))
    for j in range(0, len(Base)):
        if Base[j,i-1] == 1:
            Z_i_mag[j] = 1
        else:
            Z_i_mag[j] = -1
    Z_i_mat = sp.diags(Z_i_mag)
    return Z_i_mat

def Z_i_Spin_Basis_sparse(n,i):
    '''
    :param n: dimension 2**n
    :param i: atom index of Z operator
    :return: Z_i in dimension 2**n sparsely!
    '''
    if i <= n:
        Z_geni = sp.kron(sp.identity(2**(i-1)),sp.kron(Z_i_Sparse, sp.identity(2**(n-i))))
    else:
        Z_geni= sp.identity(2**n)
    return Z_geni

def X_i_Dict_PXP_Full_Basis_Sparse(n, i):
    '''
    Builds X_i in FULL basis of PXP entry Sparsely
    :param n: number of atoms
    :param i: index of X_i
    :return: X_i matrix sparse
    '''
    Base = Basis(n)
    Search = Base.copy()
    x = np.empty((len(Base), 2))
    x[:, 0] = np.arange(0, int(len(Base)), 1)
    for j in range(0, len(Base)):
        if Base[j, i] == 1:
            Search[j, i] = 0
        else:
            Search[j, i] = 1
        x[j, 1] = int(np.nonzero(np.all(Base == Search[j, :], axis=1))[0])  # gets basis row index of new vector after X acts on initial state vector
    return x


def X_i_Mat_PXP_Full_Basis_Sparse(n, i):
    '''
    X_i in PXP space - CAN'T BE WRITTEN IN SUBSPACE BECAUSE IT TAKES OUT OF SUBSPACE!!!
    Sparse method - IS it even sparse??
    :param n: No of atoms
    :param i: site wanted
    :return: X Matrix for specific i sparsley
    '''
    X_mat = sp.dok_matrix((2**n,2**n))
    dict = X_i_Dict_PXP_Full_Basis(n, i)
    for j in range(0,2**n):
        X_mat[int(dict[j,0]),int(dict[j,1])] = 1
    return X_mat

#TODO ALGORITHM IMPROVEMENT SUGGESTION FOR EVERYTHING IN DICTIONARIES = GO ONLY UP UNTIL HALF THE DICT WITH LOOP AND THEN TRANSPOSE AND CONNECT

def PXP_Ham_OBC_Sparse(n, Subspace):
    '''
    builds the Hamiltonian from pairs of PXP_connected_states
    :param n: number of atoms
    :return: Matrix (the Hamiltonian) N_subspace x N_subspace
    '''
    x = PXP_connected_states(n, Subspace)
    PXP = sp.dok_matrix((Subspace_basis_count_faster(n), Subspace_basis_count_faster(n)))
    for i in range(0, len(x)):
        PXP[x[i, 0], x[i, 1]] = 1   # should be 1 and then 0 in the cols of the x's, but doesn't really matter
    return PXP

def TIOBCNew_Sparse(n_TI, J, h_x, h_z): # faster method
    """
    Tilted Ising Hamiltonian OBC SPARSE
    :param n_TI: No of Tilted ising atoms MUST BE =>2
    :param J: Ising coupling parameter
    :param h_x: transverse field strength
    :param h_z: Z field (Longtitudinal) strength
    :return: Tilted Ising Hamiltonian (for i=>2) SPARSE
    """
    d = 2 ** n_TI # dimension
    Zi = Z_i_Sparse # Notation convenience
    Zi_plus1 = Z_i_Sparse # Notation convenience
    Xi = X_i_Sparse  # Notation convenience
    Ising_part = sp.csr_matrix((d, d))
    Transverse = Ising_part.copy()
    Longtitude = Ising_part.copy()
    for i in range(1, n_TI+1): # i = atom number
        Transverse = Transverse + sp.kron(sp.identity(2 ** (i-1)),sp.kron(Xi,sp.identity(2 ** (n_TI-i))))
        Longtitude = Longtitude + sp.kron(sp.identity(2 ** (i-1)),sp.kron(Zi,sp.identity(2 ** (n_TI-i))))
        if i != n_TI:
            Ising_part = Ising_part + sp.kron(sp.identity(2 ** (i-1)),sp.kron(Zi,sp.kron(Zi_plus1, sp.identity(2 ** (n_TI-1-i)))))
    TI_fin = ((h_x) * Transverse) + ((h_z) * Longtitude) + ((J) * Ising_part)
    return TI_fin

def TIOBCNewImpure_sparse(n, J, h_x, h_z, h_imp, m): # faster method
    """
    Tilted Ising Hamiltonian OBC with Z_i impurity on the m'th site SPARSE
    :param n: No of Tilted ising atoms MUST BE =>2
    :param J: Ising coupling parameter
    :param h_x: transverse field strength
    :param h_z: Z field (Longtitudinal) strength
    :param h_imp: impurity strength
    :param m: impurity site! (from 1 to n_TI)
    :return: Tilted Ising Hamiltonian WITH IMPURITY (for n=>2) SPARSE
    """
    TI = TIOBCNew_Sparse(n, J, h_x, h_z)
    if n==0:
        Z_impure = np.array(0)
    else:
        Z_impure = (h_imp) * Z_i_Spin_Basis_sparse(n, m)
    TI_impure = TI+ Z_impure
    return TI_impure


def Z_i_Coupling_PXP_Entry_to_TI_Sparse(n_PXP, n_TI, h_c):
    """
    SPARSE Z_i nature coupling matrix (two-site) of Subspace NEW pxp version and TI regular (dimension is Fib(n_PXP+3)*(2**n_TI) x Fib(n_PXP+3)*(2**n_TI))
    :param n_PXP: Number of PXP atoms (0 to whatever)
    :param n_TI: Number of TI atoms (0 to whatever)
    :param h_c: coupling strength parameter
    :return: matrix in (full) Fib(n_PXP+3)*(2**n_TI) x Fib(n_PXP+3)*(2**n_TI) dimension, sparse
    """
    n_tot = n_PXP + n_TI  # Total number of atoms
    d_TOT = Subspace_basis_count_faster(n_PXP) + 2 ** n_TI  # Total dimension
    Coupling = (h_c) * sp.kron(Z_i_PXP_Entry_Sparse(n_PXP, n_PXP, PXP_Subspace_Algo), Z_i_Spin_Basis_sparse(n_TI, 1))
    if n_TI == 0 or n_PXP == 0 or h_c == 0:
        Coupterm = sp.csr_matrix((d_TOT, d_TOT))
    else:
        Coupterm = Coupling
    return Coupterm


def PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, Subspace, m):
    '''
    Full coupled Hamiltonian with PXP Subspace Entry by entry code, sparse!!
    :param n_PXP:
    :param n_TI:
    :param J:
    :param h_x:
    :param h_z:
    :param h_c:
    :param h_imp:
    :param Subspace:
    :param m:
    :return: Matrix (full Hamiltonian), Sparse
    '''
    n_tot= n_PXP + n_TI
    d_PXP = 2 ** n_PXP
    d_TI = 2 ** n_TI
    PXP = PXP_Ham_OBC_Sparse(n_PXP, Subspace)
    TI = TIOBCNewImpure_sparse(n_TI, J, h_x, h_z, h_imp, m)
    HamNoCoupl = sp.kron(PXP, sp.identity(d_TI))+sp.kron(sp.identity(d_PXP), TI)
    TotalHam = HamNoCoupl + Z_i_Coupling_PXP_Entry_to_TI_Sparse(n_PXP,n_TI, h_c)
    return TotalHam


#TODO think how the coupling looks on PXP model
#TODO TI_Impure_sparse

