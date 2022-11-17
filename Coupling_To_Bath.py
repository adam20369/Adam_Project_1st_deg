import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import numpy.linalg as la
from time import time
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
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
from scipy.stats import bootstrap

# ========================== definitions Dim=1 ======================
g = np.array([0, 1])  # ground state
# print("ground=\n", g)

r = np.array([1, 0])  # rydberg state
# print("rydberg=\n", r)

X_i = np.array([[0, 1], [1, 0]])  # flips ground -> rydberg/ rydberg-> ground |g><r|+|r><g|
# print("X_i = \n", X_i)

Q_i = np.array([[1, 0], [0, 0]])  # projector on rydberg state (density of excited states) |r><r|
# print("Q_i = \n", Q_i)

P_i = np.array([[0, 0], [0, 1]])  # projector on ground state (density of ground states) |g><g|
# print("P_i = \n",P_i)

Z_i = np.array([[1, 0], [0, -1]])  # pauli Z


# ======================== declarations of some operators in D=2**n Hilbert (Kronecker) space ===========================

def Z_1(n):
    Z_1 = np.kron(Z_i, np.identity(2**(n-1)))
    return Z_1
# print(Z_1(3))

def Z_generali(n,i): # calculates Z for general i
    '''

    :param n: dimension 2**n
    :param i: atom index of Z operator
    :return: Z_i in dimension 2**n
    '''
    if i <= n:
        Z_geni = np.kron(np.identity(2**(i-1)),np.kron(Z_i, np.identity(2**(n-i))))
    else:
        Z_geni= np.identity(2**n)
    return Z_geni
# print(Z_1(3))

def O_znew(n): #mean of Z_i's
    '''
    Arithmetic Mean of Z_i's
    :param n: number of atoms (2**n is dimension)
    :return: O_z
    '''
    d = 2 ** n
    Z_sum= np.array(0)
    for i in range(1,n+1):
        Z_sum= Z_sum +Z_generali(n,i)
    O_zop= (1/n) * Z_sum
    return O_zop

def Neelstate(n): # GENERAL NEELSTATE
    d = 2 ** n
    Neel= np.array(1)
    Even= np.arange(0,n,2)
    for i in range(0,n):
        if i in Even:
            Neel = np.kron(Neel, r)
        else:
            Neel = np.kron(Neel, g)
    return Neel

def Haarstate(n): # GENERAL HAAR STATE
    '''
    General Haar state, A Haar state is a state that simulates an "average" eigenstate of the system,
    since it replaces an arithmetic mean over all eigenstates of an observable
    :param n: number of atoms
    :return: Haar Vector
    '''
    if n == 0:
         HaarVec= np.array(1)
    else:
        d = 2 ** n
        alpha= np.random.normal(0, 1, 2 ** n)
        betta= np.random.normal(0, 1, 2 ** n)
        v= alpha + 1j*betta
        HaarVec= np.divide(v, la.norm(v))
    return HaarVec.round(6)

def NeelHaar(n_PXP, n_TI):
    """
    Combination of the Neel and Haar states
    :param n_PXP:  Number of PXP chain atoms
    :param n_TI: number of TI chain atoms
    :return: Neel-Haar combined state
    """
    NeelHaarstate = np.kron(Neelstate(n_PXP), Haarstate(n_TI))
    return NeelHaarstate

# def Haarstate_seed(n,seed):
#     '''
#     General Haar state (seeded), A Haar state is a state that simulates an "average" thermal eigenstate of the system,
#     since it replaces an arithmetic mean over all eigenstates of an observable
#     :param n: number of atoms
#     :return: Haar Vector
#     '''
#     if n == 0:
#          HaarVec= np.array(1)
#     else:
#         d = 2 ** n
#         alpha_betta= np.random.RandomState(seed).normal(0, 1, 2 ** (n+1))
#         alpha_betta= alpha_betta.reshape((2,2**n)).astype('complex')
#         alpha_betta[1,:]= 1j*(alpha_betta[1,:].copy())
#         v = np.sum(alpha_betta,axis=0)
#         HaarVec= np.divide(v, la.norm(v))
#     return HaarVec.round(6)
#
#
# def NeelHaar_seed(n_PXP, n_TI,seed):
#     """
#     Combination of the Neel and SEEDED Haar states
#     :param n_PXP:  Number of PXP chain atoms
#     :param n_TI: number of TI chain atoms
#     :return: Neel-Haar combined state
#     """
#     NeelHaarseededstate = np.kron(Neelstate(n_PXP), Haarstate_seed(n_TI,seed))
#     return NeelHaarseededstate
#

# ===========================   Declarations of the separate Hamiltonians, coupling, and full coupled hamiltonian (basis of 2**ntot) ==========================================

def PXPOBCNew2(n): # faster method
    '''
    Faster way of PXP OBC
    :param n: number of atoms
    :return: PXP model hamiltonian OBC
    '''
    d = 2 ** n # full dimension
    Pi_minus1 = P_i  # notation convenience
    Xi = X_i  # notation convenience
    Pi_plus1 = P_i  # notation convenience
    PXPleftbound = np.kron(Xi, np.kron(Pi_plus1, np.identity(2 ** (n - 2))))
    PXPrightbound = np.kron(np.identity(2 ** (n - 2)), np.kron(Pi_minus1, Xi))
    PXPnobound = np.zeros([d, d])
    for i in range(2, n): #i marks the number of the atom that Xi sits on
        PXPnobound = PXPnobound + np.kron(np.identity(2**(i-2)),np.kron(Pi_minus1,np.kron(Xi,np.kron(Pi_plus1,np.identity(2**(n-1-i))))))
    pxp_fin = PXPleftbound + PXPnobound + PXPrightbound
    return pxp_fin

def PXPOBCNew2_Impure(n,j,st): # faster method
    '''
    Faster way of PXP OBC
    :param n: number of atoms
    :param j: impurity site number
    :param st: impurity site number

    :return: PXP model hamiltonian OBC
    '''
    d = 2 ** n # full dimension
    Pi_minus1 = P_i  # notation convenience
    Xi = X_i  # notation convenience
    Pi_plus1 = P_i  # notation convenience
    PXPleftbound = np.kron(Xi, np.kron(Pi_plus1, np.identity(2 ** (n - 2))))
    PXPrightbound = np.kron(np.identity(2 ** (n - 2)), np.kron(Pi_minus1, Xi))
    PXPnobound = np.zeros([d, d])
    for i in range(2, n): #i marks the number of the atom that Xi sits on
        PXPnobound = PXPnobound + np.kron(np.identity(2**(i-2)),np.kron(Pi_minus1,np.kron(Xi,np.kron(Pi_plus1,np.identity(2**(n-1-i))))))
    pxp_fin = PXPleftbound + PXPnobound + PXPrightbound + (st * np.kron(np.identity(2**(j-1)),np.kron(X_i,np.identity(2**(n-j)))))
    return pxp_fin

def PXPOBCNew2old(n): # faster method
    '''
    Faster way of PXP OBC
    :param n: number of atoms
    :return: PXP model hamiltonian OBC
    '''
    d = 2 ** n # full dimension
    Pi_minus1 = P_i  # notation convenience
    Xi = X_i  # notation convenience
    Pi_plus1 = P_i  # notation convenience
    PXPleftbound = np.zeros((d, d))
    PXPrightbound = np.zeros((d, d))
    PXPnobound = np.zeros((d, d))
    for i in range(1, n+1):  # i marks the number of atom from 1 to n
        if i == 1:
            PXPleftbound = np.kron(Xi,np.kron(Pi_plus1,np.identity(2**(n-2))))
        elif i == n:
            PXPrightbound = np.kron(np.identity(2**(n-2)),np.kron(Pi_minus1,Xi))
        else:
            PXPnobound = PXPnobound + np.kron(np.identity(2**(i-2)),np.kron(Pi_minus1,np.kron(Xi,np.kron(Pi_plus1,np.identity(2**(n-1-i))))))
    pxp_fin = PXPleftbound + PXPnobound + PXPrightbound
    return pxp_fin

def TIOBCNew2(n_TI, J, h_x, h_z): # faster method
    """
    Tilted Ising Hamiltonian OBC, faster method
    :param n_TI: No of Tilted ising atoms MUST BE =>2
    :param J: Ising coupling parameter
    :param h_x: transverse field strength
    :param h_z: Z field (Longtitudinal) strength
    :return: Tilted Ising Hamiltonian (for i=>2)
    """
    d = 2 ** n_TI # dimesion
    Zi = Z_i # Notation convenience
    Zi_plus1 = Z_i # Notation convenience
    Xi = X_i  # Notation convenience
    Ising_part = np.zeros((d, d))
    Transverse = np.zeros((d, d))
    Longtitude = np.zeros((d, d))
    for i in range(1, n_TI+1): # i = atom number
        Transverse= Transverse + np.kron(np.identity(2 ** (i-1)),np.kron(Xi,np.identity(2 ** (n_TI-i))))
        Longtitude= Longtitude + np.kron(np.identity(2 ** (i-1)),np.kron(Zi,np.identity(2 ** (n_TI-i))))
        if i != n_TI:
            Ising_part= Ising_part + np.kron(np.identity(2 ** (i-1)),np.kron(Zi,np.kron(Zi_plus1, np.identity(2 ** (n_TI-1-i)))))
    TI_fin = ((h_x) * Transverse) + ((h_z) * Longtitude) + ((J) * Ising_part)
    return TI_fin

def TIOBCNewImpure2(n, J, h_x, h_z, h_imp, m): # faster method
    """
    Tilted Ising Hamiltonian OBC with impurity on the m'th site
    :param n: No of Tilted ising atoms MUST BE =>2
    :param J: Ising coupling parameter
    :param h_x: transverse field strength
    :param h_z: Z field (Longtitudinal) strength
    :param h_imp: impurity strength
    :param m: impurity site! (from 1 to n_TI)
    :return: Tilted Ising Hamiltonian WITH IMPURITY (for i=>2)
    """
    d = 2 ** n #dimension
    TI = TIOBCNew2(n, J, h_x, h_z)
    if n==0:
        Z_impure = np.array(0)
    else:
        Z_impure = (h_imp) * Z_generali(n, m)
    TI_impure = np.add(TI, Z_impure)
    return TI_impure


def Coupling2(n_PXP, n_TI , Coupmat, h_c): #faster way
    """
    faster way of
    2 site coupling matrix in TOTAL Hamiltonian dimension (2**(n_PXP + n_TI))
    :param n_PXP: Number of PXP atoms (0 to whatever)
    :param n_TI: Number of TI atoms (0 to whatever)
    :param Coupmat: Coupling 2x2 base matrix (one-site matrix)
    :param h_c: coupling strength parameter
    :return:
    """
    n_tot = np.add(n_PXP, n_TI)  # Total number of atoms
    d_pxp = 2 ** n_PXP #dimension of PXP
    d_TI = 2 ** n_TI #dimension of TI
    d_TOT = 2 ** n_tot  #Total dimension
    Coupling = (h_c) * np.kron(Coupmat, Coupmat)
    if n_TI == 0 or n_PXP == 0 or h_c == 0:
        Coupterm = np.zeros((d_TOT,d_TOT))
    else:
        Coupterm = np.kron(np.kron(np.identity(2 ** (n_PXP - 1)), Coupling), np.identity(2 ** (n_TI-1)))
    return Coupterm

def Coupling2Subspc(n_PXP, n_TI , Coupmat, h_c):
    '''
    2 site Coupling matrix in subspace basis of PXP
    :param n_PXP: number of PXP atoms
    :param n_TI: number of TI atoms
    :param Coupmat: 2x2 base one-site matrix (Nature of coupling)
    :param h_c: Coupling strength
    :return: coupling subspace matrix of size (fib(n_pxp-3)*n_TI x fib(n_pxp-3)*n_TI)
    '''
    n_tot = np.add(n_PXP, n_TI)  # Total number of atoms
    d_pxp = 2 ** n_PXP #dimension of PXP
    d_TI = 2 ** n_TI #dimension of TI
    d_TOT = 2 ** n_tot  #Total dimension
    Coupling = (h_c) * np.kron(Coupmat, Coupmat)
    if n_TI == 0 or n_PXP == 0 or h_c == 0:
        Coupterm = np.zeros((d_TOT,d_TOT))
    else:
        Coupterm = np.kron(np.kron(np.identity(2 ** (n_PXP - 1)), Coupling), np.identity(2 ** (n_TI-1)))
    return Coupterm


def PXPBathHam2(n_PXP, n_TI, Coupmat, J, h_x, h_z, h_c, h_imp, m=1):
    """
    FULL 2**(n_tot) dimension COUPLED PXP and TI hamiltonian Builder
    :param n_PXP: PXP Atom number
    :param n_TI: TI Atom number
    :param Coupmat: 2x2 base matrix of coupling
    :param h_x: transverse field strength
    :param h_z: Z field strength
    :param h_c: coupling strength
    :param h_imp: impurity strength
    :param m: impurity site (default is 1)
    :return: Full coupled Hamiltonian
    """
    n_tot= np.add(n_PXP,n_TI)
    d_PXP = 2 ** n_PXP
    d_TI = 2 ** n_TI
    d_tot= 2 ** n_tot
    PXP = PXPOBCNew2(n_PXP)
    TI = TIOBCNewImpure2(n_TI, J, h_x, h_z, h_imp, m)
    HamNoCoupl = np.add(np.kron(PXP, np.identity(d_TI)), np.kron(np.identity(d_PXP), TI))
    TotalHam = np.add(HamNoCoupl, Coupling2(n_PXP, n_TI, Coupmat, h_c))
    return TotalHam

def PXPBathHamNoCoupl2(n_PXP, n_TI, J, h_x, h_z, h_imp, m=1):
    """
    Just for checks - FULL 2**(n_tot) dimension UNCOUPLED PXP and TI hamiltonian Builder
    :param n_PXP: PXP Atom number
    :param n_TI: TI Atom number
    :param h_x: transverse field strength
    :param h_z: Z field strength
    :param h_imp: impurity strength
    :param m: impurity site (default is 1)
    :return: Full coupled Hamiltonian
    """
    n_tot= np.add(n_PXP,n_TI)
    d_PXP = 2 ** n_PXP
    d_TI = 2 ** n_TI
    d_tot= 2 ** n_tot
    PXP = PXPOBCNew2(n_PXP)
    TI = TIOBCNewImpure2(n_TI, J, h_x, h_z, h_imp, m)
    HamNoCoupl = np.add(np.kron(PXP, np.identity(d_TI)), np.kron(np.identity(d_PXP), TI))
    return HamNoCoupl

########## DIAGONALIZATION AND EIGENSTATE SPAN / RECOMBINE OF A GENERAL VECTOR STATE #############

def Diagonalize(Mat):
    '''
    calculates eigenvalues and eigenstates of a HERMITIAN matrix
    :param Mat: Any Hermitian matrix
    :return: eigenvalues and eigenstates
    '''
    eval, evec = la.eigh(Mat)
    return np.real(np.round(eval, 5)), np.round(evec, 5)
#Diagonalize(PXPBathHam2(7,3,Coupmat=Z_i, J=1,h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi),h_c=0,h_imp=0.01,m=1))
#Diagonalize(PXPOBCNew2(7))

def EigenSpan(Mat,
              VecState):
    """
    Spans some vector (VecState) in the eigenbasis of a matrix.
    :param Mat: Input matrix for eigenstate basis decomposition
    :param VecState: Some vector state (initial state usually) that we want to span in eigenstate basis
    :return: vector of weights (in Eigenstate basis)
    """
    Eval, Evec = Diagonalize(Mat)
    W = np.dot(np.transpose(Evec),VecState)
    return W

def EigenCombine(Mat,VecState):
    '''
    checks tat multiplying the weights back with the eigenstates gives the original vector
    :param Mat: Any matrix that we will use the eigenbasis of
    :param VecState: vector we want to span in the eigenbasis of the matrix
    :return: recombined vector
    '''
    W = EigenSpan(Mat,VecState)
    Eval, Evec = Diagonalize(Mat) #Evec is matrix!
    Recombine= np.round(np.dot(Evec,W),4)
    return Recombine

################################ CONNECTED SUBSPACE REDUCTION OF PXP OBC ####################################

def GroundstateCheck(n_PXP, n_TI, h_c=0 ,Coupmat=Z_i, J=1,h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi) ,h_imp=0.01):
    '''
    Checks ground state and anti ground state of PXP-TI Vs PXP and TI seperately
    :param n_PXP: number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param Coupmat: Coupling nature
    :param J: Ising strength
    :param h_x: X direction strength
    :param h_z: Z direction strength
    :param h_imp: TI Impurity strength
    :return: boolean for ground state and boolean for Anti ground state
    '''
    d_pxp=2**(n_PXP)
    d_TI=2**(n_TI)
    d_tot=2**(n_TI+n_PXP)

    EvalPXP, EvecPXP = Diagonalize(PXPOBCNew2(n_PXP))
    EvalTI, EvecTI = Diagonalize(TIOBCNewImpure2(n_TI,J, h_x, h_z, h_imp, m=2))
    EvalPXP_TI, EvecPXP_TI = Diagonalize(PXPBathHam2(n_PXP, n_TI, Coupmat, J, h_x, h_z, h_c, h_imp, m=2))
    print('ground state=', np.allclose(EvalPXP[0]+EvalTI[0],EvalPXP_TI[0]))
    print('anti ground state=', np.allclose(EvalPXP[d_pxp-1]+EvalTI[d_TI-1],EvalPXP_TI[d_tot-1]))
    return


def SubspcConnect(n_PXP,j,st):
    '''
    Maps the connected basis vectors of PXP OBC with impurity
    :param n_PXP: No. of PXP OBC atoms
    :param j: impurity site
    :param st: impurity strength
    :return: No. of components connected (No. of different blocks existing)
                & labeling of the vectors to corresponding blocks (which vector belongs to which block, numbering the blocks from 0..)
    '''
    sparse= csr_matrix(PXPOBCNew2_Impure(n_PXP,j,st))
    n_components, labels =connected_components(sparse)
    return n_components, labels

def Block_find(n_PXP,j,st):
    '''
    Finds largest connected block, and outputs indeces of vectors that belong to that block
    :param n_PXP: No. of PXP OBC atoms
    :param j: impurity site
    :param st: impurity strength
    :return: indeces (row/col numbers) of vectors of largest block
    '''
    n_components, labels= SubspcConnect(n_PXP, j, st)
    reoccur = np.bincount(labels).argmax() #block number that reoccurs the most
    vec_indeces = np.argwhere(labels==reoccur) #indeces of vectors of block
    return vec_indeces

def Block_dim(n_PXP,j,st):
    '''
    finds dimension  of largest connected block
    :param n_PXP: No. of PXP OBC atoms
    :param j: impurity site
    :param st: impurity strength
    :return: dimension of largest connected block
    '''
    n_components, labels= SubspcConnect(n_PXP, j, st)
    dim = np.bincount(labels)[np.bincount(labels).argmax()]
    return dim

def Subspc_Proj(n_PXP,j,st):
    '''
    Creates a projector of the relevant subspace
    :param n_PXP: No. of PXP OBC atoms (general)
    :param j: impurity site
    :param st: impurity strength
    :return: Subspace projector of PXP, containing only largest connected block
    '''
    vec_indeces= Block_find(n_PXP,j,st)
    block_array= np.zeros(2**(n_PXP))
    block_array[vec_indeces]=1
    proj = np.diag(block_array)
    return proj

def Subspace_PXP(n_PXP,j,st):
    '''
    Creates a subspace Impurity PXP Hamiltonian OF THE SAME DIMENSION, of the largest connected block
    :param n_PXP: No. of PXP OBC atoms (general)
    :param j: impurity site
    :param st: impurity strength
    :return: Subspace matrix of PXP, containing only largest connected block
    '''
    proj = Subspc_Proj(n_PXP,j,st)
    PXP_block_impure = np.matmul(proj,np.matmul(PXPOBCNew2_Impure(n_PXP,j,st),proj))
    return PXP_block_impure

def Subspace_reduced_PXP(n_PXP,j,st):
    '''
    reducing the PXP matrix by removing rows/cols with only zeros
    :param n_PXP: No. of PXP OBC atoms (general)
    :param j: impurity site
    :param st: impurity strength
    :return: reduced PXP matrix, without rows/cols with only zeros

    '''
    full_dim_proj_PXP= Subspace_PXP(n_PXP,j,st)
    red_dim_proj_PXP=full_dim_proj_PXP[:,~np.all(full_dim_proj_PXP==0,axis=1)]
    red_dim_proj_PXP=red_dim_proj_PXP[~np.all(full_dim_proj_PXP==0,axis=1),:]
    return red_dim_proj_PXP


def PXP_Subspace_Mat(Mat,n_PXP,j,st):
    '''
    projects a 2**N x 2**N matrix onto the kinetic constrained block of the PXP model
    :param n_PXP: No. of PXP OBC atoms (general)
    :param j: impurity site
    :param st: impurity strength
    :return: Subspace matrix of PXP, containing only largest connected block
    '''
    proj = Subspc_Proj(n_PXP,j,st)
    Projected_Mat = np.matmul(proj,np.matmul(Mat,proj))
    return Projected_Mat

def PXP_Subspace_mat_reduced(Mat,n_PXP,j=2,st=0):
    '''
    reduces the projected matrix to Fib(n_pxp+3) x Fib(n_pxp +3) dimension
    :param n_PXP: No. of PXP OBC atoms (general)
    :param j: impurity site
    :param st: impurity strength
    :return: reduced PXP matrix, without rows/cols with only zeros

    '''
    full_dim_Projected_Mat= PXP_Subspace_Mat(Mat,n_PXP,j,st)
    reduced_dim_proj_Mat=full_dim_Projected_Mat[:,~np.all(full_dim_Projected_Mat==0,axis=1)]
    reduced_dim_proj_Mat=reduced_dim_proj_Mat[~np.all(reduced_dim_proj_Mat==0,axis=1),:]
    return reduced_dim_proj_Mat

def Subspace_Coupling(n_PXP,n_TI, h_c,Mat= Z_generali ,j=2,st=0):
    '''
    Coupling in PXP subspace basis
    :param n_PXP: number of PXP atoms
    :param n_TI: number of TI atoms
    :param Mat: Coupling nature (2x2 matrix one site)
    :param h_c: coupling strength (controlled)
    :param j: impurity site
    :param st: impurity strength
    :return: coupling matrix in subspace
    '''
    Z_PXP_Subspace_Last = PXP_Subspace_mat_reduced(Mat(n_PXP,n_PXP),n_PXP,j,st)
    Z_TI_First = Mat(n_TI,1)
    Subspace_coup = h_c * np.kron(Z_PXP_Subspace_Last, Z_TI_First)
    return Subspace_coup

def EigenvalueUniqueness(n,j,st):
    '''
    Checks repetitivity of eigenvalues in PXP model, i.e, the degeneracy.
    :param n: no of atoms
    :param j: impurity site
    :param st: impurity strength
    :return: array of the number of repititions of each eigenstate
    '''
    eval, evec = Diagonalize(Subspace_reduced_PXP(n,j,st))
    Unique= np.unique(eval,False,False,True)
    return Unique

###################################################################################
#                                 EE CALCULATION                                  #
###################################################################################
#       NOT NEEDED DUE TO BUILDING NEW HAMILTONIAN ENTRY BY ENTRY                 #


def Binary_State_Mapping(state): #TODO check for some more cases
    '''
    maps some basis state of PXP model to binary basis (N X 2**N in size),
            where the first state [0,0...,1] (2**N) maps to [0,0,..,0] (N)
    :param state: a state in PXP model basis
    :return: vector in binary basis (N X 2**N in size)
    '''
    dim_size = np.size(state)
    atom_chain_size = np.log2(dim_size)
    index = int(dim_size - (np.flatnonzero(state)+1)) # Index is the number # to be converted to binary
    binary_representation = np.binary_repr(index) # number
    binary_array= list(map(int, str(binary_representation))) #array
    if atom_chain_size-np.size(binary_array) !=0 : # appending zeros in front of array to match size N
        zeros_missing= int(atom_chain_size-np.size(binary_array))
        for i in range(0,zeros_missing):
            binary_array=np.insert(binary_array,i,0)
    return binary_array

def Binary_Basis_Mapping(n_PXP,j=2,st=0):
    '''
    maps PXP subspace basis states to binary basis states matrix (Subspace_dim X N in size), Binary states as rows.
    :param n_PXP: no of PXP atoms
    :param j: impurity site
    :param st: impurity strength
    :return: matrix of binary basis states (of the reduced Subspace!) as Row vectors
    '''
    Subspc_Indeces = np.squeeze(Block_find(n_PXP,j,st))
    binary_basis= np.empty([0,n_PXP])
    index_array = 2**n_PXP - (Subspc_Indeces+1) # Indeces inverted to fit counting from end to start of vectors
    for k in index_array:
        binary_representation = np.binary_repr(k) # number
        binary_array= list(map(int, str(binary_representation))) #array
        if n_PXP-np.size(binary_array) != 0 : # appending zeros in front of array to match size N
            zeros_missing= int(n_PXP-np.size(binary_array))
            for i in range(0,zeros_missing):
                 binary_array=np.insert(binary_array,i,0)
        binary_basis=np.append(binary_basis,[binary_array], axis=0)
    return binary_basis

def Reverse_Mapping(binary_array):
#
    return

def Bipartition_Basis(n_PXP,j=2,st=0):
    '''
    Bipartition basis state subspace (does not take into account combinations that aren't allowed) - EVEN PXP number!!
    :param n_PXP: number of total atoms we want to bipartite (in middle)
    :param j: impurity site
    :param st: impurity strength
    :return: Basis of subsystem (when system is cut in half)
    '''
    sub_system_size = int(n_PXP / 2)
    sub_system_basis = Binary_Basis_Mapping(sub_system_size, j, st)
    return sub_system_basis

def Decompose_Eigenstates(n_PXP,j=2,st=0):
    '''
    Decompose each eigenstate to it's Schmitt's form
    :param n_PXP:
    :param j:
    :param st:
    :return:
    '''
    eval, evec = Diagonalize(Subspace_PXP(n_PXP,j=2,st=0))
    return eval, evec
