from Coupling_To_Bath import *
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from time import time
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations
from scipy.sparse import csr_matrix, lil_matrix, kron, identity
from scipy.special import comb
#from scipy.sparse.csr_matrix import multiply

def Basis(n):
    '''
    Builds binary basis of all possible states (not the kinetically constrained subspace yet)
    :param n: No of atoms in chain
    :return: matrix of row vectors of new basis
    '''
    vec= np.zeros(n)
    vec[0]=1
    permute = np.vstack((np.zeros(n), list(multiset_permutations(vec.astype('int')))))
    for i in range(0,n-1):
        vec[i+1]=1
        permute = np.vstack((permute,list(multiset_permutations(vec.astype('int')))))
    # print((np.fliplr(permute)))
    return permute

def Basis_No_check(n):
    '''
    calculates number of basis state (the zero state+ all magnetization states) - as a sum of newtons binom
    :param n: No of atoms in chain
    :return: scalar (number of states in basis) should be 2**n
    '''
    Sum = np.zeros(1)
    for i in range(0,n+1):
        Sum=Sum+comb(n,i)
    return Sum

def Subspace_basis_indeces(n):
    '''
    reduces binary basis state indeces!! of kinetically constrained subspace OBC
    :param n: No of atoms in chain
    :return: Array of indeces of rows of Subspace states
    '''
    basis = Basis(n)
    Index_array = np.empty(0)
    for i in range(0,np.shape(basis)[0]):
        for j in range(0,np.shape(basis)[1]-1):
            if basis[i,j]==1 and basis[i,j+1] == 1:
                Index_array= np.append(Index_array,i)
                break
    Tot_rows= np.arange(0,np.shape(basis)[0],1)
    Subspace_rows= np.delete(Tot_rows,(np.isin(Tot_rows,Index_array)))
    return Subspace_rows

def Subspace_basis(n):
    '''
    Returns subspace matrix (consisting of only constrained subspace states)
    :param n: No of atoms in chain
    :return: matrix of allowed states
    '''
    basis = Basis(n)
    Subspace_rows = Subspace_basis_indeces(n)
    return basis[Subspace_rows,:]

def Subspace_basis_count(n):
    '''
    counts number of Subspace vectors (Fibonacci of n+3 if we start from {0,1})
    :param n: No of atoms in chain
    :return: Scalar - No of subspace vectors
    '''
    start= np.array((0,1))
    for i in range(0,n+1):
        start=np.append(start,start[i]+start[i+1])
    return start[len(start)-1]

def Subspace_basis_count_faster(n):
    ''' Counts number of Subspace vectors in faster way, with the aid of Binet formula'''
    Fibonacci = np.arange(0,n+3)
    lengthFibo = len(Fibonacci)
    sqrtFive= np.sqrt(5)
    alpha = (1 + sqrtFive) / 2
    beta = (1 - sqrtFive) / 2
    F_n = np.rint(((alpha ** Fibonacci)- (beta ** Fibonacci)) / (sqrtFive))
    return int(F_n[len(F_n)-1])

# def PXP(n):
#     '''
#     PXP Hamiltonian construction in subspace basis - Open Boundary Conditions!
#     :param n: number of PXP atoms
#     :return: matrix
#     '''
#     Base = Subspace_basis(n)
#     Ham= np.zeros((np.shape(Base)[0], np.shape(Base)[0])) # Hamiltonian is of shape N_subspace x N_subspace
#     for i in range(0, np.shape(Base)[0]):
#         for j in range(0,n):
#             if Base[i, j]==0 and Base[i, j + 2]==0:
#                 if Base[i,j+1]==0:
#                     vecj= Base[i,:].copy()
#                 vecj[i]=0
#                 Ham[np.argwhere((Base == vecj).all(axis=1))[0],j]= 1
#             if Base[j, i]==0 and Base[j, i - 1]==0 and Base[j, i + 1]==0:
#                 vecj=Base[j].copy()
#                 vecj[i]=1
#             vecj
# #      Ham[masheu,masheu]=1
# #      if Base[j, i] and Base[j + 1, i]:
#     return

def PXP_connected_states(n):
    '''
    Gets connected states of every subspace basis vector
    :param n: number of PXP atoms
    :return: matrix of # x 2 of basis vectors and products of Hamiltonian multipication
    '''
    Base = Subspace_basis(n)
    x = np.empty((0,2))
    for i in range(0, len(Base)):
        # rows = np.count_nonzero(Base[i,:]==1) #Number of rows in a transformed matrix of state i is number of excitations in i
        # if rows != 0:
        for j in range(0,n):
                if Base[i, j] == 1: # finds all vectors that are -1 magnetization from the original one
                    vec1 = Base[i,:].copy()
                    vec1[j] = 0
                    index1 = np.squeeze((np.where(np.all(Base==vec1,axis=1)))) #index of the basis state that the state moves to under PXP Ham (flip DOWN)
                    x = np.vstack((x,np.array((i, index1)))) # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
                elif j == 0 and Base[i,j+1] == 0:
                    vec2 = Base[i,:].copy()
                    vec2[j] = 1
                    index2 = np.squeeze((np.where(np.all(Base==vec2,axis=1)))) #index of the basis state that the state moves to under PXP Ham (flip UP)
                    x = np.vstack((x,np.array((i, index2)))) # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
                elif j == n-1 and Base[i,j-1] == 0:
                    vec3 = Base[i,:].copy()
                    vec3[j] = 1
                    index3 = np.squeeze((np.where(np.all(Base==vec3,axis=1)))) #index of the basis state that the state moves to under PXP Ham (flip UP)
                    x = np.vstack((x,np.array((i, index3)))) # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
                elif Base[i,j-1] == 0 and Base[i,j+1] == 0:
                    vec4 = Base[i,:].copy()
                    vec4[j] = 1
                    index4 = np.squeeze((np.where(np.all(Base==vec4,axis=1)))) #index of the basis state that the state moves to under PXP Ham (flip UP)
                    x = np.vstack((x,np.array((i, index4)))) # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
    return x.astype('int')

def PXP_connected_states_option2(n):
    '''
    Gets connected states of every subspace basis vector
    :param n: number of PXP atoms
    :return: matrix of # x 2 of basis vectors and products of Hamiltonian multipication
    '''
    Base = Subspace_basis(n)
    # Ham= np.zeros((np.shape(Base)[0], np.shape(Base)[0])) # Hamiltonian is of shape N_subspace x N_subspac
    x = np.empty((0,2))
    for i in range(0, len(Base)):
        # rows = np.count_nonzero(Base[i,:]==1) #Number of rows in a transformed matrix of state i is number of excitations in i
        # if rows != 0:
        for j in range(1,n-1):
                if Base[i, j] == 1: # finds all vectors that are -1 magnetization from the original one
                    vec1 = Base[i,:].copy()
                    vec1[j] = 0
                    index1 = np.squeeze((np.where(np.all(Base==vec1,axis=1)))) #index of the basis state that the state moves to under PXP Ham (flip DOWN)
                    x = np.vstack((x,np.array((i, index1)))) # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
                elif Base[i,j-1] == 0 and Base[i,j+1] == 0:
                    vec2 = Base[i,:].copy()
                    vec2[j] = 1
                    index2 = np.squeeze((np.where(np.all(Base==vec2,axis=1)))) #index of the basis state that the state moves to under PXP Ham (flip UP)
                    x = np.vstack((x,np.array((i, index2)))) # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
        if Base[i,0] == 1:
            vec3 = Base[i, :].copy()
            vec3[0] = 0
            index3 = np.squeeze((np.where(np.all(Base == vec3,
                                                 axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
            x = np.vstack((x, np.array((i,
                                        index3))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
        elif Base[i,1] == 0:
            vec4 = Base[i, :].copy()
            vec4[0] = 1
            index4 = np.squeeze((np.where(np.all(Base == vec4,
                                                 axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
            x = np.vstack((x, np.array((i,
                                        index4))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
        if Base[i,n-1] == 1:
            vec5 = Base[i, :].copy()
            vec5[n-1] = 0
            index5 = np.squeeze((np.where(np.all(Base == vec5,
                                                 axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
            x = np.vstack((x, np.array((i,
                                        index5))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
        elif Base[i,n-2] == 0:
            vec6 = Base[i, :].copy()
            vec6[n-1] = 1
            index6 = np.squeeze((np.where(np.all(Base == vec6,
                                                 axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
            x = np.vstack((x, np.array((i,
                                        index6))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
    return x.astype('int')

def PXP_Ham_OBC_Entrybyentry(n):
    '''
    builds the Hamiltonian from pairs of PXP_connected_states
    :param n: number of atoms
    :return: Matrix (the Hamiltonian) N_subspace x N_subspace
    '''
    x = PXP_connected_states(n)
    PXP = np.zeros((Subspace_basis_count_faster(n),Subspace_basis_count_faster(n)))
    for i in range(0,len(x)):
        PXP[x[i,0],x[i,1]] = 1
    return PXP

def O_z_PXP_Entry_basis(n):
    '''
    Building O_z in basis of PXP entry
    :param n: number of atoms
    :return: O_z matrix
    '''
    Base = Subspace_basis(n)
    print(Base)
    Avg_mag = np.empty((len(Base)))
    for i in range(0,len(Base)):
        posmag = np.sum(Base[i,:])
        Avg_mag[i] = np.divide(((-1)* np.shape(Base)[1] + posmag),np.shape(Base)[1])
    O_z_mat = np.diag(Avg_mag)
    return O_z_mat

# def O_z_PXP_basis(n):
#     Avg_mag = PXP_basis_magnetization(n)
#     O_z_mat= np.zeros((len(Avg_mag),len(Avg_mag)))
#     for i in range(0,len(Avg_mag)):
#         O_z_mat[i,i]=



    #TODO start saving info in pickle and extracting back
    #TODO cluster - fixing so I can work from home Devendra