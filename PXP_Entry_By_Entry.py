import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from Coupling_To_Bath import *
import numpy.linalg as la
from time import time
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations
import scipy.sparse as sp
from scipy.special import comb
#from scipy.sparse.csr_matrix import multiply

####################################################################################################################################
#                                                     definitions dim = 1                                                          #
####################################################################################################################################
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


####################################################################################################################################
#                                               OLD SUBSPACE BASIS FINDING METHOD                                                  #
####################################################################################################################################

def Basis(n):
    '''
    Builds binary basis of all possible states (not the kinetically constrained subspace yet)
    :param n: No of atoms in chain
    :return: matrix of row vectors of new basis
    '''
    permute = np.zeros(n)
    vec = np.zeros(n)
    for i in range(0, n):
        vec[i] = 1
        permute = np.vstack((permute, list(multiset_permutations(vec.astype('int')))))
    # print((np.fliplr(permute)))
    return permute

def Half_Basis(n):
    '''
    Creates half basis in smarter way (only half filling permutations) FASTER!!!!!
    :param n: No of atoms in chain
    :return: Basis
    '''
    permute= np.zeros(n)
    vec = np.zeros(n)
    for i in range(0,int(n/2)+1):
        vec[i]=1
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
    basis = Half_Basis(n)
    Index_array = np.empty(0)
    for i in range(0,np.shape(basis)[0]):
        for j in range(0,np.shape(basis)[1]-1):
            if basis[i,j]==1 and basis[i,j+1] == 1:
                Index_array= np.append(Index_array,i)
                break
    Tot_rows= np.arange(0,np.shape(basis)[0],1)
    Subspace_rows= np.delete(Tot_rows,(np.isin(Tot_rows,Index_array)))
    return Subspace_rows

# def Subspace_basis_DIRECT(n):
#     '''
#     Direct Method of getting subspace vectors
#     :param n: No of atoms in chain
#     :return: SUBSPACE Basis
#     '''
#     permute= np.zeros(n)
#     vec = np.zeros(n)
#     for i in range(0,int(n/2)+1):
#         vec[i]=1
#         permute = np.vstack((permute,list(multiset_permutations(vec.astype('int')))))
#         for j in range(0,len(list(multiset_permutations(vec.astype('int'))))):
#             if multiset #???????????????????????????????
#     # print((np.fliplr(permute)))
#     return permute

def Subspace_basis(n): #OLD VERSION!!!
    '''
    Returns subspace matrix (consisting of only constrained subspace states) OLD VERSION!!!!
    :param n: No of atoms in chain
    :return: matrix of allowed states
    '''
    basis = Half_Basis(n)
    Subspace_rows = Subspace_basis_indeces(n)
    return basis[Subspace_rows,:]

################################################################################################################################################
#                                                                       End                                                                    #
################################################################################################################################################

def Subspace_basis_count(n):
    '''
    counts number of Subspace vectors (Fibonacci of n+3 if we start from {0,1})
    :param n: No of atoms in chain
    :return: Scalar - No of subspace vectors = fib(n+3)
    '''
    start= np.array((0,1))
    for i in range(0,n+1):
        start=np.append(start,start[i]+start[i+1])
    return start[len(start)-1]

def Subspace_basis_count_faster(n):
    ''' Counts number of Subspace vectors in faster way, with the aid of Binet formula
    :param n: No of atoms in chain
    :return: Scalar - No of subspace states = fib(n+3)

    '''
    Fibonacci = np.arange(0,n+3)
    lengthFibo = len(Fibonacci)
    sqrtFive= np.sqrt(5)
    alpha = (1 + sqrtFive) / 2
    beta = (1 - sqrtFive) / 2
    F_n = np.rint(((alpha ** Fibonacci)- (beta ** Fibonacci)) / (sqrtFive))
    return int(F_n[len(F_n)-1])

def Extended_X_i_Subspace_basis_count_faster(n):
    '''
    Counts the number of states in the Exntended X_i PXP subspace
    :param n: number of atoms in PXP chain
    :return: Scaler - number of subspace states = fib(n+3)+fib(n)
    '''
    return Subspace_basis_count_faster(n)+Subspace_basis_count_faster(n-3)

def TI_Allowed_No(n_pxp):
    '''
    How many TI atoms are allowed when PXP atoms are given, for 10^6 x 10^6 matrix
    :param n_pxp: No. of PXP atoms
    :return: Scalar, No. of TI atoms
    '''
    N = np.log(np.divide(10**6,Subspace_basis_count_faster(n_pxp)))*np.divide(1,np.log(2))
    return print('No. of TI atoms allowed: {}'.format(N))


def PXP_Subspace_Algo(n):
    '''
    New algorithm of producing PXP subspace from the start
    :param n: No of atoms
    :return: PXP constrained subspace matrix (Fib(n+3) x n )
    '''
    initial = np.zeros(n)
    vec1 = initial.copy()
    vec1[0] = 1
    single_excited_states = np.identity(n) # base vectors of single excitations, col index of 1 is the row index of course
    Subspace = np.vstack((initial, single_excited_states.copy()))
    init_indeces = np.arange(0,n,1)
    for i in range(1,Subspace_basis_count_faster(n)):
        Og_One_indeces= np.where(Subspace[i,:] == 1) # returns array of col. indeces of 1's of a specific row of subspace mat
        # print(Og_One_indeces)
        Plus_one = np.array(Og_One_indeces)+1
        Minus_one = np.array(Og_One_indeces)-1
        One_indeces = np.unique(np.concatenate((np.concatenate((Og_One_indeces, Plus_one),axis=0),Minus_one),axis=0)) # unique array of indeces where one's are not allowed
        # print(One_indeces)
        OG_Last = Og_One_indeces[0][np.shape(Og_One_indeces)[1]-1] #Last 1 in subspace vector No. i
        # print(OG_Last)
        Boolean = np.isin(init_indeces, One_indeces, invert = True) # searches for One_indeces in init_indeces and returns flipped boolean
            # print(Boolean)
        Boolean[np.where(init_indeces < OG_Last)] = False #defining all entries before some entry as false
            # print(Boolean)
        Vec = init_indeces.copy()[Boolean] # indeces of all initial vectors that are allowed to add
            # print(Vec)
        Subspace = np.vstack((Subspace, single_excited_states[Vec,:]+Subspace[i,:]))
    return Subspace

def PXP_Subspace_Algo_extended_X_i(n):
    '''
    Algorithm for producing PXP subspace INCLUDING ...11 states from the start
    :param n: No of atoms
    :return: PXP constrained subspace matrix +....11 states [(Fib(n+3)+Fib(n)) x n]
    '''
    initial = np.zeros(n)
    vec1 = initial.copy()
    vec1[0] = 1
    single_excited_states = np.identity(n) # base vectors of single excitations, col index of 1 is the row index of course
    Subspace = np.vstack((initial, single_excited_states.copy()))
    init_indeces = np.arange(0,n,1)
    for i in range(1,Subspace_basis_count_faster(n)):
        Og_One_indeces= np.where(Subspace[i,:] == 1) # returns array of col. indeces of 1's of a specific row of subspace mat
        # print(Og_One_indeces)
        Plus_one = np.array(Og_One_indeces)+1
        Minus_one = np.array(Og_One_indeces)-1
        One_indeces = np.unique(np.concatenate((np.concatenate((Og_One_indeces, Plus_one),axis=0),Minus_one),axis=0)) # unique array of indeces where one's are not allowed
        # print(One_indeces)
        OG_Last = Og_One_indeces[0][np.shape(Og_One_indeces)[1]-1] #Last 1 in subspace vector No. i
        # print(OG_Last)
        Boolean = np.isin(init_indeces, One_indeces, invert = True) # searches for One_indeces in init_indeces and returns flipped boolean
            # print(Boolean)
        Boolean[np.where(init_indeces < OG_Last)] = False #defining all entries before some entry as false
            # print(Boolean)
        Vec = init_indeces.copy()[Boolean] # indeces of all initial vectors that are allowed to add
            # print(Vec)
        Subspace = np.vstack((Subspace, single_excited_states[Vec,:]+Subspace[i,:]))
    #######Extension start##########
    #print(np.size(Subspace,axis=0))
    count=0
    for i in range(1,Subspace_basis_count_faster(n)):
        if Subspace[i,n-2] == 1:
            count=count+1
            vec = Subspace[i,:].copy()
            vec[n-1]=1
            Subspace = np.vstack((Subspace, vec))
    matsize = Extended_X_i_Subspace_basis_count_faster(n)
    #print(np.isclose(np.size(Subspace,axis=0),matsize)) #check
    return Subspace

def PXP_connected_states(n,Subspace):
    '''
    Gets connected states of every subspace basis vector for regular PXP subspace
    :param n: number of PXP atoms
    :return: matrix of # x 2 of basis vectors and products of Hamiltonian multiplication
    - Generally a map that tells you to which vectors every starting vector (all numbered by their index in the basis) maps under the Hamiltonian transformation
    (first col is initial vector and second col is vector after transformation)
    '''
    Base = Subspace(n)
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


# def PXP_connected_states_option2(n,basis_generating_func):
#     '''
#     Gets connected states of every subspace basis vector
#     :param n: number of PXP atoms
#     :return: matrix of # x 2 of basis vectors and products of Hamiltonian multipication
#     '''
#     Base = Subspace_basis(n,basis_generating_func)
#     # Ham= np.zeros((np.shape(Base)[0], np.shape(Base)[0])) # Hamiltonian is of shape N_subspace x N_subspac
#     x = np.empty((0,2))
#     for i in range(0, len(Base)):
#         # rows = np.count_nonzero(Base[i,:]==1) #Number of rows in a transformed matrix of state i is number of excitations in i
#         # if rows != 0:
#         for j in range(1,n-1):
#                 if Base[i, j] == 1: # finds all vectors that are -1 magnetization from the original one
#                     vec1 = Base[i,:].copy()
#                     vec1[j] = 0
#                     index1 = np.squeeze((np.where(np.all(Base==vec1,axis=1)))) #index of the basis state that the state moves to under PXP Ham (flip DOWN)
#                     x = np.vstack((x,np.array((i, index1)))) # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
#                 elif Base[i,j-1] == 0 and Base[i,j+1] == 0:
#                     vec2 = Base[i,:].copy()
#                     vec2[j] = 1
#                     index2 = np.squeeze((np.where(np.all(Base==vec2,axis=1)))) #index of the basis state that the state moves to under PXP Ham (flip UP)
#                     x = np.vstack((x,np.array((i, index2)))) # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
#         if Base[i,0] == 1:
#             vec3 = Base[i, :].copy()
#             vec3[0] = 0
#             index3 = np.squeeze((np.where(np.all(Base == vec3,
#                                                  axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
#             x = np.vstack((x, np.array((i,
#                                         index3))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
#         elif Base[i,1] == 0:
#             vec4 = Base[i, :].copy()
#             vec4[0] = 1
#             index4 = np.squeeze((np.where(np.all(Base == vec4,
#                                                  axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
#             x = np.vstack((x, np.array((i,
#                                         index4))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
#         if Base[i,n-1] == 1:
#             vec5 = Base[i, :].copy()
#             vec5[n-1] = 0
#             index5 = np.squeeze((np.where(np.all(Base == vec5,
#                                                  axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
#             x = np.vstack((x, np.array((i,
#                                         index5))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
#         elif Base[i,n-2] == 0:
#             vec6 = Base[i, :].copy()
#             vec6[n-1] = 1
#             index6 = np.squeeze((np.where(np.all(Base == vec6,
#                                                  axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
#             x = np.vstack((x, np.array((i,
#                                         index6))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
#     return x.astype('int')

def PXP_connected_states_extended_X_i(n,Subspace):
    '''
    Gets connected states of every subspace basis vector FOR X_i EXTENDED!!!
    :param n: number of PXP atoms
    :return: matrix of # x 2 of basis vectors and products of Hamiltonian multiplication
    - Generally a map that tells you to which vectors every starting vector (all numbered by their index in the basis) maps under the Hamiltonian transformation
    (first col is initial vector and second col is vector after transformation)
    '''
    Base = Subspace(n)
    x = np.empty((0,2))
    for i in range(0, len(Base)):
        # rows = np.count_nonzero(Base[i,:]==1) #Number of rows in a transformed matrix of state i is number of excitations in i
        # if rows != 0:
        if Base[i, n-2] == 1 and Base[i,n-1] ==1:
            for j in range(0, n-3):
                if Base[i, j] == 1:  # finds all vectors that are -1 magnetization from the original one
                    vec1 = Base[i, :].copy()
                    vec1[j] = 0
                    index1 = np.squeeze((np.where(np.all(Base == vec1,
                                                         axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip DOWN)
                    x = np.vstack((x, np.array((i,
                                                index1))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
                elif j == 0 and Base[i, j + 1] == 0:
                    vec2 = Base[i, :].copy()
                    vec2[j] = 1
                    index2 = np.squeeze((np.where(np.all(Base == vec2,
                                                         axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
                    x = np.vstack((x, np.array((i,
                                                index2))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
                elif Base[i, j - 1] == 0 and Base[i, j + 1] == 0:
                    vec4 = Base[i, :].copy()
                    vec4[j] = 1
                    index4 = np.squeeze((np.where(np.all(Base == vec4,
                                                         axis=1))))  # index of the basis state that the state moves to under PXP Ham (flip UP)
                    x = np.vstack((x, np.array((i,
                                                index4))))  # matrix of 2-vectors that specify index of original basis vector and indeces of the vectors it transforms too under the Hamiltonian
        else:
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

def PXP_Ham_OBC_Entrybyentry(n,Subspace):
    '''
    builds the Hamiltonian from pairs of PXP_connected_states
    :param n: number of atoms
    :return: Matrix (the Hamiltonian) N_subspace x N_subspace
    '''
    x = PXP_connected_states(n,Subspace)
    Base = Subspace(n)
    PXP = np.zeros((len(Base),len(Base)))
    for i in range(0,len(x)):
        PXP[x[i,0],x[i,1]] = 1
    return PXP

def PXP_Ham_OBC_Entrybyentry_extended_X_i(n,Subspace):
    '''
    builds the Hamiltonian from pairs of PXP_connected_states EXTENDED BASIS X_i
    :param n: number of atoms
    :return: Matrix (the Hamiltonian) N_subspace x N_subspace
    '''
    x = PXP_connected_states_extended_X_i(n,Subspace)
    Base = Subspace(n)
    PXP = np.zeros((len(Base),len(Base)))
    for i in range(0,len(x)):
        PXP[x[i,0],x[i,1]] = 1
    return PXP


def O_z_PXP_Entry_basis(n,Subspace):
    '''
    Building O_z in basis of PXP entry ALSO WORKS FOR X_i EXTENDED!!!!
    :param n: number of atoms
    :return: O_z matrix
    '''
    Base = Subspace(n)
    Avg_mag = np.empty((len(Base)))
    for i in range(0,len(Base)):
        posmag = np.sum(Base[i,:])
        Avg_mag[i] = np.divide(((-1)* (np.shape(Base)[1] - posmag) + posmag),np.shape(Base)[1])
    O_z_mat = np.diag(Avg_mag)
    return O_z_mat

def Z_i_PXP_Entry_basis(n, i, Subspace):
    '''
    Building Z_i in basis of PXP entry ALSO WORKS FOR X_i EXTENDED!!!!
    :param n: number of atoms
    :param i: index of Z_i
    :return: Z_i matrix
    '''
    Base = Subspace(n)
    Z_i_mag = np.empty((len(Base)))
    for j in range(0, len(Base)):
        if Base[j,i-1] == 1:
            Z_i_mag[j] = 1
        else:
            Z_i_mag[j] = -1
    return np.diag(Z_i_mag)

def Z_i_Spin_Basis(n,i): # calculates Z for general i
    '''
    Z_i matrix for TI 2**n basis
    :param n: dimension 2**n (spin)
    :param i: atom index of Z operator
    :return: Z_i matrix in dimension 2**n
    '''
    if i <= n:
        Z_geni = np.kron(np.identity(2**(i-1)),np.kron(Z_i, np.identity(2**(n-i))))
    else:
        Z_geni= np.identity(2**n)
    return Z_geni

# def X_i_Dict_PXP_Full_Basis(n, i):
#     '''
#     X_i in PXP space - CAN'T BE WRITTEN IN SUBSPACE BECAUSE IT TAKES OUT OF SUBSPACE!!!
#     :param n: No of atoms
#     :param i: site wanted
#     :return: X dictionary (first col = the initial vectors ; second col = final vectors)
#     '''
#
#     Base = Basis(n)
#     Search = Base.copy()
#     x = np.empty((len(Base), 2))
#     x[:,0] = np.arange(0,int(len(Base)),1)
#     for j in range(0,len(Base)):
#         if Base[j,i] == 1:
#             Search[j,i] = 0
#         else:
#             Search[j,i] = 1
#         x[j,1] = int(np.nonzero(np.all(Base == Search[j,:],axis=1))[0]) #gets basis row index of new vector after X acts on initial state vector
#     return x
#
# def X_i_Mat_PXP_Full_Basis(n, i):
#     '''
#     X_i in PXP space - CAN'T BE WRITTEN IN SUBSPACE BECAUSE IT TAKES OUT OF SUBSPACE!!!
#     :param n: No of atoms
#     :param i: site wanted
#     :return: X Matrix for specific i
#     '''
#     X_mat = np.empty((2**n,2**n))
#     dict = X_i_Dict_PXP_Full_Basis(n, i)
#     for j in range(0,2**n):
#         X_mat[int(dict[j,0]),int(dict[j,1])] = 1
#     return X_mat

def X_n_Dict_PXP_X_i_Extended_Subspace(n):
    '''
    Builds final atom X_n dictionary (start state -> end state) in X_i extended Subspace basis of PXP entry Sparsely
    :param n: number of atoms
    :return: X_f matrix sparse
    '''
    Base = PXP_Subspace_Algo_extended_X_i(n)
    Search = Base.copy()
    x = np.empty((len(Base), 2))
    x[:, 0] = np.arange(0, int(len(Base)), 1)
    for j in range(0, len(Base)):
        if Base[j, n-1] == 1:
            Search[j, n-1] = 0
        elif Base[j,n-1] == 0:
            Search[j, n-1] = 1
        x[j, 1] = int(np.nonzero(np.all(Base == Search[j, :], axis=1))[0])  # gets basis row index of new vector after X acts on initial state vector
    return x

def X_n_Mat_PXP_X_i_Extended_Subspace(n):
    '''
    X_n (final atom) matrix in PXP extended X_i subspace
    :param n: No of atoms
    :return: X Matrix for final atom in extended X_i basis
    '''
    X_mat = np.zeros((Extended_X_i_Subspace_basis_count_faster(n),Extended_X_i_Subspace_basis_count_faster(n)))
    dict = X_n_Dict_PXP_X_i_Extended_Subspace(n)
    for j in range(0,Extended_X_i_Subspace_basis_count_faster(n)):
        X_mat[int(dict[j,0]),int(dict[j,1])] = 1
    return X_mat

def X_i_Spin_Basis(n,i):
    '''
    :param n: full dimension (2**n) X_i sparse matrix (Regular KRON SPIN BASIS FOR TI !!!)
    :param i: atom index of X operator
    :return: X_i in dimension 2**n sparsely!
    '''
    if i <= n:
        X_geni = np.kron(np.identity(2**(i-1)),np.kron(X_i, np.identity(2**(n-i))))
    else:
        X_geni= np.identity(2**n)
    return X_geni

def Z_i_Coupling_PXP_Entry_to_TI(n_PXP, n_TI, h_c):
        """
        Z_i nature coupling matrix (two-site) of Subspace PXP version and TI regular (dimension is Fib(n_PXP+3)*(2**n_TI) x Fib(n_PXP+3)*(2**n_TI))
        :param n_PXP: Number of PXP atoms (0 to whatever)
        :param n_TI: Number of TI atoms (0 to whatever)
        :param h_c: coupling strength parameter
        :return: matrix in (full) Fib(n_PXP+3)*(2**n_TI) x Fib(n_PXP+3)*(2**n_TI) dimension
        """
        d_TOT = Subspace_basis_count_faster(n_PXP)+2 ** n_TI   # Total dimension
        Coupling = (h_c) * np.kron(Z_i_PXP_Entry_basis(n_PXP, n_PXP, PXP_Subspace_Algo), Z_i_Spin_Basis(n_TI,1))
        if n_TI == 0 or n_PXP == 0 or h_c == 0:
            Coupterm = np.zeros((d_TOT, d_TOT))
        else:
            Coupterm = Coupling
        return Coupterm

def X_i_Coupling_PXP_Entry_to_TI(n_PXP,n_TI,h_c):
    """
    X_i EXTENDED BASIS!!!! nature coupling matrix (two-site) of Subspace PXP version and TI regular (dimension is [Fib(n_PXP+3)+Fib(n_PXP)]*(2**n_TI) x [Fib(n_PXP+3)+Fib(n_PXP)]*(2**n_TI))
    :param n_PXP: Number of PXP atoms (0 to whatever)
    :param n_TI: Number of TI atoms (0 to whatever)
    :param h_c: coupling strength parameter
    :return: matrix in full [Fib(n_PXP+3)+Fib(n_PXP)]*(2**n_TI) x [Fib(n_PXP+3)+Fib(n_PXP)]*(2**n_TI) dimension
    """
    d_TOT = Extended_X_i_Subspace_basis_count_faster(n_PXP)+(2 ** n_TI)   # Total dimension
    Coupling = (h_c) * np.kron(X_n_Mat_PXP_X_i_Extended_Subspace(n_PXP), X_i_Spin_Basis(n_TI,1))
    if n_TI == 0 or n_PXP == 0 or h_c == 0:
        Coupterm = np.zeros((d_TOT, d_TOT))
    else:
        Coupterm = Coupling
    return Coupterm


def PXP_EBE_BathHam(n_PXP, n_TI, Subspace, J, h_x, h_z, h_c, h_imp, m): #Needs fixing
    """
    FULL 2**(n_tot) dimension Z_i nature COUPLED PXP and TI hamiltonian Builder
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
    PXP = PXP_Ham_OBC_Entrybyentry(n_PXP, Subspace)
    TI = TIOBCNewImpure2(n_TI, J, h_x, h_z, h_imp, m)
    d_TI = 2 ** n_TI
    d_PXP = len(PXP)
    HamNoCoupl = np.add(np.kron(PXP, np.identity(d_TI)), np.kron(np.identity(d_PXP), TI))
    TotalHam = np.add(HamNoCoupl, Z_i_Coupling_PXP_Entry_to_TI(n_PXP, n_TI, h_c))
    return TotalHam

def Neelstate_spin_base_faster(n_PXP):
    '''
    faster method Generates Neelstate in binary basis: first 1 in first site for ODD, first 1 in second site for EVEN!!!!!
    :param n_PXP: No. of atoms
    :return: Vector (Neelstate)
    '''
    Neel = np.zeros((n_PXP))
    if n_PXP % 2 ==0:
        Even = np.arange(1,n_PXP,2)
        Neel[Even]=1
    else:
        Odd = np.arange(0,n_PXP,2)
        Neel[Odd]=1
    return Neel

def Neelstate_spin_base_faster_old(n_PXP):
    '''
    faster method Generates Neelstate in binary basis: first 1 in first site for Odd and Even
    :param n_PXP: No. of atoms
    :return: Vector (Neelstate)
    '''
    Neel = np.zeros((n_PXP))
    Neel_arr = np.arange(0,n_PXP,2)
    Neel[Neel_arr]=1
    return Neel

def Neel_Subspace_Basis(n_PXP):
    '''
    Finds Neelstate in Subspace (PXP) Basis
    :param n_PXP: Number of PXP atoms
    :return: vector of Neelstate in Subspace Basis of PXP
    '''
    Neel = Neelstate_spin_base_faster(n_PXP)
    Subspace = PXP_Subspace_Algo(n_PXP)
    Neel_index = np.where(np.all(Subspace==Neel,axis=1))
    Subs_Base_Neel = np.zeros((Subspace_basis_count_faster(n_PXP)))
    Subs_Base_Neel[Neel_index] = 1
    return Subs_Base_Neel

def Neel_X_i_Extended_Subspace_Basis(n_PXP):
    '''
    Finds Neelstate in X_i extended PXP Subspace
    :param n_PXP: Number of PXP atoms
    :return: vector of Neelstate in Subspace Basis of PXP
    '''
    Neel = Neelstate_spin_base_faster(n_PXP)
    Subspace = PXP_Subspace_Algo_extended_X_i(n_PXP)
    Neel_index = np.where(np.all(Subspace==Neel,axis=1))
    Subs_Base_Neel = np.zeros((Extended_X_i_Subspace_basis_count_faster(n_PXP)))
    Subs_Base_Neel[Neel_index] = 1
    return Subs_Base_Neel

def Neel_EBE_Haar(n_PXP, n_TI): #Haarstate from Coupling_To_Bath
    """
    Combination of the Neel EBE Subspace basis and Haar states
    :param n_PXP:  Number of PXP chain atoms
    :param n_TI: number of TI chain atoms
    :return: Neel-Haar combined state
    """
    NeelHaarstate = np.kron(Neel_Subspace_Basis(n_PXP), Haarstate(n_TI))
    return NeelHaarstate

def Neel_EBE_Haar_X_i_Extended(n_PXP, n_TI): #Haarstate from Coupling_To_Bath
    """
    Combination of the Neel EBE X_i extended PXP Subspace basis and Haar states in regular spin basis
    :param n_PXP:  Number of PXP chain atoms
    :param n_TI: number of TI chain atoms
    :return: Neel-Haar combined state
    """
    NeelHaarstate = np.kron(Neel_X_i_Extended_Subspace_Basis(n_PXP), Haarstate(n_TI))
    return NeelHaarstate

def TI_Neelstate(n_TI):
    '''
    Tilted Ising (only) Neel state - just Neel in 2^n kron basis
    :param n_TI: Tilted Ising atom number
    :return:
    '''
    Neel_base= np.zeros((2**n_TI))
    Neel= Neel_base.copy()
    if n_TI%2==0:
        Base_array= np.arange(1,n_TI,2)
        Neel_no= np.sum(2**(Base_array))
        Neel[int(Neel_no)]=1
        Neel=np.flip(Neel)
    else:
        Base_array= np.arange(0,n_TI,2)
        Neel_no= np.sum(2**(Base_array))
        Neel[int(Neel_no)]=1
        Neel=np.flip(Neel)
    return Neel

# def Neel_EBE_Haar_seeded(n_PXP, n_TI,seed): #Haarstate from Coupling_To_Bath
#     """
#     Combination of the Neel EBE Subspace basis and SEEDED Haar states
#     :param n_PXP:  Number of PXP chain atoms
#     :param n_TI: number of TI chain atoms
#     :return: Neel-Haar combined state
#     """
#     NeelHaarstate = np.kron(Neel_Subspace_Basis(n_PXP), Haarstate_seed(n_TI,seed))
#     return NeelHaarstate




# TODO ALGORITHM IMPROVEMENT SUGGESTION FOR EVERYTHING IN DICTIONARIES = GO ONLY UP UNTIL HALF THE DICT WITH LOOP AND THEN TRANSPOSE AND CONNECT
# TODO Check where I put Subspace_basis_count_faster in code and switch to len()