import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from Coupling_To_Bath import *
from PXP_E_B_E_Sparse import *
import numpy.linalg as la
import scipy.linalg as scla
from time import time
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations
import scipy.sparse as sp
from scipy.special import comb
#from Cluster_Sparse_Osc_Para import *

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
    :return: Scalar - No of subspace vectors = fib(n+3)

    '''
    Fibonacci = np.arange(0,n+3)
    lengthFibo = len(Fibonacci)
    sqrtFive= np.sqrt(5)
    alpha = (1 + sqrtFive) / 2
    beta = (1 - sqrtFive) / 2
    F_n = np.rint(((alpha ** Fibonacci)- (beta ** Fibonacci)) / (sqrtFive))
    return int(F_n[len(F_n)-1])

def PXP_Subspace_Algo(n): #TODO option to add stop condition instead of feeding subspace basis count
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

#########################################################################################################################################################
#####                                   GOOD FOR NOTHING CODE I DID IN ORDER TO UNDERSTAND WHAT'S GOING ON                                          #####
#########################################################################################################################################################

def PXP_basis_gen(n):
    '''
    Generates basis of all posible PXP states in fib(n_PXP+3) basis {mapping between both bases PXP_Subspace_Algo[i,:] : -> vec=np.zeros(fib(n+3)) & vec[i]=1}
    :param n: number of PXP atoms
    :return: a basis of all PXP states, in a matrix of size fib(n_PXP+3)] X [fib(n_PXP+3)] {mapping between both bases PXP_Subspace_Algo[i,:] : -> vec=np.zeros(fib(n+3)) & vec[i]=1}
    '''
    return np.eye(Subspace_basis_count_faster(n))

def PXP_inverse_map(n_PXP,basis_vec_ind):
    '''
    inverse mapping between standard subspace PXP basis to the PXP Algo basis (binary!!!)
    :param n_PXP: total number of PXP atoms
    :param basis_vec_ind: basis vector index of standard basis (location of 1 in basis vector) - starts from 0 !!!!
    :return: the vector of algo.
    '''
    Algo_basis_vec = PXP_Subspace_Algo(n_PXP)[basis_vec_ind,:]
    return Algo_basis_vec

def TI_basis_gen(n):
        '''
        Generates basis of all posible TI states in 2**N basis
        :param n: number of TI atoms
        :return: a basis of all TI states, in a matrix of size [2^(n)] X [2^(n)]
        '''
        return np.eye(2**n)

def Full_PXP_TI_Basis(n_PXP,n_TI):
    '''
    generates Full space of PXP-TI vector Basis; ordering of the basis is that first PXP state is kronickered with all TI states, and so on for the second PXP state etc.
    :param n_PXP: number of PXP atoms
    :param n_TI: number of TI atoms
    :return: matrix of size  [fib(n_PXP +3)*2^(n_TI)] X  [fib(n_PXP +3)*2^(n_TI)] with full basis rows ; ordering of the basis is that first PXP state is kronickered with all TI states, and so on for the second PXP state etc.
    '''
    return np.eye(Subspace_basis_count_faster(n_PXP)*(2**n_TI))

def PXP_TI_kron_decompose(n_PXP,n_TI,basis_vec_ind):
    '''
    decomposing subspace PXP kron TI basis into PXP basis state and a TI basis state (just the standard basis for now) indeces
    :param n_PXP: number of PXP atoms
    :param n_TI: number of TI atoms
    :param basis_vec_ind: index of joint PXP TI standard basis vector (where the 1 is) - starts from 0!!!
    :return: index of PXP subspace standard basis and index of TI standard basis vectors
    '''
    PXP_subspace_basis_index = int((basis_vec_ind) / (2**n_TI))
    TI_subspace_basis_index = (basis_vec_ind) % (2**n_TI) # % is modulus - gives the remainder from division
    return PXP_subspace_basis_index, TI_subspace_basis_index

def Full_PXP_TI_inverse_mapping(n_PXP, n_TI, Full_vec):
    '''
    inverse mapping between full PXP TI standard vector (standard subspace PXP vector kron TI basis vector) to ALGO PXP vector and TI standard vector
    :param n_PXP: number of PXP atoms
    :param n_TI: number of TI atoms
    :param Full_vec:  PXP TI standard basis vector
    :return:  PXP algo subspace vector and standard basis TI vector
    '''
    basis_vec_ind = np.argwhere(Full_vec==1)
    PXP_ind, TI_ind = PXP_TI_kron_decompose(n_PXP, n_TI, basis_vec_ind)
    PXP_algo_base_vec = PXP_inverse_map(n_PXP, PXP_ind)
    TI_base_vec = np.zeros(2**n_TI)
    TI_base_vec[TI_ind] = 1
    return PXP_algo_base_vec, TI_base_vec

def VecSpan(n_PXP,n_TI,
              VecState):
    """
    Spans some vector (VecState) in some chosen basis.
    :param Mat: Input matrix of which rows form a basis
    :param VecState: Some vector state (initial state usually) that we want to span in eigenstate basis
    :return: vector of weights (in Eigenstate basis)
    """
    Basis = Full_PXP_TI_Basis(n_PXP,n_TI)
    W = np.dot(Basis,VecState)
    return W

######################################################################################################################################################
####                                                PXP/TI CUT ENTANGLEMENT ENTROPY CODE                                                          ####
######################################################################################################################################################



def Evec_Reshape_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2):
    '''
    Reshapes each eigenvector to the shape of a matrix for schmidt decomposition (splitting at PXP - TI boundary)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param Eigenvec: eigenvectors (as cols of a matrix)
    :return: rank 3 tensor of matrices of each eigenstate's decomposition to joint basis
    '''
    eval, evec = la.eigh((PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m)).todense())
    Ham_Dim = len(eval) #also number of evecs
    Tensor = np.empty((Subspace_basis_count_faster(n_PXP), 2**(n_TI), Ham_Dim))
    for i in range(0, Ham_Dim):
        Tensor[:,:,i]=np.reshape(evec[:,i],(Subspace_basis_count_faster(n_PXP),2**(n_TI)))
    return Tensor, eval


def Evec_SVD_PXP_TI(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    singular values of each matrix of eigenvector in the tensor (see Eigenvec_Reshape) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: singular values of the matrix
    '''
    Tensor, eval = Evec_Reshape_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(2**(n_TI),Subspace_basis_count_faster(n_PXP))
    SVD_vec_mat = np.zeros((len(eval),SVD_num))
    for i in range(0,len(eval)):
        SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    return SVD_vec_mat, eval

def Evec_SVD_PXP_TI_Cluster(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    Cluster Calculation of singular values of each matrix of eigenvector in the tensor (see Eigenvec_Reshape) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: singular values of the matrix, eigenvalues (saved in numpy format)
    '''
    try:
        os.mkdir('EE_PXP_{}_TI_{}'.format(n_PXP, n_TI))
    except:
        pass
    Tensor, eval = Evec_Reshape_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(2**(n_TI),Subspace_basis_count_faster(n_PXP))
    SVD_vec_mat = np.empty((len(eval),SVD_num))
    for i in range(0,len(eval)):
         SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    np.save(os.path.join('EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Entanglement_h_c_{}.npy'.format(h_c)), SVD_vec_mat)
    np.save(os.path.join('EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Eval_h_c_{}.npy'.format(h_c)), eval)
    return

#Evec_SVD_PXP_TI_Cluster(n_PXP, n_TI, h_c)

def Entanglement_entropy_calc_PXP_TI(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    calculating entanglement entropy of each eigenstate from singular values -sigma^2ln(sigma) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat, eval = Evec_SVD_PXP_TI(n_PXP, n_TI, np.round(h_c,5))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    plt.scatter(eval, np.round(Entanglement_entropy_vec,32))
    plt.title('Entanglement vs Energy for {} PXP & {} TI atoms, $h_c$ {} '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return plt.show()

def EE_Avg_plot(n_PXP, n_TI,h_c_start, h_c_max, interval_width):
    '''
    Plots average entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of average entanglement vs h_c
    '''
    h_c= np.arange(h_c_start,h_c_max+0.1,0.1)
    avg= np.empty((np.size(h_c)))
    std= avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i,1)*10-h_c_start*10)], std[int(np.round(i,1)*10-h_c_start*10)] = Entanglement_entropy_avg_std(n_PXP, n_TI, i, interval_width)
    plt.plot(h_c,avg, color='b')
    plt.title('Average Entanglement vs coupling strength $h_c$ for {} PXP & {} TI atoms'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Average Entanglement Entropy')
    #plt.savefig('Figures/Entanglement_Entropy/Average_Entanglement_Entropy')
    return plt.show()

def Entanglement_entropy_avg_std(n_PXP, n_TI, h_c, interval_width): #8 PXP 10 TI max calc
    '''
    Calculating entanglement entropy average and standard deviation for some energy interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: average, standard deviation (two scalars)
    '''
    SVD_vec_mat, eval = Evec_SVD_PXP_TI(n_PXP, n_TI, np.round(h_c,5))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    eval_interval_arg_min= np.min(np.argwhere(eval>(h_c-interval_width)))
    print('Minimum interval point', eval_interval_arg_min)
    eval_interval_arg_max= np.min(np.argwhere(eval>(h_c+interval_width)))
    print('Maximum interval point', eval_interval_arg_max)
    average = np.average(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    std = np.std(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    return np.round(average,4), np.round(std,4)

def EE_Avg_Std_plot(n_PXP, n_TI,h_c_start, h_c_max, interval_width):
    '''
    Plots average and Standard deviation of entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of average entanglement and Standard deviation vs h_c
    '''
    h_c = np.arange(h_c_start, h_c_max + 0.1, 0.1)
    avg = np.empty((np.size(h_c)))
    std = avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i, 1) * 10 - h_c_start * 10)], std[int(np.round(i, 1) * 10 - h_c_start * 10)] = Entanglement_entropy_avg_std(n_PXP, n_TI, i, interval_width)
    plt.plot(h_c[:], avg[:], linestyle='-',color='b')
    #plt.errorbar(h_c[:], avg[:], yerr=std[:],marker='s', markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3)
    plt.fill_between(h_c[:], avg[:] - std[:], avg[:]+std[:], alpha=0.4)
    plt.title('Average Entanglement vs $h_c$ for {} PXP & {} TI atoms'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Average Entanglement Entropy')
    plt.show()

def Entanglement_entropy_avg_std_Cluster(n_PXP, n_TI, h_c, interval_width): #8 PXP 10 TI max calc
    '''
    Calculating entanglement entropy average and standard deviation for some energy interval from cluster data!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: average, standard deviation (two scalars)
    '''
    SVD_vec_mat = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Entanglement_h_c_{}.npy'.format(h_c)))
    eval = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Eval_h_c_{}.npy'.format(h_c)))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    eval_interval_arg_min= np.min(np.argwhere(eval>(h_c-interval_width)))
    eval_interval_arg_max= np.min(np.argwhere(eval>(h_c+interval_width)))
    average = np.average(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    std = np.std(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    try:
        os.mkdir('EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP,n_TI,interval_width))
    except:
        pass
    np.save(os.path.join('EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP,n_TI,interval_width),'Average_h_c_{}_width_{}.npy'.format(h_c,interval_width)), average)
    np.save(os.path.join('EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP,n_TI,interval_width),'STD_h_c_{}_width_{}.npy'.format(h_c,interval_width)), std)
    return
#Entanglement_entropy_avg_std_Cluster(n_PXP, n_TI, h_c, interval_width=1)



#
#
# def EE_Std_plot(n_PXP, n_TI,h_c_start, h_c_max, interval_width):
#     '''
#     Plots Standard deviation of entanglement for given interval
#     :param n_PXP: No. of PXP atoms
#     :param n_TI: No. of TI atoms
#     :param h_c: coupling strength
#     :param interval_width: energy interval width +-delta
#     :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
#     :return: plot of Standard deviation vs h_c
#     '''
#     h_c= np.arange(h_c_start,h_c_max+0.1,0.1)
#     avg= np.empty((np.size(h_c)))
#     std= avg.copy()
#     for i in np.nditer(h_c):
#         avg[int(np.round(i,1)*10-h_c_start*10)], std[int(np.round(i,1)*10-h_c_start*10)] = Entanglement_entropy_avg_std(n_PXP, n_TI, i, interval_width)
#     plt.plot(h_c,std, color='r')
#     plt.title('Entanglement STD vs coupling strength $h_c$ for {} PXP & {} TI atoms'.format(n_PXP, n_TI))
#     plt.xlabel('Coupling Strength $h_c$')
#     plt.ylabel('Standard Deviation of Entanglement Entropy')
#     plt.savefig('Figures/Entanglement_Entropy/Standard_Deviation_of_Entanglement_Entropy')
#     plt.show()


def EE_Avg_Std_Cluster_plot(n_PXP, n_TI,h_c_start, h_c_max,interval_width):
    '''
    Plots average and Standard deviation of entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of average entanglement and Standard deviation vs h_c
    '''
    h_c = np.arange(h_c_start, h_c_max + 0.1, 0.1)
    avg = np.empty((np.size(h_c)))
    std = avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i, 1) * 10 - h_c_start * 10)]  = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP,n_TI,interval_width),'Average_h_c_{}_width_{}.npy'.format(np.round(i,1),interval_width)))
        std[int(np.round(i, 1) * 10 - h_c_start * 10)] = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP,n_TI,interval_width),'STD_h_c_{}_width_{}.npy'.format(np.round(i,1),interval_width)))
    plt.plot(h_c[:], avg[:], linestyle='-',color='b')
    plt.fill_between(h_c[:], avg[:] - std[:], avg[:]+std[:], alpha=0.4)
    plt.title('Average Entanglement vs $h_c$ for {} PXP & {} TI atoms'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Average Entanglement Entropy')
    plt.savefig('Figures/Entanglement_Entropy/Average_EE_filled_{}_PXP_{}_TI_{}_max_h_c'.format(n_PXP,n_TI,h_c_max))
    plt.show()

#Evec_SVD_PXP_TI_Cluster(n_PXP, n_TI, h_c)
#Entanglement_entropy_avg_std_Cluster(n_PXP, n_TI, h_c, interval_width=1)

#################################################################################### XX coupling #############################################################################

def Evec_Reshape_True_X_i_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2):
    '''
    Reshapes each eigenvector of XX coup. to the shape of a matrix for schmidt decomposition (splitting at PXP - TI boundary)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param Eigenvec: eigenvectors (as cols of a matrix)
    :return: rank 3 tensor of matrices of each eigenstate's decomposition to joint basis
    '''
    eval, evec = la.eigh((PXP_TI_coupled_Sparse_Xi(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m)).todense())
    Ham_Dim = len(eval) #also number of evecs
    Tensor = np.empty((Extended_X_i_Subspace_basis_count_faster(n_PXP), 2**(n_TI), Ham_Dim))
    for i in range(0, Ham_Dim):
        Tensor[:,:,i]=np.reshape(evec[:,i],(Extended_X_i_Subspace_basis_count_faster(n_PXP),2**(n_TI)))
    return Tensor, eval


def Evec_SVD_True_X_i_PXP_TI(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    singular values of each matrix of eigenvector in the tensor (see Eigenvec_Reshape) splitting at PXP - TI Boundary XX coupl
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: singular values of the matrix
    '''
    Tensor, eval = Evec_Reshape_True_X_i_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(2**(n_TI),Extended_X_i_Subspace_basis_count_faster(n_PXP))
    SVD_vec_mat = np.zeros((len(eval),SVD_num))
    for i in range(0,len(eval)):
        SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    return SVD_vec_mat, eval

def Evec_SVD_True_X_i_PXP_TI_Cluster(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    ''' XX coupling!
    Cluster Calculation of singular values of each matrix of eigenvector in the tensor (see Eigenvec_Reshape) splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: singular values of the matrix, eigenvalues (saved in numpy format)
    '''
    try:
        os.mkdir('EE_PXP_{}_TI_{}_True_X_i'.format(n_PXP, n_TI))
    except:
        pass
    Tensor, eval = Evec_Reshape_True_X_i_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(2**(n_TI),Extended_X_i_Subspace_basis_count_faster(n_PXP))
    SVD_vec_mat = np.empty((len(eval),SVD_num))
    for i in range(0,len(eval)):
         SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    np.save(os.path.join('EE_PXP_{}_TI_{}_True_X_i'.format(n_PXP,n_TI),'Entanglement_h_c_{}_True_X_i'.format(h_c)), SVD_vec_mat)
    np.save(os.path.join('EE_PXP_{}_TI_{}_True_X_i'.format(n_PXP,n_TI),'Eval_h_c_{}_True_X_i'.format(h_c)), eval)
    return

#Evec_SVD_PXP_TI_Cluster(n_PXP, n_TI, h_c)

def Entanglement_entropy_calc_True_X_i_PXP_TI(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    ''' XX coupling!
    calculating entanglement entropy of each eigenstate from singular values -sigma^2ln(sigma) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat, eval = Evec_SVD_True_X_i_PXP_TI(n_PXP, n_TI, np.round(h_c,5))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    plt.scatter(eval, np.round(Entanglement_entropy_vec,32))
    plt.title('Entanglement vs Energy for {} PXP & {} TI atoms, $h_c$ {} XX coupl '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return plt.show()


def Entanglement_entropy_True_X_i_avg_std(n_PXP, n_TI, h_c, interval_width): #8 PXP 10 TI max calc
    ''' XX coupling!
    Calculating entanglement entropy average and standard deviation for some energy interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: average, standard deviation (two scalars)
    '''
    SVD_vec_mat, eval = Evec_SVD_True_X_i_PXP_TI(n_PXP, n_TI, np.round(h_c,5))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    print(Entanglement_entropy_vec)
    eval_interval_arg_min= np.min(np.argwhere(eval>(h_c-interval_width)))
    print('Minimum interval point', eval_interval_arg_min)
    eval_interval_arg_max= np.min(np.argwhere(eval>(h_c+interval_width)))
    print('Maximum interval point', eval_interval_arg_max)
    average = np.average(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    std = np.std(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    return np.round(average,4), np.round(std,4)

# def EE_True_X_i_Avg_plot(n_PXP, n_TI,h_c_start, h_c_max, interval_width):
#     '''
#     Plots average entanglement for given interval
#     :param n_PXP: No. of PXP atoms
#     :param n_TI: No. of TI atoms
#     :param h_c: coupling strength
#     :param interval_width: energy interval width +-delta
#     :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
#     :return: plot of average entanglement vs h_c
#     '''
#     h_c= np.arange(h_c_start,h_c_max+0.1,0.1)
#     avg= np.empty((np.size(h_c)))
#     std= avg.copy()
#     for i in np.nditer(h_c):
#         avg[int(np.round(i,1)*10-h_c_start*10)], std[int(np.round(i,1)*10-h_c_start*10)] = Entanglement_entropy_True_X_i_avg_std(n_PXP, n_TI, i, interval_width)
#     plt.plot(h_c,avg, color='b')
#     plt.title('Average Entanglement vs coupling strength $h_c$ for {} PXP & {} TI atoms XX coupl'.format(n_PXP, n_TI))
#     plt.xlabel('Coupling Strength $h_c$')
#     plt.ylabel('Average Entanglement Entropy')
#     #plt.savefig('Figures/Entanglement_Entropy/Average_Entanglement_Entropy')
#     return plt.show()

def EE_True_X_i_Avg_Std_plot(n_PXP, n_TI,h_c_start, h_c_max, interval_width):
    '''XX coupling
    Plots average and Standard deviation of entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of average entanglement and Standard deviation vs h_c
    '''
    h_c = np.arange(h_c_start, h_c_max + 0.1, 0.1)
    avg = np.empty((np.size(h_c)))
    std = avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i, 1) * 10 - h_c_start * 10)], std[int(np.round(i, 1) * 10 - h_c_start * 10)] = Entanglement_entropy_True_X_i_avg_std(n_PXP, n_TI, i, interval_width)
    plt.plot(h_c[:], avg[:], linestyle='-',color='b')
    #plt.errorbar(h_c[:], avg[:], yerr=std[:],marker='s', markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3)
    plt.fill_between(h_c[:], avg[:] - std[:], avg[:]+std[:], alpha=0.4)
    plt.title('Average Entanglement vs $h_c$ for {} PXP & {} TI atoms XX coupl'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Average Entanglement Entropy')
    plt.show()

def Entanglement_entropy_True_X_i_avg_std_Cluster(n_PXP, n_TI, h_c, interval_width): #8 PXP 10 TI max calc
    ''' XX coupling
    Calculating entanglement entropy average and standard deviation for some energy interval from cluster data!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: average, standard deviation (two scalars)
    '''
    SVD_vec_mat = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}_True_X_i'.format(n_PXP,n_TI),'Entanglement_h_c_{}_True_X_i.npy'.format(h_c)))
    eval = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}_True_X_i'.format(n_PXP,n_TI),'Eval_h_c_{}_True_X_i.npy'.format(h_c)))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    eval_interval_arg_min= np.min(np.argwhere(eval>(h_c-interval_width)))
    eval_interval_arg_max= np.min(np.argwhere(eval>(h_c+interval_width)))
    average = np.average(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    std = np.std(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    try:
        os.mkdir('EE_PXP_{}_TI_{}_True_X_i/AVG_STD_width_{}_True_X_i'.format(n_PXP,n_TI,interval_width))
    except:
        pass
    np.save(os.path.join('EE_PXP_{}_TI_{}_True_X_i/AVG_STD_width_{}_True_X_i'.format(n_PXP,n_TI,interval_width),'Average_h_c_{}_width_{}_True_X_i'.format(h_c,interval_width)), average)
    np.save(os.path.join('EE_PXP_{}_TI_{}_True_X_i/AVG_STD_width_{}_True_X_i'.format(n_PXP,n_TI,interval_width),'STD_h_c_{}_width_{}_True_X_i'.format(h_c,interval_width)), std)
    return


def EE_True_X_i_Avg_Std_Cluster_plot(n_PXP, n_TI,h_c_start, h_c_max,interval_width):
    '''
    Plots average and Standard deviation of entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of average entanglement and Standard deviation vs h_c
    '''
    h_c = np.arange(h_c_start, h_c_max + 0.1, 0.1)
    avg = np.empty((np.size(h_c)))
    std = avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i, 1) * 10 - h_c_start * 10)]  = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}_True_X_i/AVG_STD_width_{}_True_X_i'.format(n_PXP,n_TI,interval_width),'Average_h_c_{}_width_{}.npy'.format(np.round(i,1),interval_width)))
        std[int(np.round(i, 1) * 10 - h_c_start * 10)] = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}_True_X_i/AVG_STD_width_{}_True_X_i'.format(n_PXP,n_TI,interval_width),'STD_h_c_{}_width_{}.npy'.format(np.round(i,1),interval_width)))
    plt.plot(h_c[:], avg[:], linestyle='-',color='b')
    plt.fill_between(h_c[:], avg[:] - std[:], avg[:]+std[:], alpha=0.4)
    plt.title('Average Entanglement vs $h_c$ for {} PXP & {} TI atoms XX coupl'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Average Entanglement Entropy')
    plt.savefig('Figures/Entanglement_Entropy/Average_EE_True_X_i_filled_{}_PXP_{}_TI_{}_max_h_c'.format(n_PXP,n_TI,h_c_max))
    plt.show()

#Evec_SVD_True_X_i_PXP_TI_Cluster(n_PXP, n_TI, h_c)
#Entanglement_entropy_True_X_i_avg_std_Cluster(n_PXP, n_TI, h_c, interval_width=1)

#####################################################################################################################################################################
#                                                                                   PXP-PXP CUT EE CODE                                                             #
######################################################################################################################################################################

#####                                                             PXP-PXP CUT EE CODE FOR PXP OBC ONLY                                                            #####

def PXP_PXP_Full_Permutation_Basis(n_PXP):
    '''
    Builds the full permutation basis of options of a cut in the middle of a PXP chain (meaning all left side states appended with all the right side states)
    :param n_PXP: Total number of PXP atoms
    :return: matrix of dimension [Fib((n_PXP+3)/2)*Fib((n_PXP+3)/2)]x(n_PXP) for EVEN number n_PXP OR [Fib((n_PXP+3)/2)*Fib((n_PXP+3)/2+1)]x(n_PXP) for ODD
    '''
    Base_full_PXP= PXP_Subspace_Algo(n_PXP) #dim Fib(n_PXP+3)x(n_PXP)
    if n_PXP%2 == 0: #Even PXP number, the cut is in the middle
        Base_left_cut= PXP_Subspace_Algo(int(np.divide(n_PXP,2)))
        Base_right_cut= Base_left_cut.copy()
        Full_Permutation_basis = np.zeros((len(Base_left_cut)*len(Base_right_cut),n_PXP))  # dim [Fib((n_PXP+3)/2)*Fib((n_PXP+3)/2)]x(n_PXP) for even
    else: #ODD PXP number, the cut is taken in a way that the LHS has an even number of atoms
        Base_left_cut= PXP_Subspace_Algo(int(np.divide(n_PXP,2)))
        Base_right_cut= PXP_Subspace_Algo(n_PXP-int(np.divide(n_PXP,2)))
        Full_Permutation_basis = np.zeros((len(Base_left_cut)*len(Base_right_cut),n_PXP))  # dim Fib[((n_PXP+3)/2)*Fib((n_PXP+3)/2 +1)]x(n_PXP) for even
    for i in range(0,len(Base_left_cut)):
        for j in range(0,len(Base_right_cut)): #TODO make more efficient?
            Full_Permutation_basis[len(Base_right_cut)*i+j,:]= np.hstack((Base_left_cut[i,:],Base_right_cut[j,:]))
    #Check_Even= np.isclose(len(Full_Permutation_basis),Subspace_basis_count_faster(int(np.divide(n_PXP,2)))*Subspace_basis_count_faster(int(np.divide(n_PXP,2))))
    #Check_Odd= np.isclose(len(Full_Permutation_basis),Subspace_basis_count_faster(int(np.divide(n_PXP,2)))*Subspace_basis_count_faster(int(np.divide(n_PXP,2))+1))
    # print(Full_Permutation_basis)
    return Base_full_PXP, Base_left_cut, Base_right_cut, Full_Permutation_basis

def PXP_Full_Permutation_Basis_To_Reg_Basis_Mapping(n_PXP):
    '''
    mapping of the Reg PXP basis to the full permutation basis, with vector of kernels
    :param n_PXP: No. of PXP atoms
    :return: mapping (dictionary) of Reg PXP basis states indeces -> new full permutation basis states indeces, and vector of kernel indeces of Permutation states
    '''
    Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP= PXP_PXP_Full_Permutation_Basis(n_PXP)
    Ordering_vec=np.zeros(len(Base_full_PXP))
    for i in range(0,len(Base_full_PXP)):
        Ordering_vec[i]= np.squeeze(np.nonzero(np.all((Permutation_basis_PXP==Base_full_PXP[i,:]),axis=1)))
    # Full_Permu_state_indeces=np.arange(0,len(Permutation_basis_PXP),1)
    # Permutation_kernel=np.squeeze(np.where(np.in1d(Full_Permu_state_indeces,Ordering_vec)==False)) #indeces of permutation basis vectors that go to zero in schmidt matrix!!!
    Mapping_Dict=np.transpose(np.array((np.arange(0,len(Base_full_PXP),1),Ordering_vec)))
    Mapping_Mat= np.zeros((len(Permutation_basis_PXP),len(Base_full_PXP)))
    Mapping_Mat[Mapping_Dict[:,1].astype(int),Mapping_Dict[:,0].astype(int)]=1
    return Mapping_Mat, Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP


def PXP_PXP_schmidt_decomposition_matrix_efficient(n_PXP,Subspace):
    '''
    Schmidt decomposition matrix tensor of eigenvectors of PXP OBC HAM ONLY NOW!! MORE EFFICIENT METHOD
    :param n_PXP: No. of PXP atoms
    :param Subspace: PXP kinetically constrained subspace
    :return: tensor of dimension [Fib(n_left_cut +3) x Fib(n_right_Cut+3) x Fib(n_PXP +3)] in ordered basis |a,b> of the full permutations of cuts, and eigenvalues!!
    '''
    eval, evec = la.eigh(csr_matrix.todense(PXP_Ham_OBC_Sparse(n_PXP, Subspace)))
    Mapping_Mat, Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP= PXP_Full_Permutation_Basis_To_Reg_Basis_Mapping(n_PXP)
    Tensor = np.zeros((len(Base_left_cut),len(Base_right_cut),len(evec)))
    for i in range(0,len(evec)):
       evec_rearanged= np.dot(Mapping_Mat,evec[:,i])
       Tensor[:, :, i] = np.reshape(evec_rearanged, (len(Base_left_cut), len(Base_right_cut)))
    return Tensor, eval

def PXP_PXP_schmidt_decomposition_matrix_old(n_PXP,Subspace):
    '''
    OLD METHOD- SLOW! Schmidt decomposition matrix tensor of eigenvectors of PXP OBC HAM ONLY NOW!!
    :param n_PXP: No. of PXP atoms
    :param Subspace: PXP kinetically constrained subspace
    :return: tensor of dimension [Fib(n_left_cut +3) x Fib(n_right_Cut+3) x Fib(n_PXP +3)] in ordered basis |a,b> of the full permutations of cuts, and eigenvalues!!
    '''
    eval, evec = la.eigh(csr_matrix.todense(PXP_Ham_OBC_Sparse(n_PXP, Subspace)))
    Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP= PXP_PXP_Full_Permutation_Basis(n_PXP)
    Tensor = np.zeros((len(Base_left_cut),len(Base_right_cut),len(evec)))
    for i in range(0,len(evec)):
        Base_vecs = Base_full_PXP[np.argwhere(np.round(evec,16)[:,i] != 0),:]
        if np.ndim(Base_vecs) == 1:  # for case of only one vector
            Base_vecs = np.expand_dims(Base_vecs, 0)
        Permu_state_array = np.zeros((len(Base_vecs), 2))
        for j in range(0, len(Base_vecs)): # gets indeces of full Permutation basis vectors of one given eigenvector - FOR PXP OBC HAM ONLY NOW!!
            Permu_state_array[j, 0] = np.squeeze(np.nonzero(np.all((Permutation_basis_PXP == Base_vecs[j, :]), axis=1)))
        Permu_state_array[:, 1] = np.squeeze(evec[:,i][(np.argwhere(np.round(evec,16)[:,i] != 0))])
        Permu_array_vector = np.zeros(len(Permutation_basis_PXP))
        Permu_array_vector[Permu_state_array[:, 0].astype(int)] = Permu_state_array[:, 1]
        # print(Permu_state_array)
        # print(Permu_array_vector)
        Tensor[:,:,i]=np.reshape(Permu_array_vector,(len(Base_left_cut),len(Base_right_cut)))
    return Tensor, eval


def Evec_SVD_PXP_PXP(n_PXP): #TOP NUMBER TO CALC IS???
    '''
    singular values of each matrix of eigenvector in the tensor (see PXP_PXP_schmidt_decomposition_matrix) for splitting at PXP - PXP boundary FOR PXP OBC ONLY!!!
    :param n_PXP: No. of PXP atoms
    :param h_c: coupling strength
    :return: singular values of the matrix
    '''
    Tensor, eval = PXP_PXP_schmidt_decomposition_matrix_efficient(n_PXP,Subspace=PXP_Subspace_Algo)
    SVD_num = np.minimum(np.shape(Tensor)[0],np.shape(Tensor)[1])
    SVD_vec_mat = np.zeros((len(eval),SVD_num))
    for i in range(0,len(eval)):
        SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    return SVD_vec_mat ,eval

def Entanglement_entropy_calc_PXP_PXP(n_PXP): #TOP NUMBER TO CALC IS ???
    '''
    calculating entanglement entropy of each eigenstate from singular values -sigma^2ln(sigma) for splitting at PXP - PXP Half chain
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat, eval = Evec_SVD_PXP_PXP(n_PXP)
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    plt.scatter(eval, Entanglement_entropy_vec)
    plt.title('Entanglement vs Energy for {} PXP atoms, pure PXP'.format(n_PXP))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/PXP_PXP_Cut/Entropy_for_PXP_PXP_{}_Atoms_Pure_PXP_Ham.png".format(n_PXP))
    return Entanglement_entropy_vec,eval

def Minimum_EE_eval_PXP_PXP(n_PXP):
    '''
    find energies (and indeces of states) of minimal entropy (L+1 states, where d is the system size)
    :param n_PXP: No. fo PXP atoms
    :param No_of_special_states: the number of anomalous eigenstates (calculated in function)
    :return: plot of lowest values
    '''
    Entanglement_entropy_vec, eval = Entanglement_entropy_calc_PXP_PXP(n_PXP)
    if n_PXP%2 == 1:
        No_of_special_states= n_PXP+1
    else:
        No_of_special_states=n_PXP
    Minimal_EE_ordered_vec=np.sort(Entanglement_entropy_vec)[0:No_of_special_states]
    Minimal_ordered_eval= np.take_along_axis(eval,np.argsort(Entanglement_entropy_vec),axis=0)[0:No_of_special_states]
    print(Minimal_EE_ordered_vec)
    print(eval)
    print(Minimal_ordered_eval)
    plt.scatter(Minimal_ordered_eval,Minimal_EE_ordered_vec)

def Evec_SVD_PXP_PXP_Cluster(n_PXP): #TOP NUMBER TO CALC IS????
    '''
    Cluster Calculation of singular values of each matrix of eigenvector in the tensor (see Eigenvec_Reshape) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: singular values of the matrix, eigenvalues (saved in numpy format)
    '''
    try:
        os.mkdir('EE_PXP_{}_OBC_CUT'.format(n_PXP))
    except:
        pass
    Tensor, eval = PXP_PXP_schmidt_decomposition_matrix_efficient(n_PXP,PXP_Subspace_Algo)
    SVD_num = np.minimum(np.shape(Tensor)[0],np.shape(Tensor)[1])
    SVD_vec_mat = np.empty((len(eval),SVD_num))
    for i in range(0,len(eval)):
         SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    np.save(os.path.join('EE_PXP_{}_OBC_CUT'.format(n_PXP),'SVD.npy'), SVD_vec_mat)
    np.save(os.path.join('EE_PXP_{}_OBC_CUT'.format(n_PXP),'Eval.npy'), eval)
    return

#Evec_SVD_PXP_PXP_Cluster(n_PXP)



######################################################################################################################################################
#                                                   PXP-PXP CUT EE CODE FOR FULL PXP TI COUPLED HAMILTONIAN                                         #
######################################################################################################################################################


def PXP_TI_Full_Permutation_Basis_To_Reg_Basis_Mapping(n_PXP,n_TI):
    '''
    mapping of the Reg PXP basis to the full permutation basis, extended to the kroniker product with TI chain!!!!!
    :param n_PXP: No. of PXP atoms
    :return: mapping (dictionary) of Reg PXP basis states indeces -> new full permutation basis states indeces, and vector of kernel indeces of Permutation states
    '''
    Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP= PXP_PXP_Full_Permutation_Basis(n_PXP)
    Ordering_vec = np.zeros(len(Base_full_PXP))
    for i in range(0,len(Base_full_PXP)):
        Ordering_vec[i]= np.squeeze(np.nonzero(np.all((Permutation_basis_PXP==Base_full_PXP[i,:]),axis=1)))
    # Full_Permu_state_indeces = np.arange(0,len(Permutation_basis_PXP),1)
    # Permutation_kernel=np.squeeze(np.where(np.in1d(Full_Permu_state_indeces,Ordering_vec)==False)) #indeces of permutation basis vectors that go to zero in schmidt matrix!!!
    Ordering_vec_resized = np.repeat(Ordering_vec*(2**n_TI),2**(n_TI))
    for i in range(0,len(Ordering_vec_resized)):
        Ordering_vec_resized[i]=Ordering_vec_resized[i]+i%(2**n_TI)
    Mapping_Dict=np.transpose(np.array([np.arange(0,len(Base_full_PXP)*(2**n_TI),1),Ordering_vec_resized]))
    Mapping_Mat= np.zeros((len(Permutation_basis_PXP)*(2**n_TI),len(Base_full_PXP)*(2**n_TI)))
    Mapping_Mat[Mapping_Dict[:,1].astype(int),Mapping_Dict[:,0].astype(int)]=1
    return Mapping_Mat, Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP

# def PXP_TI_Full_Permutation_Basis_To_Reg_Basis_Mapping_faster(n_PXP,n_TI): #TODO write in matrix form when multiplying the rows and cols
#     '''
#     Faster method of mapping of the Reg PXP basis to the full permutation basis, extended to the kroniker product with TI chain!!!!!
#     :param n_PXP: No. of PXP atoms
#     :return: mapping (dictionary) of Reg PXP basis states indeces -> new full permutation basis states indeces, and vector of kernel indeces of Permutation states
#     '''
#     Mapping_Mat, Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP= PXP_Full_Permutation_Basis_To_Reg_Basis_Mapping(n_PXP)
#     #which peula happens on the transformation matrix????
#     Ordering_vec = np.zeros(len(Base_full_PXP))
#     for i in range(0,len(Base_full_PXP)):
#         Ordering_vec[i]= np.squeeze(np.nonzero(np.all((Permutation_basis_PXP==Base_full_PXP[i,:]),axis=1)))
#     # Full_Permu_state_indeces = np.arange(0,len(Permutation_basis_PXP),1)
#     # Permutation_kernel=np.squeeze(np.where(np.in1d(Full_Permu_state_indeces,Ordering_vec)==False)) #indeces of permutation basis vectors that go to zero in schmidt matrix!!!
#     Ordering_vec_resized = np.repeat(Ordering_vec*(2**n_TI),2**(n_TI))
#     for i in range(0,len(Ordering_vec_resized)):
#         Ordering_vec_resized[i]=Ordering_vec_resized[i]+i%(2**n_TI)
#     print(Ordering_vec_resized)
#     Mapping_Dict=np.transpose(np.array((np.arange(0,len(Base_full_PXP)*(2**n_TI),1),Ordering_vec_resized)))
#     Mapping_Mat= np.zeros((len(Permutation_basis_PXP)*(2**n_TI),len(Base_full_PXP)*(2**n_TI)))
#     Mapping_Mat[Mapping_Dict[:,1].astype(int),Mapping_Dict[:,0].astype(int)]=1
#     return Mapping_Mat, Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP

def PXP_PXP_TI_schmidt_decomposition_matrix_efficient(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2):
    '''
    Schmidt decomposition matrix tensor of eigenvectors of PXP TI coupled Hamiltonian for a cut in middle of PXP chain!
    :param n_PXP: No. of PXP atoms
    :param Subspace: PXP kinetically constrained subspace
    :return: tensor of dimension [Fib(n_left_cut +3) x Fib(n_right_Cut+3)*(2**n_TI) x Fib(n_PXP +3)] in ordered basis |a,b> of the full permutations of cuts, and eigenvalues!!
    '''
    eval, evec = la.eigh(csr_matrix.todense(PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m)))
    Mapping_Mat, Base_full_PXP, Base_left_cut, Base_right_cut, Permutation_basis_PXP= PXP_TI_Full_Permutation_Basis_To_Reg_Basis_Mapping(n_PXP,n_TI)
    Tensor = np.zeros((len(Base_left_cut),len(Base_right_cut)*(2**n_TI),len(evec)))
    for i in range(0,len(eval)):
       evec_rearanged= np.dot(Mapping_Mat,evec[:,i])
       Tensor[:, :, i] = np.reshape(evec_rearanged, (len(Base_left_cut), len(Base_right_cut)*(2**n_TI)))
    return Tensor, eval


def Evec_SVD_PXP_PXP_TI(n_PXP,n_TI, h_c): #TOP NUMBER TO CALC IS???
    '''
    singular values of each matrix of eigenvector in the tensor  for splitting at PXP - PXP boundary for PXP TI FULL HAMILTONIAN
    :param n_PXP: No. of PXP atoms
    :param h_c: coupling strength
    :return: singular values of the matrix
    '''
    Tensor, eval = PXP_PXP_TI_schmidt_decomposition_matrix_efficient(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(np.shape(Tensor)[0],np.shape(Tensor)[1])
    SVD_vec_mat = np.zeros((len(eval),SVD_num))
    for i in range(0,len(eval)):
        SVD_vec_mat[i,:]= scla.svdvals((Tensor[:,:,i]))
    return SVD_vec_mat ,eval

def Entanglement_entropy_calc_PXP_PXP_TI(n_PXP,n_TI, h_c): #TOP NUMBER TO CALC IS ???
    '''
    calculating entanglement entropy of each eigenstate from singular values -sigma^2ln(sigma) for splitting at PXP - PXP Half chain for full PXP TI coupled
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat, eval = Evec_SVD_PXP_PXP_TI(n_PXP,n_TI, np.round(h_c,5))
    #Entanglement_entropy_normalization_check= np.sum(SVD_vec_mat**2,axis=1)
    #print(np.nonzero(np.round(Entanglement_entropy_normalization_check,5)!=1))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    #print('Entanglement_entropy_vec', Entanglement_entropy_vec,'eval',eval)
    plt.scatter(eval, np.round(Entanglement_entropy_vec,32)) #TODO check if doesn't screw anything
    plt.title('Entanglement vs Energy for {} PXP {} TI, {} $h_c$ PXP-PXP cut'.format(n_PXP,n_TI, np.round(h_c,5)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/PXP_PXP_TI_Cut/Entropy_for_PXP_PXP_{}_TI_{}_h_c_{}.png".format(n_PXP,n_TI, np.round(h_c,5)))
    return eval, Entanglement_entropy_vec

def Entanglement_Entropy_unique_check(n_PXP,n_TI, h_c):
    '''
    Checking Entanglement entropy values that return less than the multiplicity of 2*2**n_TI, and corresponding energies (are they the 0?)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: number of "unique" entanglement entropy values, plot with the "unique" values marked in dif color
    '''
    eval, Entanglement_entropy_vec = Entanglement_entropy_calc_PXP_PXP_TI(n_PXP,n_TI, h_c)
    Entanglement_Values, Repetitions = np.unique(np.round(Entanglement_entropy_vec,14),return_counts=True)
    print(Entanglement_Values, Repetitions)
    Where_uniques_npunique=np.squeeze(np.argwhere(Repetitions!=2**(n_TI+1)))
    Where_nonuniques=np.squeeze(np.argwhere(Repetitions==2**(n_TI+1)))
    Who_uniques= Entanglement_Values[Where_uniques_npunique]
    print('number of "unique" evals:', np.sum(Repetitions[Where_uniques_npunique])) #counts the number of data dots that are not part of the 2*2**n_TI multiplicity
   #print('number of NON-"unique" evals (WITHOUT MULTIPLICITY!!):', len(Where_nonuniques))
    Where_uniques=np.nonzero(np.isin(np.round(Entanglement_entropy_vec,14),Who_uniques)) #indeces of "unique" entanglement entropy
    Unique_evals=eval[Where_uniques]
    print('Unique Energies for {} PXP atoms'.format(n_PXP),np.unique(np.round(Unique_evals,8)))
    #print('Unique Entropies of these energies:')
    #print(np.transpose(np.array((Unique_evals,Entanglement_entropy_vec[Where_uniques]))))
    plt.scatter(np.round(Unique_evals,14), Entanglement_entropy_vec[Where_uniques])
    plt.show()

def TI_PXP_eval_check(n_PXP,n_TI,J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi)):
    '''
    finds eigenvalues of TI and PXP separately and PXP+TI combined eigenvalues (for h_c=0!!) to try explain degeneracies
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param J: Ising strength
    :param h_x: longtitudinal strength
    :param h_z: transverse strengh
    :return: Eigenvalues of PXP, TI and all combinations of additions of them (in the dimension of PXP kron TI!)
    '''
    eval, evec=la.eigh(csr_matrix.todense(PXP_Ham_OBC_Sparse(n_PXP, PXP_Subspace_Algo)))
    eval=np.round(eval,14)
    #print(eval)
    eval2, evec2 = la.eigh(csr_matrix.todense(TIOBCNew_Sparse(n_TI, J, h_x, h_z)))
    print(eval2)
    extended_dim_eval=np.kron(eval,np.ones((len(eval2))))
    extended_dim_repeated_eval2=np.tile(eval2,len(eval))
    #print(np.allclose(len(extended_dim_eval),len(extended_dim_repeated_eval2)))
    PXP_TI_combined_ev= extended_dim_eval+extended_dim_repeated_eval2  # all options of combined energies for h_c=0
    PXP_TI_combined_ev_sorted=np.sort(PXP_TI_combined_ev)

def Evec_SVD_PXP_PXP_TI_Cluster(n_PXP,n_TI,h_c): #TOP NUMBER TO CALC IS????
    '''
    Cluster Calculation of singular values of each matrix of eigenvector in the tensor (see Eigenvec_Reshape) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: singular values of the matrix, eigenvalues (saved in numpy format)
    '''
    try:
        os.mkdir('EE_PXP_PXPTI_CUT_PXP_{}'.format(n_PXP))
    except:
        pass
    Tensor, eval =  PXP_PXP_TI_schmidt_decomposition_matrix_efficient(n_PXP, n_TI, np.round(h_c,5), J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(np.shape(Tensor)[0],np.shape(Tensor)[1])
    SVD_vec_mat = np.empty((len(eval),SVD_num))
    for i in range(0,len(eval)):
         SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    np.save(os.path.join('EE_PXP_PXPTI_CUT_PXP_{}'.format(n_PXP),'TI_{}_h_c_{}_SVD.npy'.format(n_TI,np.round(h_c,5))), SVD_vec_mat)
    np.save(os.path.join('EE_PXP_PXPTI_CUT_PXP_{}'.format(n_PXP),'TI_{}_h_c_{}_Eval.npy'.format(n_TI,np.round(h_c,5))), eval)
    return

#Evec_SVD_PXP_PXP_TI_Cluster(n_PXP,n_TI,h_c)


def Neel_Overlap_calc_Pure_PXP_plt(n_PXP):
    '''
    Calculates different overlaps of pure PXP OBC Hamiltonian eigenstates with the Neel state
    :param n_PXP: No. of PXP atoms
    :return: plot of log10 of overlap vs energies
    '''
    eval, evec = la.eigh((PXP_Ham_OBC_Sparse(n_PXP, PXP_Subspace_Algo).todense()))
    Neel_state= Neel_Subspace_Basis(n_PXP)
    overlap_vec = np.squeeze(np.absolute(np.matmul(np.transpose(evec),Neel_state).round(35)))**2
    plt.scatter(eval,np.log10(overlap_vec))
    plt.xlabel('$E$')
    plt.ylabel(r'$log_{10}(|\langle\mathbb{Z}_{2}|\psi\rangle|)^{2}$')
    plt.title('Log of Neel State Eigenstates Overlap vs. Energies for {} PXP atoms'.format(n_PXP))
    # plt.savefig('Overlap_{}_PXP_atoms.pdf'.format(n_PXP))
    return plt.show()

def Neel_Overlap_calc_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp = 0, m=2):
    '''
    Calculates different overlaps of PXP-TI Hamiltonian eigenstates with the Neel state (Neel matrix x identity)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: log10 of overlap vector and evals (ordered same manner)
    '''
    eval, evec = la.eigh(PXP_TI_coupled_Sparse(n_PXP, n_TI,J ,h_x ,h_z ,h_c ,h_imp ,m).todense())
    Neel_state_coupled_mat= np.kron(np.outer(Neel_Subspace_Basis(n_PXP),Neel_Subspace_Basis(n_PXP)),np.identity(2**n_TI))
    overlap_vec_Neel_outer= np.diag(np.matmul(np.conjugate(np.transpose(evec)),np.matmul(Neel_state_coupled_mat,evec)))
    return eval, np.log10(overlap_vec_Neel_outer.round(32))

def Neel_Overlap_calc_PXP_TI_plt(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp = 0, m=2):
    '''
    Calculates different overlaps of PXP-TI Hamiltonian eigenstates with the Neel state (Neel matrix x identity)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: evals, log of overlap of each eigenvector with Neel state
    '''
    eval, evec = la.eigh(PXP_TI_coupled_Sparse(n_PXP, n_TI,J ,h_x ,h_z ,h_c ,h_imp ,m).todense())
    Neel_state_coupled_mat= np.kron(np.outer(Neel_Subspace_Basis(n_PXP),Neel_Subspace_Basis(n_PXP)),np.identity(2**n_TI))
    overlap_vec_Neel_outer= np.diag(np.matmul(np.conjugate(np.transpose(evec)),np.matmul(Neel_state_coupled_mat,evec)))
    plt.scatter(eval,np.log10(overlap_vec_Neel_outer.round(32)))
    plt.xlabel('Energy')
    plt.ylabel(r'$log_{10}(|\langle\mathbb{Z}_{2}|\psi\rangle|)^{2}$')
    plt.title('Overlap of Neel State with Eigenstates vs. Eigenstate Energy')
    #plt.savefig('Overlap_{}_PXP_atoms.pdf'.format(n_PXP))
    return plt.show()

def Neel_Max_Overlap(n_PXP, n_TI, h_c): #TODO need to fix #will work if the 2 highest overlap values have a multiplicity of 1 only
    '''
    finds 3/4 x 2^n_TI max Overlapped eigenstates with the Neel state
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :anomalous_eigenstates: number is n_PXP + 1
    :return: eigenenergies of max overlap (ordered in descending order), OPTION FOR INDECES
    '''
    eval, overlap_vec = Neel_Overlap_calc_PXP_TI(n_PXP, n_TI, h_c)
    if n_PXP%2==0:
        overlap_max_vals=np.flip(np.sort(overlap_vec))[(1*(2**n_TI)):4*(2**n_TI)] #take first 4 (*2**n_TI) anomalous eigenvalues, EXCLUDING 0
        print(overlap_max_vals)
    else:
        overlap_max_vals=np.flip(np.sort(overlap_vec))[:4*(2**n_TI)] #take first 4 (*2**n_TI) anomalous eigenvalues
        #print(overlap_max_vals)
    overlap_max_indeces=np.squeeze(np.nonzero(np.isin(overlap_vec,overlap_max_vals)))
    max_overlap_evals=eval[overlap_max_indeces]
    overlap_max_vals_eval_ordered= overlap_vec[overlap_max_indeces]
    return max_overlap_evals, overlap_max_vals_eval_ordered

def Neel_Max_Overlap_plt(n_PXP, n_TI, h_c): #TODO need to fix #will work if the 2 highest overlap values have a multiplicity of 1 only
    '''
    finds 3/4 x 2^n_TI max Overlapped eigenstates with the Neel state
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :anomalous_eigenstates: n_PXP + 1
    :return: eigenenergies of max overlap
    '''
    eval, overlap_vec = Neel_Overlap_calc_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    if n_PXP%2==0:
        overlap_max_vals=np.flip(np.sort(overlap_vec))[(1*(2**n_TI)):4*(2**n_TI)] #take first 4 (*2**n_TI) anomalous eigenvalues, EXCLUDING 0
        #print(overlap_max_vals)
    else:
        overlap_max_vals=np.flip(np.sort(overlap_vec))[:4*(2**n_TI)] #take first 4 (*2**n_TI) anomalous eigenvalues
        #print(overlap_max_vals)
    overlap_max_indeces=np.squeeze(np.nonzero(np.isin(overlap_vec,overlap_max_vals)))
    max_overlap_evals=eval[overlap_max_indeces]
    #print((max_overlap_evals))
    plt.scatter(max_overlap_evals,overlap_vec[overlap_max_indeces])
    return plt.show()



def Max_Overlap_EE_plt(n_PXP, n_TI, h_c): #TDOO does not work
    '''
    plot the chosen max overlap eigenvectors' entanglement entropy (vs energy)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: plot of EE with chosen max overlap eigenvectors in different color
    '''
    overlap_max_indeces = Neel_Max_Overlap(n_PXP, n_TI, h_c)
    eval, EE_vec = Entanglement_entropy_calc_PXP_PXP_TI(n_PXP, n_TI, h_c)
    plt.scatter(eval[overlap_max_indeces],EE_vec[overlap_max_indeces])
    plt.show()










