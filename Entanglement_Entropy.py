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


def Evec_Reshape_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2):
    '''
    Reshapes each eigenvector to the shape of a matrix for schmidt decomposition (splitting at PXP - TI boundary)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param Eigenvec: eigenvectors (as cols of a matrix)
    :return: rank 3 tensor of matrices of each eigenstate's decomposition to joint basis
    '''
    eval, evec = la.eigh(csr_matrix.todense(PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m)))
    Ham_Dim = len(eval) #also number of evecs
    Tensor = np.empty((Subspace_basis_count_faster(n_PXP),2**(n_TI),Ham_Dim))
    for i in range(0,Ham_Dim):
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
    SVD_vec_mat = np.empty((len(eval),SVD_num))
    for i in range(0,len(eval)):
        SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    return SVD_vec_mat ,eval

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
    np.save(os.path.join('EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Entanglement_h_c_{}'.format(h_c)), SVD_vec_mat)
    np.save(os.path.join('EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Eval_h_c_{}'.format(h_c)), eval)
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
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.log(SVD_vec_mat),axis=1)
    plt.scatter(eval, Entanglement_entropy_vec)
    plt.title('Entanglement vs Energy for {} PXP & {} TI atoms, $h_c$ {} '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return

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
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.log(SVD_vec_mat),axis=1)
    eval_interval_arg_min= np.min(np.argwhere(eval>(h_c-interval_width)))
    eval_interval_arg_max= np.min(np.argwhere(eval>(h_c+interval_width)))
    average = np.average(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    std = np.std(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    return np.round(average,4), np.round(std,4)

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
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.log(SVD_vec_mat),axis=1)
    eval_interval_arg_min= np.min(np.argwhere(eval>(h_c-interval_width)))
    eval_interval_arg_max= np.min(np.argwhere(eval>(h_c+interval_width)))
    average = np.average(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    std = np.std(Entanglement_entropy_vec[eval_interval_arg_min:eval_interval_arg_max])
    try:
        os.mkdir('EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP,n_TI,interval_width))
    except:
        pass
    np.save(os.path.join('EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP,n_TI,interval_width),'Average_h_c_{}_width_{}'.format(h_c,interval_width)), average)
    np.save(os.path.join('EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP,n_TI,interval_width),'STD_h_c_{}_width_{}'.format(h_c,interval_width)), std)
    return
#Entanglement_entropy_avg_std_Cluster(n_PXP, n_TI, h_c, interval_width=1)


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
    plt.savefig('Figures/Entanglement_Entropy/Average_Entanglement_Entropy')
    plt.show()


def EE_Std_plot(n_PXP, n_TI,h_c_start, h_c_max, interval_width):
    '''
    Plots Standard deviation of entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of Standard deviation vs h_c
    '''
    h_c= np.arange(h_c_start,h_c_max+0.1,0.1)
    avg= np.empty((np.size(h_c)))
    std= avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i,1)*10-h_c_start*10)], std[int(np.round(i,1)*10-h_c_start*10)] = Entanglement_entropy_avg_std(n_PXP, n_TI, i, interval_width)
    plt.plot(h_c,std, color='r')
    plt.title('Entanglement STD vs coupling strength $h_c$ for {} PXP & {} TI atoms'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Standard Deviation of Entanglement Entropy')
    plt.savefig('Figures/Entanglement_Entropy/Standard_Deviation_of_Entanglement_Entropy')
    plt.show()

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

def EE_Avg_Std_plot_Cluster(n_PXP, n_TI,h_c_start, h_c_max,interval_width):
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



#####################################################################################################################################
#                                                   PXP-PXP EE CODE                                                                 #
#####################################################################################################################################



def Subspace_complement(n_PXP,i):
    '''
    Calculates number of complement states of some PXP reduced state after breaking a PXP chain in the middle (for odd numbers- the A subsystem contains an EVEN number of states)
    :param n_PXP: number of PXP atoms
    :return: number of states for a corresponding index of PXP state
    '''

def Evec_Reshape_PXP_PXP(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2):
    '''
    Reshapes each eigenvector to the shape of a matrix for schmidt decomposition - for splitting PXP at middle of chain!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param Eigenvec: eigenvectors (as cols of a matrix)
    :return: rank 3 tensor of matrices of each eigenstate's decomposition to joint basis
    '''
    eval, evec = la.eigh(csr_matrix.todense(PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m)))
    Ham_Dim = len(eval) #also number of evecs
    Tensor = np.empty((Subspace_basis_count_faster(int(n_PXP/2)),int(np.rint(Subspace_basis_count_faster(n_PXP)/Subspace_basis_count_faster(int(n_PXP/2))))*2**(n_TI),Ham_Dim))
    for i in range(0,Ham_Dim):
         Tensor[:,:,i]=np.reshape(evec[:,i],(Subspace_basis_count_faster(int(n_PXP/2)),int(np.rint(Subspace_basis_count_faster(n_PXP)/Subspace_basis_count_faster(int(n_PXP/2))))*2**(n_TI))) #TODO check this
    return Tensor, eval

def Evec_SVD_PXP_PXP(n_PXP, n_TI, h_c):
    '''
    singular values of each matrix of eigenvector in the tensor (see Eigenvec_Reshape) for splitting PXP at middle of chain
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: singular values of the matrix
    '''
    Tensor, eval = Evec_Reshape_PXP_PXP(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(Subspace_basis_count_faster(int(n_PXP/2)),int(np.rint(Subspace_basis_count_faster(n_PXP)/Subspace_basis_count_faster(int(n_PXP/2))))*2**(n_TI))
    SVD_vec_mat = np.empty((len(eval),SVD_num))
    for i in range(0,len(eval)):
        SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    return SVD_vec_mat ,eval

def Entanglement_entropy_calc_PXP_PXP(n_PXP, n_TI, h_c):
    '''
    calculating entanglement entropy of each eigenstate from singular values -sigma^2ln(sigma) for splitting PXP at middle of chain
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat, eval = Evec_SVD_PXP_PXP(n_PXP, n_TI, np.round(h_c,5))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.log(SVD_vec_mat),axis=1)
    plt.scatter(eval, Entanglement_entropy_vec)
    plt.title('Entanglement vs Energy for {} PXP & {} TI atoms, $h_c$ {} '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return


