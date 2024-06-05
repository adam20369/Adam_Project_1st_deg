import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PXP_Entry_By_Entry import *
from PXP_E_B_E_Sparse import *
from O_z_Oscillations import *
import numpy.linalg as la
import scipy.linalg as scla
from time import time
from scipy import integrate
from scipy.optimize import root_scalar
from scipy.optimize import RootResults
from sympy.utilities.iterables import multiset_permutations
import scipy.sparse as sp
from scipy.special import comb

def Neel_Neel_state_Sparse_Time_prop(n_PXP, n_TI,h_c, Initialstate, T_start, T_max, T_step,J, h_x, h_z, h_imp, m):
    '''
    Returns matrix of |Neel(t)>, each col is for a different time
    :param n_PXP: No. of PXP atoms
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: matrix of vectors |Neel(t)>, each Col for a different time
    '''
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse(n_PXP,n_TI,J,h_x,h_z,h_c,h_imp,m),Initialstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    return Propagated_ket_fin.round(4).astype('complex')

def Neel_Time_Reshape_PXP_TI(n_PXP, n_TI, h_c, T_start, T_max, T_step,J, h_x, h_z, h_imp, m):
    '''
    Reshapes each eigenvector to the shape of a matrix for schmidt decomposition (splitting at PXP - TI boundary)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param Eigenvec: eigenvectors (as cols of a matrix)
    :return: rank 3 tensor of matrices of each eigenstate's decomposition to joint basis
    '''
    Initialstate= PXP_TI_Neelstate(n_PXP,n_TI)
    Propagated_ket=Neel_Neel_state_Sparse_Time_prop(n_PXP, n_TI,h_c, Initialstate, T_start, T_max, T_step,J, h_x, h_z, h_imp, m)
    Dim=np.shape(Propagated_ket)[1]
    Tensor = np.zeros((Subspace_basis_count_faster(n_PXP), 2**(n_TI), Dim)).astype('complex')
    for i in range(0, Dim):
        Tensor[:,:,i]=np.reshape(Propagated_ket[:,i],(Subspace_basis_count_faster(n_PXP),2**(n_TI)))
    #np.save(os.path.join('EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Evec_h_c_{}_compare.npy'.format(h_c)), evec)
    #np.save(os.path.join('EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Eval_h_c_{}_compare.npy'.format(h_c)), eval)
    return Tensor

def Neel_Time_SVD_PXP_TI(n_PXP, n_TI, h_c,T_start, T_max, T_step): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    singular values of each matrix of Neelstate at different time steps for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Matrix with Rows!!! as singular values of time step
    '''
    Tensor = Neel_Time_Reshape_PXP_TI(n_PXP, n_TI, h_c, T_start, T_max, T_step,J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(2**(n_TI),Subspace_basis_count_faster(n_PXP))
    SVD_vec_mat = np.zeros((T_step,SVD_num))
    Time=np.linspace(T_start,T_max,T_step)
    for i in range(0,T_step):
        SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    return SVD_vec_mat, Time

def Neel_Time_Entanglement_Entropy_PXP_TI_plt(n_PXP, n_TI, h_c,T_start, T_max, T_step): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    calculating entanglement entropy of Neelstate for each time step from singular values -sigma^2ln(sigma) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs time for every timestep
    '''
    SVD_vec_mat, Time = Neel_Time_SVD_PXP_TI(n_PXP, n_TI, np.round(h_c,5),T_start, T_max, T_step)
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    plt.plot(Time, np.round(Entanglement_entropy_vec,32))
    plt.title('Entanglement vs time of Neelstate for {} PXP & {} TI atoms, $h_c$ {} ZZ coupl '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Time')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return plt.show()

def Neel_Neel_state_True_X_i_Sparse_Time_prop(n_PXP, n_TI,h_c, Initialstate, T_start, T_max, T_step,J, h_x, h_z, h_imp, m):
    '''
    Returns matrix of |Neel(t)>, each col is for a different time, XX coupling!!
    :param n_PXP: No. of PXP atoms
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: matrix of vectors |Neel(t)>, each Col for a different time XX coupling!
    '''
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse_Xi(n_PXP,n_TI,J,h_x,h_z,h_c,h_imp,m),Initialstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    return Propagated_ket_fin.round(4).astype('complex')

def Neel_Time_True_X_i_Reshape_PXP_TI(n_PXP, n_TI, h_c, T_start, T_max, T_step,J, h_x, h_z, h_imp, m):
    '''
    Reshapes each eigenvector to the shape of a matrix for schmidt decomposition (splitting at PXP - TI boundary) XX coupling!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param Eigenvec: eigenvectors (as cols of a matrix)
    :return: rank 3 tensor of matrices of each eigenstate's decomposition to joint basis
    '''
    Initialstate= PXP_TI_Neelstate_True_X_i(n_PXP,n_TI)
    Propagated_ket=Neel_Neel_state_True_X_i_Sparse_Time_prop(n_PXP, n_TI,h_c, Initialstate, T_start, T_max, T_step,J, h_x, h_z, h_imp, m)
    Dim=np.shape(Propagated_ket)[1]
    Tensor = np.zeros((Extended_X_i_Subspace_basis_count_faster(n_PXP), 2**(n_TI), Dim)).astype('complex')
    for i in range(0, Dim):
        Tensor[:,:,i]=np.reshape(Propagated_ket[:,i],(Extended_X_i_Subspace_basis_count_faster(n_PXP),2**(n_TI)))
    #np.save(os.path.join('EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Evec_h_c_{}_compare.npy'.format(h_c)), evec)
    #np.save(os.path.join('EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Eval_h_c_{}_compare.npy'.format(h_c)), eval)
    return Tensor
def Neel_Time_True_X_i_SVD_PXP_TI(n_PXP, n_TI, h_c,T_start, T_max, T_step): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    singular values of each matrix of Neelstate at different time steps for splitting at PXP - TI boundary XX coupling!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Matrix with Rows!!! as singular values of time step XX coupling!!
    '''
    Tensor = Neel_Time_True_X_i_Reshape_PXP_TI(n_PXP, n_TI, h_c, T_start, T_max, T_step,J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp=0, m=2)
    SVD_num = np.minimum(2**(n_TI),Extended_X_i_Subspace_basis_count_faster(n_PXP))
    SVD_vec_mat = np.zeros((T_step,SVD_num))
    Time=np.linspace(T_start,T_max,T_step)
    for i in range(0,T_step):
        SVD_vec_mat[i,:]= scla.svdvals(Tensor[:,:,i])
    return SVD_vec_mat, Time

def Neel_Time_Entanglement_Entropy_True_X_i_PXP_TI_plt(n_PXP, n_TI, h_c,T_start, T_max, T_step): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    ''' XX coupling!
    calculating entanglement entropy of Neelstate for each time step from singular values -sigma^2ln(sigma) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs time for every timestep
    '''
    SVD_vec_mat, Time = Neel_Time_True_X_i_SVD_PXP_TI(n_PXP, n_TI, np.round(h_c,5),T_start, T_max, T_step)
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    plt.plot(Time, np.round(Entanglement_entropy_vec,32))
    plt.title('Entanglement vs time of Neelstate for {} PXP & {} TI atoms, $h_c$ {} XX coupl '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Time')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return plt.show()


def Average_Energy_Neel_PXP_Time(n_PXP,n_TI,h_c,T_start,T_max,T_step, J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi),h_imp=0,m=2):
    ''' finding Neelstate energy of PXP for time 0 to time T_max
    :param T_start: start time
    :param T_max: max time
    :param T_step: number of steps
    :return: array of the PXP energy as a function of time for every step of time
    '''
    Neelstate=PXP_TI_Neelstate(n_PXP,n_TI)
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse(n_PXP,n_TI,J,h_x,h_z,h_c,h_imp,m),Neelstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket.astype('complex'))
    Propagated_bra_fin= np.conjugate(Propagated_ket.astype('complex'))
    Average_Energy_Neel=Propagated_bra_fin@sp.kron(PXP_Ham_OBC_Sparse(n_PXP,PXP_Subspace_Algo),sp.eye(2**n_TI))@Propagated_ket_fin
    Energy_Time_array=np.diag(Average_Energy_Neel)
    return Energy_Time_array.astype('float')

# def Average_Energy_Neel_PXP_True_X_i(n_PXP):
#     ''' finding Neelstate energy of PXP
#     :return: scalar, Neelstate energy of PXP is 0
#     '''
#     Neelstate=Neel_X_i_Extended_Subspace_Basis(n_PXP)
#     Average_Energy_Neel=Neelstate@PXP_Ham_OBC_Sparse_True_X_i(n_PXP,PXP_Subspace_Algo_extended_X_i)@Neelstate
#     return Average_Energy_Neel

def Average_Energy_Neel_TI_Time(n_PXP,n_TI,h_c,T_start,T_max,T_step,J=1,h_x=np.sin(0.485*np.pi),h_z=np.cos(0.485*np.pi),h_imp=0,m=2):
    ''' finding Neelstate energy of TI for time 0 to time T_max
    :param T_start: start time
    :param T_max: max time
    :param T_step: number of steps
    :return: array of the TI energy as a function of time for every step of time (1-N even and 1+h_z-N for odd for time T=0)
    '''
    Neelstate=PXP_TI_Neelstate(n_PXP,n_TI)
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse(n_PXP,n_TI,J,h_x,h_z,h_c,h_imp,m),Neelstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket.astype('complex'))
    Propagated_bra_fin= np.conjugate(Propagated_ket.astype('complex'))
    Average_Energy_Neel=Propagated_bra_fin@sp.kron(sp.eye(Subspace_basis_count_faster(n_PXP)),TIOBCNewImpure_sparse(n_TI,J,h_x,h_z,h_imp,m))@Propagated_ket_fin
    Energy_Time_array=np.diag(Average_Energy_Neel)
    return Energy_Time_array.astype('float')

def Temperature_Equation_PXP_Time(beta, n_PXP,n_TI,h_c,Time,T_start,T_max,T_step):
    '''
    Temperature equation and derivative for PXP chain for a specific time (minus the PXP energy of that specific time)
    :return: function of temperature equation (time dependent), derivative of function
    '''
    eval, evec = la.eigh(PXP_Ham_OBC_Sparse(n_PXP,PXP_Subspace_Algo).todense())
    Average_Energy_PXP_time= Average_Energy_Neel_PXP_Time(n_PXP,n_TI,h_c,T_start,T_max,T_step, J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi),h_imp=0,m=2)[int((Time/T_max)*T_step)]
    Temperature_Equation_array= np.sum(eval*np.exp(-beta*eval)/(np.sum(np.exp(-beta*eval))))-Average_Energy_PXP_time
    Temperature_Derivative_Equation= np.sum(-(eval**2)*np.exp(-beta*eval))/(np.sum(np.exp(-beta*eval))) + ((np.sum(eval*np.exp(-beta*eval))**2)/(np.sum(np.exp(-beta*eval)))**2)
    return Temperature_Equation_array, Temperature_Derivative_Equation

def Temperature_Equation_TI_Time(beta, n_PXP,n_TI,h_c,Time, T_start,T_max,T_step):
    '''
    Temperature equation and derivative for TI chain for a specific time (minus the TI energy of that specific time)
    :return: function of temperature equation (time dependent), derivative of function
    '''
    eval, evec = la.eigh(TIOBCNewImpure_sparse(n_TI,J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi),h_imp=0,m=2).todense())
    Average_Energy_TI_time= Average_Energy_Neel_TI_Time(n_PXP,n_TI,h_c,T_start,T_max,T_step, J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi),h_imp=0,m=2)[int((Time/T_max)*T_step)]
    Temperature_Equation= np.sum(eval*np.exp(-beta*eval)/(np.sum(np.exp(-beta*eval))))-Average_Energy_TI_time
    Temperature_Derivative_Equation= np.sum(-(eval**2)*np.exp(-beta*eval))/(np.sum(np.exp(-beta*eval))) + ((np.sum(eval*np.exp(-beta*eval))**2)/(np.sum(np.exp(-beta*eval)))**2)
    return Temperature_Equation, Temperature_Derivative_Equation

def eigenvalue_sum_check(n_PXP, n_TI, h_c, J=1,h_x=np.sin(0.485*np.pi),h_z=np.cos(0.485*np.pi)):
    eval, evec = la.eigh((PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp=0, m=2)).todense())
    return np.sum(eval)

def Beta_Calc_PXP(n_PXP,n_TI,h_c,Time,T_start,T_max,T_step,initial_guess_1,initial_guess_2):
    '''
    :return: solutions for beta
    '''
    Iteration_Solver=root_scalar(Temperature_Equation_PXP_Time,args=(n_PXP,n_TI,h_c,Time,T_start,T_max,T_step),fprime=True,method='newton',x0=initial_guess_1,x1=initial_guess_2)
    return Iteration_Solver.root

def Beta_Calc_TI(n_PXP,n_TI,h_c,Time,T_start,T_max,T_step,initial_guess_1,initial_guess_2):
    '''
    :return: solutions for beta
    '''
    Iteration_Solver=root_scalar(Temperature_Equation_TI_Time,args=(n_PXP,n_TI,h_c,Time,T_start,T_max,T_step),fprime=True,method='newton',x0=initial_guess_1,x1=initial_guess_2)
    return Iteration_Solver.root

def Beta_PXP_Time_plt(n_PXP,n_TI,h_c,Max_Time,T_start,T_max,T_step,initial_guess_1,initial_guess_2):
    '''

    :param n_PXP:
    :param n_TI:
    :param h_c:
    :param Max_Time:
    :param T_start:
    :param T_max:
    :param T_step:
    :param initial_guess_1:
    :param initial_guess_2:
    :return:
    '''
    Time= np.arange(0,Max_Time,1)
    Beta_array_time=np.zeros(len(Time))
    for T in Time:
        Beta_array_time[T]=Beta_Calc_PXP(n_PXP,n_TI,h_c,T,T_start,T_max,T_step,initial_guess_1,initial_guess_2)
    plt.plot(Time,Beta_array_time)
    return plt.show()

def Beta_TI_Time_plt(n_PXP,n_TI,h_c,Max_Time,T_start,T_max,T_step,initial_guess_1,initial_guess_2):
    '''

    :param n_PXP:
    :param n_TI:
    :param h_c:
    :param Max_Time:
    :param T_start:
    :param T_max:
    :param T_step:
    :param initial_guess_1:
    :param initial_guess_2:
    :return:
    '''
    Time= np.arange(0,Max_Time,1)
    Beta_array_time=np.zeros(len(Time))
    for T in Time:
        Beta_array_time[T]=Beta_Calc_TI(n_PXP,n_TI,h_c,T,T_start,T_max,T_step,initial_guess_1,initial_guess_2)
    plt.plot(Time,Beta_array_time)
    return plt.show()