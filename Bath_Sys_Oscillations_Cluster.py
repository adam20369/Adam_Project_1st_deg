import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import PXP_E_B_E_Sparse as Ebe
from PXP_Entry_By_Entry import *
import numpy.linalg as la
from scipy.linalg import expm
from time import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Coupling_To_Bath import *
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
from scipy.optimize import curve_fit
from Cluster_Sparse_Osc_Para import *

def O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <Neel|O_z(t)|Neel> values and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param J: TI ising term strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param h_x: longtitudinal term strength (TI)
    :param h_z: transverse term strength
    :param h_c: coupling term strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param h_imp: impurity (TI) strength
    :param m: impurity site
    :return: vector -  <NeelxHaar|O_z(t)|NeelxHaar>
    '''
    O_z_PXP = Ebe.O_z_PXP_Entry_Sparse(n_PXP, PXP_Subspace_Algo)
    O_z_Full = sp.kron(O_z_PXP,sp.eye(2**n_TI))
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),Initialstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ O_z_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_O_z_Sparse_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param Initialstate: NeelHaar state usually
    :param J: Ising term strength
    :param h_x: longtitudinal field strength
    :param h_z: Traverse field strength
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    try:
        os.mkdir('PXP_{}_TI_{}_Osc'.format(n_PXP, n_TI))
    except:
        pass
    if os.path.isfile('PXP_{}_TI_{}_Osc/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP,n_TI,n_PXP,n_TI,h_c,T_max,T_step)) == False:
        Initialstate = Neel_EBE_Haar(n_PXP, n_TI)
        J = 1
        h_x = np.sin(0.485 * np.pi)
        h_z = np.cos(0.485 * np.pi)
        Sandwich= O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)
        np.save(os.path.join('PXP_{}_TI_{}_Osc'.format(n_PXP, n_TI),'Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP,n_TI,h_c,T_max,T_step)),Sandwich)
Run_O_z_Sparse_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step)

def PXP_TI_Neelstate(n_PXP,n_TI):
    '''
    Combined Neel state of PXP and TI!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :return: kronekered Neel state
    '''
    TI_Neel= TI_Neelstate(n_TI)
    PXP_Neel= Neel_Subspace_Basis(n_PXP)
    PXP_TI_Neel = np.kron(PXP_Neel,TI_Neel)
    return PXP_TI_Neel

def Z_i_TI_only_time_prop(n_TI, Initialstate, J, h_x, h_z, T_start, T_max, T_step,i_index):
    '''
    Returns <Neel_bath|Z_i(t)|Neel_bath> values FOR TI MODEL ONLY (uncoupled to anything) and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param J: TI ising term strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param h_x: longtitudinal term strength (TI)
    :param h_z: transverse term strength
    :param h_c: coupling term strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param h_imp: impurity (TI) strength
    :param m: impurity site
    :return: vector -  <NeelxHaar|O_z(t)|NeelxHaar>
    '''
    Z_i = Ebe.Z_i_Spin_Basis_sparse(n_TI, i_index)
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.TIOBCNew_Sparse(n_TI, J, h_x, h_z),Initialstate,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ Z_i @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_Z_i_TI_only_Sparse_Time_prop(n_TI,i_index,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation FOR TI MODEL ONLY (uncoupled to anything) plotter in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param Initialstate: NeelHaar state usually
    :param J: Ising term strength
    :param h_x: longtitudinal field strength
    :param h_z: Traverse field strength
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = TI_Neelstate(n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return Z_i_TI_only_time_prop(n_TI, Initialstate, J, h_x, h_z, T_start, T_max, T_step,i_index)


def Z_i_Bath_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index, h_imp=0, m=2):
    '''
    Returns <Neel_bath|Z_i(t)|Neel_bath> values and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param J: TI ising term strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param h_x: longtitudinal term strength (TI)
    :param h_z: transverse term strength
    :param h_c: coupling term strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param h_imp: impurity (TI) strength
    :param m: impurity site
    :return: vector -  <NeelxHaar|O_z(t)|NeelxHaar>
    '''
    Z_i = Ebe.Z_i_Spin_Basis_sparse(n_TI, i_index)
    Z_i_Full = sp.kron(sp.eye(Subspace_basis_count_faster(n_PXP)),Z_i)
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),Initialstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ Z_i_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_Z_i_Bath_Sparse_Time_prop(n_PXP, n_TI, h_c,i_index,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param Initialstate: NeelHaar state usually
    :param J: Ising term strength
    :param h_x: longtitudinal field strength
    :param h_z: Traverse field strength
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = PXP_TI_Neelstate(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return Z_i_Bath_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index,h_imp=0, m=2)



def O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <Neel|O_z(t)|Neel> values and corresponding time values, FOR EXTENDED PXP BASIS, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param J: TI ising term strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param h_x: longtitudinal term strength (TI)
    :param h_z: transverse term strength
    :param h_c: coupling term strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param h_imp: impurity (TI) strength
    :param m: impurity site
    :return: vector -  <NeelxHaar|O_z(t)|NeelxHaar>
    '''
    O_z_PXP = Ebe.O_z_PXP_Entry_Sparse(n_PXP, PXP_Subspace_Algo_extended_X_i)
    O_z_Full = sp.kron(O_z_PXP,sp.eye(2**n_TI))
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse_Xi(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),Initialstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ O_z_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param Initialstate: NeelHaar state usually
    :param J: Ising term strength
    :param h_x: longtitudinal field strength
    :param h_z: Traverse field strength
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    try:
        os.mkdir('PXP_{}_TI_{}_Osc'.format(n_PXP, n_TI))
    except:
        pass
    if os.path.isfile('PXP_{}_TI_{}_Osc/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP,n_TI,n_PXP,n_TI,h_c,T_max,T_step)) == False:
        Initialstate = Neel_EBE_Haar_X_i_Extended(n_PXP, n_TI)
        J = 1
        h_x = np.sin(0.485 * np.pi)
        h_z = np.cos(0.485 * np.pi)
        Sandwich= O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)
        np.save(os.path.join('PXP_{}_TI_{}_Osc'.format(n_PXP, n_TI),'Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP,n_TI,h_c,T_max,T_step)),Sandwich)
Run_O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step)


def PXP_TI_Neelstate_True_X_i(n_PXP,n_TI):
    '''
    Combined Neel state of PXP and TI!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :return: kronekered Neel state
    '''
    TI_Neel= TI_Neelstate(n_TI)
    PXP_Neel= Neel_X_i_Extended_Subspace_Basis(n_PXP)
    PXP_TI_Neel = np.kron(PXP_Neel,TI_Neel)
    return PXP_TI_Neel


def Z_i_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0, m=2):
    '''
    Returns <Neel_bath|Z_i(t)|Neel_bath> values and corresponding time values FOR X_i coupling!, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param J: TI ising term strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param h_x: longtitudinal term strength (TI)
    :param h_z: transverse term strength
    :param h_c: coupling term strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param h_imp: impurity (TI) strength
    :param m: impurity site
    :return: vector -  <NeelxHaar|O_z(t)|NeelxHaar>
    '''
    Z_i = Ebe.Z_i_Spin_Basis_sparse(n_TI, i_index)
    Z_i_Full = sp.kron(sp.eye(Extended_X_i_Subspace_basis_count_faster(n_PXP)), Z_i)
    Propagated_ket = Ebe.spla.expm_multiply(-1j * Ebe.PXP_TI_coupled_Sparse_Xi(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),
                                            Initialstate,
                                            start=T_start, stop=T_max, num=T_step, endpoint=True)
    Propagated_ket_fin = np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ Z_i_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')


def Run_Z_i_Bath_True_X_i_Sparse_Time_prop(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param Initialstate: NeelHaar state usually
    :param J: Ising term strength
    :param h_x: longtitudinal field strength
    :param h_z: Traverse field strength
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = PXP_TI_Neelstate_True_X_i(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return Z_i_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0,
                              m=2)



