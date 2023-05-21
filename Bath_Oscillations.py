import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import PXP_E_B_E_Sparse as Ebe
from PXP_Entry_By_Entry import *
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.linalg import expm
from scipy.signal import find_peaks
from time import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Coupling_To_Bath import *
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
from scipy.optimize import curve_fit


def TI_Neelstate(n_TI): #Pure bath
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

def Z_i_TI_only_time_prop(n_TI, Initialstate, J, h_x, h_z, T_start, T_max, T_step,i_index):  #Pure bath
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
    :return: vector - <Neel_bath|Z_i(t)|Neel_bath>
    '''
    Z_i = Ebe.Z_i_Spin_Basis_sparse(n_TI, i_index)
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.TIOBCNew_Sparse(n_TI, J, h_x, h_z),Initialstate,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ Z_i @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_Z_i_TI_only_Sparse_Time_prop(n_TI,i_index,T_start, T_max, T_step):  #Pure bath
    '''
    Runs time propagation FOR TI MODEL ONLY (uncoupled to anything) plotter in EBE sparse method
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


def Run_Z_i_TI_only_Sparse_Time_prop_plt(n_TI,i_index,T_start, T_max, T_step):
    '''
    Runs time propagation FOR TI MODEL ONLY (uncoupled to anything) plotter in EBE sparse method
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
    sandwich= Z_i_TI_only_time_prop(n_TI, Initialstate, J, h_x, h_z, T_start, T_max, T_step,i_index)
    plt.plot(np.linspace(T_start,T_max,T_step),sandwich)
    plt.title('$<Z_{}>$ vs. time for {} TI atoms (Pure TI)'.format(i_index,n_TI))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    return plt.show()

def O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <NeelxHaar|O_z(t)|NeelxHaar> values and corresponding time values, working with EBE sparse method
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
    :return: vector -   <NeelxHaar|O_z(t)|NeelxHaar>
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
    Runs time  propagation plotter in EBE sparse method
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
    Initialstate = Neel_EBE_Haar(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)

def Run_O_z_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    Runs time propagation plotter in EBE sparse method
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
    Initialstate = Neel_EBE_Haar(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    sandwich = O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(r'Amp. of <Neel|O_z(t)|Neel> vs Time, Z_i Coupling Str. {}'.format(h_c))
    plt.savefig("Figures/System_Oscillations/System_Osc_Z_i_Coup_{}_{}_{}_M_time_{}.png".format(n_PXP,n_TI,h_c,T_max))
    return plt.show()


def PXP_TI_Neelstate(n_PXP,n_TI): #regular Z_i coupling
    '''
    Combined Neel state of PXP and TI! Z_i coupling!!!!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :return: kronekered Neel state
    '''
    TI_Neel= TI_Neelstate(n_TI)
    PXP_Neel= Neel_Subspace_Basis(n_PXP)
    PXP_TI_Neel = np.kron(PXP_Neel,TI_Neel)
    return PXP_TI_Neel


def Z_i_Bath_PXP_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index, h_imp=0, m=2):
    '''
    Returns <NeelxNeel_bath|Z_i(t)|NeelxNeel_bath> values and corresponding time values, working with EBE sparse method
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
    :return: vector -  <NeelxNeel_bath|Z_i(t)|NeelxNeel_bath>
    '''
    Z_i = Ebe.Z_i_Spin_Basis_sparse(n_TI, i_index)
    Z_i_Full = sp.kron(sp.eye(Subspace_basis_count_faster(n_PXP)),Z_i)
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),Initialstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ Z_i_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_Z_i_Bath_PXP_Sparse_Time_prop(n_PXP, n_TI, h_c,i_index,T_start, T_max, T_step):
    '''
    Runs time propagation plotter in EBE sparse method
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
    return Z_i_Bath_PXP_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index,h_imp=0, m=2)


def Run_Z_i_Bath_PXP_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step):
    '''
    Runs time propagation plotter in EBE sparse method
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
    sandwich= Z_i_Bath_PXP_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index,h_imp=0, m=2)
    plt.plot(np.linspace(T_start,T_max,T_step),sandwich)
    plt.title('$<Z_{}>$ vs. time for {} TI, {} PXP Atoms, $h_c$={} Z_i coupling'.format(i_index,n_TI,n_PXP,np.round(h_c,5)))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.savefig('Figures/Bath_Oscillations/Z_{}_Osc_{}_PXP_{}_TI_{}_Z_Coup.png'.format(i_index,n_PXP,n_TI,h_c))
    return plt.show()

# def Run_Z_i_Bath_Sparse_time_prop_fig_sys_size(n_PXP, i_index, T_start, T_max, T_step):
#     '''
#     Runs time propagation plotter in EBE sparse method
#     :param n_PXP: No. of PXP atoms
#     :param n_TI: No. of TI atoms
#     :param Initialstate: NeelHaar state usually
#     :param J: Ising term strength
#     :param h_x: longtitudinal field strength
#     :param h_z: Traverse field strength
#     :param h_c: coupling strength
#     :param T_start: start time
#     :param T_max: end time
#     :param T_step: time division
#     :return: Plot of Time propagation
#     '''
#     J = 1
#     h_x = np.sin(0.485 * np.pi)
#     h_z = np.cos(0.485 * np.pi)
#     avg = np.zeros((4))
#     n_TI_arr= np.arange(6,10,1)
#     for h_c in np.arange(0, 1.1, 0.1):
#         for n_TI in n_TI_arr:
#             Initialstate = PXP_TI_Neelstate(n_PXP, n_TI)
#             sandwich= Z_i_Bath_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index,h_imp=0, m=2)
#             avg[n_TI-6]= np.mean(sandwich)
#         plt.plot(n_TI_arr,avg, label='$h_c$={}'.format(np.round(h_c,5)))
#     plt.legend()
#     plt.xlabel('TI chain size')
#     plt.ylabel('Avg Mag Amp ({}th site)'.format(i_index))
#     plt.title('time averaged $<Z_{}>$ vs. TI size for var. $h_c$, {} PXP'.format(i_index,n_PXP))
#     return plt.show()

def O_z_True_X_i_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <NeelxHaar|O_z(t)|NeelxHaar> values and corresponding time values for True X_i, working with EBE sparse method
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

def Run_O_z_True_X_i_Sparse_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step):
    '''
    Runs time propagation True X_i plotter in EBE sparse method
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
    Initialstate = Neel_EBE_Haar_X_i_Extended(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return O_z_True_X_i_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)

def Run_O_z_True_X_i_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    Runs time propagation True X_i plotter in EBE sparse method
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
    Initialstate = Neel_EBE_Haar_X_i_Extended(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    sandwich = O_z_True_X_i_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(r'Amp. of <Neel|O_z(t)|Neel> vs Time, X_i Coupling Str. {}'.format(h_c))
    plt.savefig("Figures/System_Oscillations/System_Osc_X_i_Coup_{}_{}_{}_M_time_{}.png".format(n_PXP,n_TI,h_c,T_max))
    return plt.show()

def PXP_TI_Neelstate_True_X_i(n_PXP,n_TI): #X_i coupling
    '''
    Combined Neel state of PXP and TI for X_i coupling!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :return: kronekered Neel state
    '''
    TI_Neel= TI_Neelstate(n_TI)
    PXP_Neel= Neel_X_i_Extended_Subspace_Basis(n_PXP)
    PXP_TI_Neel = np.kron(PXP_Neel,TI_Neel)
    return PXP_TI_Neel


def Z_i_Bath_PXP_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0, m=2):
    '''
    Returns <NeelxNeel_bath|Z_i(t)|NeelxNeel_bath> values and corresponding time values FOR X_i coupling!, working with EBE sparse method
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
    :return: vector -  <NeelxNeel_bath|Z_i(t)|NeelxNeel_bath>
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


def Run_Z_i_Bath_PXP_True_X_i_Sparse_Time_prop(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step):
    '''
    Runs time propagation plotter in EBE sparse method
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
    return Z_i_Bath_PXP_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0,
                              m=2)

def Run_Z_i_Bath_PXP_True_X_i_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step):
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
    sandwich = Z_i_Bath_PXP_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0,
                                  m=2)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.title('$<Z_{}>$ vs. time for {} TI atoms {} PXP atoms $h_c$={} X_i coupling'.format(i_index, n_TI, n_PXP, np.round(h_c, 5)))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    #plt.savefig('Figures/Bath_Oscillations/Z_{}_Osc_{}_PXP_{}_TI_{}_X_Coup.png'.format(i_index,n_PXP,n_TI,h_c))
    return plt.show()


##########################################################################################################################################################
#                                                           FOURIER TRANSFORM check                                                                      #
##########################################################################################################################################################

def FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm):
    '''
    Gets positive frequency (absolute value of) fourier components of the propagation signal and positive frequencies
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 arrays (Positive freq, positive freq fourier components)
    '''
    VecProp = Run_O_z_Sparse_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step) #Run_Time_prop uses time division
    Fourier_components= rfft(VecProp)
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=(T_max/T_step)) # Freq =n/Tmax up to n=T_step
    return Freq, (Height_norm * np.abs(Fourier_components))


def Plot_FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff,End_cutoff):
    '''
    Plots positive frequency fourier components as a function of (positive) frequency
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot
    '''
    Freq, Inverse_sig = FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_cutoff:End_cutoff], Inverse_sig[Start_cutoff:End_cutoff], marker='o', markersize=3,
             color='b')
    return plt.show()

def Lorentzian_function(omega, omega_0, gamma, amp):
    '''
    Lorentzian function shape definer (variables and parameters)
    :param omega: Frequency variable
    :param omega_0: Frequency of oscillations = offset of delta function point
    :param gamma: damping
    :return: Lorentzian "shape" (scalar, just the f(x)= output of lorentzian)
    '''
    return np.divide(amp * gamma, gamma**2 + (omega-omega_0)**2)

def Lorentzian_curvefit(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff,End_cutoff):
    '''
    Fits lorentzian function to Fourier signal, returns gamma (damping coefficient)
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: optimal coefficients (in the order: Omega_0, gamma, Amplitude)
    '''
    Freq, sig_func = FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    return popt , pcov

def Lorentzian_curvefit_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff,End_cutoff):
    #T_max about 100-300 and T_step 1000-2000ish, remember that t_max/t_step = N_tot ; N_tot/t_max = max freq
    # (need to increas t_max and increase t_step so that number of N_tot does not !! go a lot over t_max)
    '''
    Fits lorentzian function to transformed signal, outputs plots of data and fit
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation; Bigger T_max - smaller frequencies! (N_tot/T_max top freq)
    :param T_step: time step (division) Bigger T_step - bigger frequencies! (N_tot=T_step, N_tot/T_max top freq)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: a plot of data (blue) and fit (red)
    '''
    Freq, sig_func = FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff], marker='o', markersize=3,
             color='b', label='data')
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    plt.plot(Freq[Start_cutoff:End_cutoff], Lorentzian_function(Freq[Start_cutoff:End_cutoff],*popt),'r-',
         label=r'fit: $\omega_0$={}, $\gamma$={}, Amp={}, $\Delta\gamma$={}'.format(*np.round(popt,4),np.round(np.diag(pcov)[1],10)))
    #plt.title('Frequency fit for {} PXP and {} TI atoms, Coupling strength {}'.format(n_PXP,n_TI,h_c))
    plt.xlabel(r'Frequency [$1/t$]')
    plt.ylabel('Amplitudes of Harmonic Functions')
    plt.legend()
    plt.title('Transformed Sig. vs Freq. {} PXP {} TI, $h_c$={} $Z_i$ Coupling'.format(n_PXP,n_TI,h_c))
    plt.savefig("Figures/Freq_Fit_May/Freq_Fit_{}_PXP_{}_TI_{}_Z_i_Coup_{}-{}.png".format(n_PXP,n_TI,h_c,Start_cutoff,End_cutoff))
    return plt.show()

def FFT_True_X_i(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm):
    '''
    Gets positive frequency (absolute value of) fourier components of the propagation signal and positive frequencies for X_i coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 arrays (Positive freq, positive freq fourier components)
    '''
    VecProp = Run_O_z_True_X_i_Sparse_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step) #Run_Time_prop uses time division
    Fourier_components= rfft(VecProp)
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=(T_max/T_step)) # Freq =n/Tmax up to n=T_step
    return Freq, (Height_norm * np.abs(Fourier_components))


def Plot_FFT_True_X_i(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff,End_cutoff):
    '''
    Plots positive frequency fourier components as a function of (positive) frequency for X_i coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot
    '''
    Freq, Inverse_sig = FFT_True_X_i(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_cutoff:End_cutoff], Inverse_sig[Start_cutoff:End_cutoff], marker='o', markersize=3,
             color='b')
    return plt.show()

def Lorentzian_curvefit_True_X_i(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff,End_cutoff):
    '''
    Fits lorentzian function to Fourier signal, returns gamma (damping coefficient) for X_i coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: optimal coefficients (in the order: Omega_0, gamma, Amplitude)
    '''
    Freq, sig_func = FFT_True_X_i(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    return popt, pcov

def Lorentzian_curvefit_True_X_i_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff,End_cutoff):
    #T_max about 100-300 and T_step 1000-2000ish, remember that t_max/t_step = N_tot ; N_tot/t_max = max freq
    # (need to increas t_max and increase t_step so that number of N_tot does not !! go a lot over t_max)
    '''
    Fits lorentzian function to transformed signal, outputs plots of data and fit
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation; Bigger T_max - smaller frequencies! (N_tot/T_max top freq)
    :param T_step: time step (division) Bigger T_step - bigger frequencies! (N_tot=T_step, N_tot/T_max top freq)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: a plot of data (blue) and fit (red)
    '''
    Freq, sig_func = FFT_True_X_i(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff], marker='o', markersize=3,
             color='b', label='data')
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    plt.plot(Freq[Start_cutoff:End_cutoff], Lorentzian_function(Freq[Start_cutoff:End_cutoff],*popt),'r-',
         label=r'fit: $\omega_0$={}, $\gamma$={}, Amp={}, $\Delta\gamma$={}'.format(*np.round(popt,4),np.round(np.diag(pcov)[1],10)))
    #plt.title('Frequency fit for {} PXP and {} TI atoms, Coupling strength {}'.format(n_PXP,n_TI,h_c))
    plt.xlabel(r'Frequency [$1/t$]')
    plt.ylabel('Amplitudes of Harmonic Functions')
    plt.legend()
    plt.title('Transformed Sig. vs Freq. {} PXP {} TI, $h_c$={} $X_i$ Coupling'.format(n_PXP,n_TI,h_c))
    plt.savefig("Figures/Freq_Fit_May/Freq_Fit_{}_PXP_{}_TI_{}_X_i_Coup_{}-{}.png".format(n_PXP,n_TI,h_c,Start_cutoff,End_cutoff))
    return plt.show()
