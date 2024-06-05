import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PXP_Entry_By_Entry import *
from PXP_E_B_E_Sparse import *
import O_z_Oscillations as Ozosc
import numpy.linalg as la
from time import time
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import comb
from scipy.stats import bootstrap
from scipy.interpolate import UnivariateSpline
import numpy.polynomial as poly
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def old_Osc_grapher():
    '''
    Old code Osc grapher
    :return:
    '''
    T_start = 0
    T_max = 400
    T_step = 1000

    n_PXP = 10
    n_TI = 13
    h_c = 0.4
    data_ave = np.load('PXP_{}_Osc_Ave/Sparse_time_propagation_ave_{}_{}_{}.npy'.format(n_PXP,n_PXP,n_TI,h_c))
    data_errors = np.load('PXP_{}_Osc_Ave/Sparse_time_propagation_errors_{}_{}_{}.npy'.format(n_PXP,n_PXP,n_TI,h_c))
    data_errors_fin_0 = data_ave-data_errors[0,:] #DATA ERRORS NEED TO BE +/- from Null
    data_errors_fin_1 = data_errors[1,:]-data_ave #DATA ERRORS NEED TO BE +/- from Null
    data_errors_fin= np.empty((2,len(data_errors_fin_0)))
    data_errors_fin[0,:]=data_errors_fin_0
    data_errors_fin[1,:]=data_errors_fin_1
    #data_errors_fin= np.fabs(data_errors_fin)
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    plt.errorbar(Time[:200], data_ave[:200], yerr= data_errors_fin[:200,:200], fmt='bs-', ecolor='r',markersize=2,elinewidth=4)
    #plt.errorbar(Time[:], data_ave[:], yerr= data_errors_fin[:,:], fmt='bs-', ecolor='r',markersize=2,elinewidth=4)
    plt.title(r'$\langle O_z(t)\rangle$ Vs. time for {} PXP atoms, {} TI Atoms, {} $h_c$ strength'.format(n_PXP,n_TI,h_c))
    plt.ylabel(r' $\langle O_z(t)\rangle$ ', fontsize=10)
    plt.xlabel(r'Time',fontsize=12)
    plt.show()


def System_O_z_Osc_Cluster_Plotter(n_TI,n_PXP_max,time_cutoff_div):
    '''
    Plots system oscillations of O_z data
    :return: plot of <NeelxHaar|O_z|Neel_Haar> for various h_c and n_TI
    '''
    T_start = 0
    T_max = 400
    T_step = 1000
    h_c_array=np.arange(0.0,1.1,0.1)
    for h_c in np.nditer(h_c_array):
        for n_PXP in range(8,n_PXP_max+1):
            data_ave = np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2)))
            Time = np.linspace(T_start, T_max, T_step, endpoint=True)
            plt.plot(Time[:time_cutoff_div], data_ave[:time_cutoff_div])
            plt.title(r'$\langle O_z(t)\rangle$ Vs. time for {} PXP atoms, {} TI Atoms, {} $Z_i$ strength'.format(n_PXP,n_TI,np.round(h_c,2)))
            plt.ylim(-1,0.5)
            plt.ylabel(r' $\langle O_z(t)\rangle$ ', fontsize=10)
            plt.xlabel(r'Time',fontsize=12)
            try:
                os.mkdir('Oscillations/TI_{}/h_c_{}'.format( n_TI, np.round(h_c, 2)))
            except:
                pass
            plt.savefig('Oscillations/TI_{}/h_c_{}/Osc_Cutoff_{}_plot_PXP_{}.png'.format(n_TI,np.round(h_c,2),time_cutoff_div,n_PXP))
            plt.show()


def System_O_z_System_O_z_Osc_Cluster_Plotter_fast(n_PXP_1, n_PXP_2, n_TI_1, n_TI_2, h_c_1,h_c_2,time_cutoff_div,T_start,T_max,T_step,sample_1,sample_2):
    '''
    Plot of comparison between O_z of system to O_z of system for Z_i coupling
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param i_index: index of Z_i operator
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :param Cutoff_step: cutoff of steps in plots
    :return: graph with 2 plots
    '''
    sandwich_O_z_1_sys=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_1,n_TI_1,T_max,T_step,h_c_1,n_PXP_1,n_TI_1,h_c_1,T_max,T_step,sample_1))
    sandwich_O_z_2_sys=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_2,n_TI_2,T_max,T_step,h_c_2,n_PXP_2,n_TI_2,h_c_2,T_max,T_step,sample_2))
    plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff_div],sandwich_O_z_1_sys[:time_cutoff_div],label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff_div],sandwich_O_z_2_sys[:time_cutoff_div],label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(r'<Néel|$O_z(t)$|Néel>')
    plt.title('Compare $O_z$ of Sys (see legend)')
    plt.ylim(-0.8,0.2)
    plt.show()



def System_O_z_Osc_True_X_i_Cluster_Plotter(n_TI,n_PXP_max,time_cutoff_div):
    '''
    Plots system oscillations of O_z data FOR extended X_i basis
    :return: plot of <NeelxHaar|O_z|Neel_Haar> for various h_c and n_TI
    '''
    T_start = 0
    T_max = 400
    T_step = 1000

    h_c_array=np.arange(0.0,1.1,0.1)
    for h_c in np.nditer(h_c_array):
        for n_PXP in range(8,n_PXP_max+1):
            data_ave = np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2)))
            Time = np.linspace(T_start, T_max, T_step, endpoint=True)
            plt.plot(Time[:time_cutoff_div], data_ave[:time_cutoff_div])
            plt.title(r'$\langle O_z(t)\rangle$ Vs. time for {} PXP atoms, {} TI Atoms, {} $X_i$ strength'.format(n_PXP,n_TI,np.round(h_c,2)))
            plt.ylim(-1,0.5)
            plt.ylabel(r' $\langle O_z(t)\rangle$ ', fontsize=10)
            plt.xlabel(r'Time',fontsize=12)
            #try:
                #os.mkdir('Oscillations/TI_{}/True_X_i_h_c_{}'.format( n_TI, np.round(h_c, 2)))
            #except:
                #pass
            #plt.savefig('Oscillations/TI_{}/True_X_i_h_c_{}/Osc_plot_Cutoff_{}_PXP_{}_True_X_i.png'.format(n_TI,np.round(h_c,2),time_cutoff_div,n_PXP))
            plt.show()

def System_O_z_System_O_z_Osc_True_X_i_Cluster_Plotter_fast(n_PXP_1, n_PXP_2, n_TI_1, n_TI_2, h_c_1,h_c_2,time_cutoff_div,T_start,T_max,T_step,sample_1,sample_2):
    '''
    Plot of comparison between O_z of system to O_z of system for X_i coupling nature!!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param i_index: index of Z_i operator
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :param Cutoff_step: cutoff of steps in plots
    :return: graph with 2 plots
    '''
    sandwich_O_z_1_sys=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_1,n_TI_1,T_max,T_step,h_c_1,n_PXP_1,n_TI_1,h_c_1,T_max,T_step,sample_1))
    sandwich_O_z_2_sys=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_2,n_TI_2,T_max,T_step,h_c_2,n_PXP_2,n_TI_2,h_c_2,T_max,T_step,sample_2))
    plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff_div],sandwich_O_z_1_sys[:time_cutoff_div], label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff_div],sandwich_O_z_2_sys[:time_cutoff_div],label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(r'<NéelxHaar|$O_z^{PXP}(t)$|NéelxHaar>')
    plt.title('Compare $O_z$ of Sys, $XX$ coupling (see legend)')
    plt.ylim(-0.8,0.2)
    #plt.xlim(0,30)
    #plt.savefig('Damped_Oscillations_presentation.png')
    plt.show()


def System_O_z_Osc_Cluster_Plotter_TI(n_PXP,time_cutoff_div):
    '''
    Plots system oscillations of O_z data
    :return: plot of <NeelxHaar|O_z|Neel_Haar> for various h_c and n_TI
    '''
    T_start = 0
    T_max = 400
    T_step = 1000
    h_c_array=np.arange(0.0,1.1,0.1)
    for h_c in np.nditer(h_c_array):
        for n_TI in range(8,11):
            data_ave = np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2)))
            Time = np.linspace(T_start, T_max, T_step, endpoint=True)
            plt.plot(Time[:time_cutoff_div], data_ave[:time_cutoff_div])
            plt.title(r'$\langle O_z(t)\rangle$ Vs. time for {} PXP atoms, {} TI Atoms, {} $Z_i$ strength'.format(n_PXP,n_TI,np.round(h_c,2)))
            plt.ylim(-1,0.5)
            plt.ylabel(r' $\langle O_z(t)\rangle$ ', fontsize=10)
            plt.xlabel(r'Time',fontsize=12)
            try:
                os.mkdir('Oscillations/PXP_{}/h_c_{}'.format(n_PXP, np.round(h_c, 2)))
            except:
                pass
            plt.savefig('Oscillations/PXP_{}/h_c_{}/Osc_plot_Cutoff_{}_TI_{}.png'.format(n_PXP,np.round(h_c,2),time_cutoff_div,n_TI))
            plt.show()


def System_O_z_Osc_True_X_i_Cluster_Plotter_TI(n_PXP,time_cutoff_div):
    '''
    Plots system oscillations of O_z data FOR extended X_i basis
    :return: plot of <NeelxHaar|O_z|Neel_Haar> for various h_c and n_TI
    '''
    T_start = 0
    T_max = 400
    T_step = 1000

    h_c_array=np.arange(0.0,1.1,0.1)
    for h_c in np.nditer(h_c_array):
        for n_TI in range(8,11):
            data_ave = np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2)))
            Time = np.linspace(T_start, T_max, T_step, endpoint=True)
            plt.plot(Time[:time_cutoff_div], data_ave[:time_cutoff_div])
            plt.title(r'$\langle O_z(t)\rangle$ Vs. time for {} PXP atoms, {} TI Atoms, {} $X_i$ strength'.format(n_PXP,n_TI,np.round(h_c,2)))
            plt.ylim(-1,0.5)
            plt.ylabel(r' $\langle O_z(t)\rangle$ ', fontsize=10)
            plt.xlabel(r'Time',fontsize=12)
            try:
                os.mkdir('Oscillations/PXP_{}/True_X_i_h_c_{}/'.format(n_PXP, np.round(h_c, 2)))
            except:
                pass
            plt.savefig('Oscillations/PXP_{}/True_X_i_h_c_{}/Osc_plot_Cutoff_{}_TI_{}_True_X_i.png'.format(n_PXP,np.round(h_c,2),time_cutoff_div,n_TI))
            plt.show()


def System_O_z_Osc_Cluster_Plotter_h_c(n_PXP,n_TI,time_cutoff_div):
    '''
    Plots system oscillations of O_z data
    :return: plot of <NeelxHaar|O_z|Neel_Haar> for various h_c and n_TI
    '''
    T_start = 0
    T_max = 400
    T_step = 1000
    h_c_array=np.arange(0.0,1.1,0.1)
    for h_c in np.nditer(h_c_array):
        data_ave = np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2)))
        Time = np.linspace(T_start, T_max, T_step, endpoint=True)
        plt.plot(Time[:time_cutoff_div], data_ave[:time_cutoff_div])
        plt.title(r'$\langle O_z(t)\rangle$ Vs. time for {} PXP atoms, {} TI Atoms, {} $Z_i$ strength'.format(n_PXP,n_TI,np.round(h_c,2)))
        plt.ylim(-1,0.5)
        plt.ylabel(r' $\langle O_z(t)\rangle$ ', fontsize=10)
        plt.xlabel(r'Time',fontsize=12)
        try:
            os.mkdir('Oscillations/PXP_{}_TI_{}/'.format(n_PXP,n_TI))
        except:
            pass
        plt.savefig('Oscillations/PXP_{}_TI_{}/Osc_plot_Cutoff_{}_h_c_{}.png'.format(n_PXP,n_TI,time_cutoff_div,np.round(h_c,2)))
        plt.show()


def System_O_z_Osc_True_X_i_Cluster_Plotter_h_c(n_PXP,n_TI,time_cutoff_div):
    '''
    Plots system oscillations of O_z data FOR extended X_i basis
    :return: plot of <NeelxHaar|O_z|Neel_Haar> for various h_c and n_TI
    '''
    T_start = 0
    T_max = 400
    T_step = 1000

    h_c_array=np.arange(0.0,1.1,0.1)
    for h_c in np.nditer(h_c_array):
        data_ave = np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2)))
        Time = np.linspace(T_start, T_max, T_step, endpoint=True)
        plt.plot(Time[:time_cutoff_div], data_ave[:time_cutoff_div])
        plt.title(r'$\langle O_z(t)\rangle$ Vs. time for {} PXP atoms, {} TI Atoms, {} $X_i$ strength'.format(n_PXP,n_TI,np.round(h_c,2)))
        plt.ylim(-1,0.5)
        plt.ylabel(r' $\langle O_z(t)\rangle$ ', fontsize=10)
        plt.xlabel(r'Time',fontsize=12)
        try:
            os.mkdir('Oscillations/True_X_i_PXP_{}_TI_{}/'.format(n_PXP,n_TI))
        except:
            pass
        plt.savefig('Oscillations/True_X_i_PXP_{}_TI_{}/Osc_plot_Cutoff_{}_h_c_{}_True_X_i.png'.format(n_PXP,n_TI,time_cutoff_div,np.round(h_c,2)))
        plt.show()


def Old_New_Even_Neel_Check(n_PXP, n_TI, h_c,sample):
    '''
    Checking if the old and new files are the same? for PXP 10
    :param n_PXP:
    :param n_TI:
    :param h_c:
    :param T_start:
    :param T_max:
    :param T_step:
    :param sample:
    :return:
    '''

    VecPropNew = np.load('PXP_{}_TI_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP, n_TI, h_c,n_PXP, n_TI, h_c,sample))
    VecPropOld = np.load('PXP_{}_TI_{}/h_c_{}/Sparse_time_propagation_Old_Even_Neel_{}_{}_{}_sample_{}.npy'.format(n_PXP, n_TI, h_c,n_PXP, n_TI, h_c,sample))
    print(np.allclose(VecPropNew,VecPropOld))

def Old_New_Even_Neel_Check_True_X_i(n_PXP, n_TI, h_c,sample):
    '''
    Checking if the old and new files are the same? for PXP 10
    :param n_PXP:
    :param n_TI:
    :param h_c:
    :param T_start:
    :param T_max:
    :param T_step:
    :param sample:
    :return:
    '''
    VecPropNew = np.load('PXP_{}_TI_{}_True_X_i/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_sample_{}.npy'.format(n_PXP, n_TI, h_c,n_PXP, n_TI, h_c,sample))
    VecPropOld = np.load('PXP_{}_TI_{}_True_X_i/h_c_{}/Sparse_time_propagation_True_X_i_Old_Even_Neel_{}_{}_{}_sample_{}.npy'.format(n_PXP, n_TI, h_c,n_PXP, n_TI, h_c,sample))
    print(np.allclose(VecPropNew, VecPropOld))

