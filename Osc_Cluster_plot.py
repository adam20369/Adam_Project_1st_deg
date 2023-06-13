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


def System_O_z_System_O_z_Osc_Cluster_Plotter_fast(n_PXP_1, n_PXP_2, n_TI_1, n_TI_2, h_c_1,h_c_2,time_cutoff_div,T_start,T_max,T_step):
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
    sandwich_O_z_1_sys=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_1,n_TI_1,h_c_1,T_max,T_step))
    sandwich_O_z_2_sys=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_2,n_TI_2,h_c_2,T_max,T_step))
    plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff_div],sandwich_O_z_1_sys[:time_cutoff_div], label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff_div],sandwich_O_z_2_sys[:time_cutoff_div],label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
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
            try:
                os.mkdir('Oscillations/TI_{}/True_X_i_h_c_{}'.format( n_TI, np.round(h_c, 2)))
            except:
                pass
            plt.savefig('Oscillations/TI_{}/True_X_i_h_c_{}/Osc_plot_Cutoff_{}_PXP_{}_True_X_i.png'.format(n_TI,np.round(h_c,2),time_cutoff_div,n_PXP))
            plt.show()

def System_O_z_System_O_z_Osc_True_X_i_Cluster_Plotter_fast(n_PXP_1, n_PXP_2, n_TI_1, n_TI_2, h_c_1,h_c_2,time_cutoff_div,T_start,T_max,T_step):
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
    sandwich_O_z_1_sys=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_1,n_TI_1,h_c_1,T_max,T_step))
    sandwich_O_z_2_sys=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_2,n_TI_2,h_c_2,T_max,T_step))
    plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff_div],sandwich_O_z_1_sys[:time_cutoff_div], label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff_div],sandwich_O_z_2_sys[:time_cutoff_div],label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Compare $O_z$ of Sys, $XX$ coupling (see legend)')
    plt.ylim(-0.8,0.2)
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

##############################################################################################################################
##########################                       Fast FFT CHECK                                        ############################
##############################################################################################################################

def Lorentzian_function(omega, omega_0, gamma, amp):
    '''
    Lorentzian function shape definer (variables and parameters)
    :param omega: Frequency variable
    :param omega_0: Frequency of oscillations = offset of delta function point
    :param gamma: damping
    :return: Lorentzian "shape" (scalar, just the f(x)= output of lorentzian)
    '''
    return np.divide(amp * gamma, gamma**2 + (omega-omega_0)**2)

def FFT_Compare_fast_plt(n_PXP_1, n_PXP_2, n_TI_1,n_TI_2, h_c_1,h_c_2,fourier_cutoff_div,Optimal_start,Optimal_end,start_cutoff_plt,end_cutoff_plt,T_start,T_max,T_step):
    '''
    compares plots of 2 Fourier transformed (absolute value of) propagation signals as a function of frequency (positive)
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param start_cutoff_plt: start cutoff visual plot ONLY
    :param end_cutoff_plt: end cutoff visual plot ONLY
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 plots of fourier transformed signals
    '''
    # T_start = 0
    # T_max = 40
    # T_step = 500
    time=np.linspace(T_start,T_max,T_step)
    sandwich_O_z_1_sys=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_1,n_TI_1,h_c_1,T_max,T_step))
    sandwich_O_z_2_sys=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_2,n_TI_2,h_c_2,T_max,T_step))
    Fourier_components_O_z_1= rfft(sandwich_O_z_1_sys[:fourier_cutoff_div])
    Fourier_components_O_z_2 = rfft(sandwich_O_z_2_sys[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    popt_1, pcov_1 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    popt_2, pcov_2 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_2[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    sandwich_O_z_gamma_0_sys_1=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_1,n_TI_1,0.0,T_max,T_step))
    Fourier_components_O_z_gamma_0_sys_1=rfft(sandwich_O_z_gamma_0_sys_1[:fourier_cutoff_div])
    popt_gamma_0_sys_1,pcov_gamma_0_sys_1= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_1[Optimal_start:Optimal_end]))
    sandwich_O_z_gamma_0_sys_2=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_2,n_TI_2,0.0,T_max,T_step))
    Fourier_components_O_z_gamma_0_sys_2=rfft(sandwich_O_z_gamma_0_sys_2[:fourier_cutoff_div])
    popt_gamma_0_sys_2,pcov_gamma_0_sys_2= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_2[Optimal_start:Optimal_end]))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_1[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$'.format(n_PXP_1,n_TI_1,h_c_1))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_2[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$'.format(n_PXP_2,n_TI_2,h_c_2))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_1),linestyle='dashed', marker='o', markersize=1,label=r'Blue $O_z$ 1 fit $\gamma={}$, $\Delta\gamma={}$'.format(np.round(popt_1[1]/popt_gamma_0_sys_1[1],5),np.round(pcov_1[1,1]/popt_gamma_0_sys_1[1],10)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_2),linestyle='dashed', marker='o', markersize=1,label=r'Orange $O_z$ 2 fit $\gamma={}$,  $\Delta\gamma={}$'.format(np.round(popt_2[1]/popt_gamma_0_sys_2[1],5),np.round(pcov_2[1,1]/popt_gamma_0_sys_2[1],10)))
    plt.legend()
    plt.title('Comparison of Fourier signals ZZ coupling')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    return plt.show()

def FFT_True_X_i_Compare_fast_plt(n_PXP_1, n_PXP_2, n_TI_1,n_TI_2, h_c_1,h_c_2,fourier_cutoff_div,Optimal_start,Optimal_end,start_cutoff_plt,end_cutoff_plt,T_start,T_max,T_step):
    '''
    compares plots of 2 XX coupling Fourier transformed (absolute value of) propagation signal as a function of frequency (positive)
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param start_cutoff_plt: start cutoff visual plot ONLY
    :param end_cutoff_plt: end cutoff visual plot ONLY
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 plots of fourier transformed signals
    '''
    # T_start = 0
    # T_max = 40
    # T_step = 500
    time=np.linspace(T_start,T_max,T_step)
    sandwich_O_z_1_sys=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_1,n_TI_1,h_c_1,T_max,T_step))
    sandwich_O_z_2_sys=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_2,n_TI_2,h_c_2,T_max,T_step))
    Fourier_components_O_z_1= rfft(sandwich_O_z_1_sys[:fourier_cutoff_div])
    Fourier_components_O_z_2 = rfft(sandwich_O_z_2_sys[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    popt_1, pcov_1 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    popt_2, pcov_2 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_2[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    sandwich_O_z_gamma_0_sys_1=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_1,n_TI_1,0.0,T_max,T_step))
    Fourier_components_O_z_gamma_0_sys_1=rfft(sandwich_O_z_gamma_0_sys_1[:fourier_cutoff_div])
    popt_gamma_0_sys_1,pcov_gamma_0_sys_1= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_1[Optimal_start:Optimal_end]))
    sandwich_O_z_gamma_0_sys_2=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_2,n_TI_2,0.0,T_max,T_step))
    Fourier_components_O_z_gamma_0_sys_2=rfft(sandwich_O_z_gamma_0_sys_2[:fourier_cutoff_div])
    popt_gamma_0_sys_2,pcov_gamma_0_sys_2= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_2[Optimal_start:Optimal_end]))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_1[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$'.format(n_PXP_1,n_TI_1,h_c_1))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_2[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$'.format(n_PXP_2,n_TI_2,h_c_2))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_1),linestyle='dashed', marker='o', markersize=1,label=r'Blue $O_z$ 1 fit $\gamma={}$, $\Delta\gamma={}$'.format(np.round(popt_1[1]/popt_gamma_0_sys_1[1],5),np.round(pcov_1[1,1]/popt_gamma_0_sys_1[1],10)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_2),linestyle='dashed', marker='o', markersize=1,label=r'Orange $O_z$ 2 fit $\gamma={}$,  $\Delta\gamma={}$'.format(np.round(popt_2[1]/popt_gamma_0_sys_2[1],5),np.round(pcov_2[1,1]/popt_gamma_0_sys_2[1],10)))
    plt.legend()
    plt.title('Comparison of Fourier signals XX coupling')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    return plt.show()

def FFT_Compare_ZZ_XX_fast_plt(n_PXP_1, n_PXP_2, n_TI_1,n_TI_2, h_c_1,h_c_2,fourier_cutoff_div,Optimal_start,Optimal_end,start_cutoff_plt,end_cutoff_plt,T_start,T_max,T_step):
    '''
    compares plots of XX and ZZ coupling Fourier transformed (absolute value of) propagation signal as a function of frequency (positive)
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param start_cutoff_plt: start cutoff visual plot ONLY
    :param end_cutoff_plt: end cutoff visual plot ONLY
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 plots of fourier transformed signals
    '''
    time=np.linspace(T_start,T_max,T_step)
    sandwich_O_z_1_sys=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_1,n_TI_1,h_c_1,T_max,T_step))
    sandwich_O_z_2_sys=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_2,n_TI_2,h_c_2,T_max,T_step))
    Fourier_components_O_z_1= rfft(sandwich_O_z_1_sys[:fourier_cutoff_div])
    Fourier_components_O_z_2 = rfft(sandwich_O_z_2_sys[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    popt_1, pcov_1 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    popt_2, pcov_2 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_2[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    sandwich_O_z_gamma_0_sys_1=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_1,n_TI_1,0.0,T_max,T_step))
    Fourier_components_O_z_gamma_0_sys_1=rfft(sandwich_O_z_gamma_0_sys_1[:fourier_cutoff_div])
    popt_gamma_0_sys_1,pcov_gamma_0_sys_1= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_1[Optimal_start:Optimal_end]))
    sandwich_O_z_gamma_0_sys_2=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP_2,n_TI_2,0.0,T_max,T_step))
    Fourier_components_O_z_gamma_0_sys_2=rfft(sandwich_O_z_gamma_0_sys_2[:fourier_cutoff_div])
    popt_gamma_0_sys_2,pcov_gamma_0_sys_2= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_2[Optimal_start:Optimal_end]))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_1[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$ ZZ Coupling'.format(n_PXP_1,n_TI_1,h_c_1))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_2[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$ XX Coupling'.format(n_PXP_2,n_TI_2,h_c_2))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_1),linestyle='dashed', marker='o', markersize=1,label=r'ZZ Coupling (blue) $O_z$ fit $\gamma={}$, $\Delta\gamma={}$'.format(np.round(popt_1[1]/popt_gamma_0_sys_1[1],5),np.round(pcov_1[1,1]/popt_gamma_0_sys_1[1],10)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_2),linestyle='dashed', marker='o', markersize=1,label=r'XX Coupling (Orange) $O_z$ fit $\gamma={}$,  $\Delta\gamma={}$'.format(np.round(popt_2[1]/popt_gamma_0_sys_2[1],5),np.round(pcov_2[1,1]/popt_gamma_0_sys_2[1],10)))
    plt.legend()
    plt.title('Comparison of Fourier signals ZZ and XX Coupling')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    return plt.show()

def FFT_Gamma_plot_fast(n_PXP, n_TI, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step):
    '''
    Plots gamma as a function of coupling for given number of atoms ZZ coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot of gamma vs h_c strength
    '''
    time=np.linspace(T_start,T_max,T_step)
    h_c_array= np.arange(0.0,h_c_max,0.1)
    gamma_array= np.zeros(len(h_c_array))
    gamma_pcov_array= np.zeros(len(h_c_array))
    count=0
    sandwich_O_z_gamma_0_sys = np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP, n_TI, 0.0, T_max,T_step))
    Fourier_components_O_z_gamma_0_sys = rfft(sandwich_O_z_gamma_0_sys[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=(T_max / T_step))  # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_gamma_0_sys[Optimal_start:Optimal_end]))
    for h_c in np.nditer(h_c_array):
        sandwich_O_z_sys=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2),T_max,T_step))
        Fourier_components_O_z_1= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
        popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
        gamma_array[count]=popt[1]/popt_gamma_0_sys[1]
        gamma_pcov_array[count]=pcov[1,1]/popt_gamma_0_sys[1]
        count=count+1
    plt.plot(h_c_array, gamma_array, marker='o', markersize=1)
    plt.title('Damping coef vs Coup str, ZZ Coup for {} PXP,{} TI, Time {} Steps {}'.format(n_PXP,n_TI,T_max,T_step))
    plt.xlabel('Coupling Strength')
    plt.ylabel('Damping Coefficient (Normalized)')
    plt.ylim(0.88,1.82)
    print(gamma_pcov_array)
    return plt.show()

def FFT_True_X_i_Gamma_plot_fast(n_PXP, n_TI, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step):
    '''
    Plots gamma as a function of coupling for given number of atoms XX coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot of gamma vs h_c strength
    '''
    time=np.linspace(T_start,T_max,T_step)
    h_c_array= np.arange(0.0,h_c_max,0.1)
    gamma_array= np.zeros(len(h_c_array))
    gamma_pcov_array= np.zeros(len(h_c_array))
    count=0
    Freq = rfftfreq(fourier_cutoff_div, d=(T_max / T_step))  # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    sandwich_O_z_gamma_0_sys = np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP, n_TI, 0.0,T_max, T_step))
    Fourier_components_O_z_gamma_0_sys = rfft(sandwich_O_z_gamma_0_sys[:fourier_cutoff_div])
    popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_gamma_0_sys[Optimal_start:Optimal_end]))
    for h_c in np.nditer(h_c_array):
        sandwich_O_z_sys=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2),T_max,T_step))
        Fourier_components_O_z_1= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
        popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
        gamma_array[count]=popt[1]/popt_gamma_0_sys[1]
        gamma_pcov_array[count]=pcov[1,1]/popt_gamma_0_sys[1]
        count=count+1
    plt.plot(h_c_array, gamma_array, marker='o', markersize=1)
    plt.title('Damping coef vs Coup str, $XX$ Coup for {} PXP,{} TI, Time {} Steps {}'.format(n_PXP,n_TI,T_max,T_step))
    plt.xlabel('Coupling Strength')
    plt.ylabel('Damping Coefficient (Normalized)')
    plt.ylim(0.88,1.82)
    print(gamma_pcov_array)
    return plt.show()

def Inverse_FFT_Signal_Compare_plt(n_PXP, n_TI, h_c,time_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step):
    '''
    compares plots of ZZ coupling inverse Fourier transformed fit of lorentzian with time series
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 plots
    '''
    time=np.linspace(T_start,T_max,T_step)
    sandwich_O_z_sys=np.load('Oscillations/Time_Propagation_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP,n_TI,h_c,T_max,T_step))
    Fourier_components_O_z= rfft(sandwich_O_z_sys[:time_cutoff_div])
    Freq = rfftfreq(time_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    Inverse_Fourier_array=np.zeros(len(Freq))
    popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    Inverse_Fourier_array[Optimal_start:Optimal_end]=Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt)
    inverse_transform_O_z_fit= irfft(Inverse_Fourier_array,len(time[:time_cutoff_div]))
    inverse_transform_O_z_full=irfft(np.abs(Fourier_components_O_z),len(time[:time_cutoff_div]))
    offset=np.mean(inverse_transform_O_z_full)
    plt.plot(time[:time_cutoff_div], inverse_transform_O_z_full, marker='o', markersize=3,label=r'Original Time series')
    plt.plot(time[:time_cutoff_div],offset+inverse_transform_O_z_fit, marker='o', markersize=1,label=r'Inverse transformed fitted signal')
    plt.legend()
    plt.title('Comparison of Time series to inverse transform of fit ZZ coupling')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    return plt.show()

def Inverse_FFT_Signal_Compare_True_X_i_plt(n_PXP, n_TI, h_c,time_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step):
    '''
    compares plots of XX coupling inverse Fourier transformed fit of lorentzian with time series
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param start_cutoff_plt: start cutoff visual plot ONLY
    :param end_cutoff_plt: end cutoff visual plot ONLY
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 plots
    '''
    time=np.linspace(T_start,T_max,T_step)
    sandwich_O_z_sys=np.load('Oscillations/Time_Propagation_True_X_i_PXP_{}_TI_{}_h_c_{}_Max_T_{}_step_{}.npy'.format(n_PXP,n_TI,h_c,T_max,T_step))
    Fourier_components_O_z= rfft(sandwich_O_z_sys[:time_cutoff_div])
    Freq = rfftfreq(time_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    Inverse_Fourier_array=np.zeros(len(Freq))
    popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    Inverse_Fourier_array[Optimal_start:Optimal_end]=Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt)
    inverse_transform_O_z_fit= irfft(Inverse_Fourier_array,len(time[:time_cutoff_div]))
    inverse_transform_O_z_full=irfft(np.abs(Fourier_components_O_z),len(time[:time_cutoff_div]))
    offset=np.mean(inverse_transform_O_z_full)
    plt.plot(time[:time_cutoff_div], inverse_transform_O_z_full, marker='o', markersize=3,label=r'Original Time series')
    plt.plot(time[:time_cutoff_div],offset+inverse_transform_O_z_fit, marker='o', markersize=1,label=r'Inverse transformed fitted signal')
    plt.legend()
    plt.title('Comparison of Time series to inverse transform of fit XX coupling')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    return plt.show()
