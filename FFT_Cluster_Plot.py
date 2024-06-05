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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler

#### the optimal cuts for fit 105 and 235 are for T=400, for other times the formula is x/400 * T_max_new (x/(400/T_max_new)) ####
n_PXP_feed = 10
n_TI_max_feed = 13
T_start_feed = 0
T_max_feed = 400
T_step_feed = 1000
h_c_max_feed = 3
h_c_step_feed = 0.2
Start_cutoff_feed = 105
End_cutoff_feed = 235

#data_check_Ave=np.load('PXP_10_True_X_i_Gammas_cutoff_105_235/Gamma_ave_True_X_i_10_10_2.0_cutoff_105_235.npy')
#print(data_check_Ave)

def Plot_vs_h_c_diff_n_TI_condifence():
    '''
    Plot for different TI numbers, a normalized damping graph vs Coupling strength h_c
    :return: a plot of plots
    '''
    n_PXP= n_PXP_feed
    T_start = T_start_feed
    T_max = T_max_feed
    T_step = T_step_feed
    Start_cutoff = Start_cutoff_feed
    End_cutoff = End_cutoff_feed
    h_c_max = h_c_max_feed
    h_c_step = h_c_step_feed
    n_TI_start = n_PXP
    n_TI_max = n_TI_max_feed
    n_TI= np.arange(n_TI_start,n_TI_max+1,1)
    h_c= np.arange(0, h_c_max+h_c_step, h_c_step)
    data_ave= np.empty((len(h_c)))
    data_errors= np.empty((2,(len(h_c))))
    cmap = cm.get_cmap('plasma',int(n_TI_max-5)) #length of colormap should be about x2 of the number of plots
    for j in np.nditer(n_TI):
        count = 0
        for i in np.nditer(h_c):
            data_ave[count]=np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_ave_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP,Start_cutoff,End_cutoff,n_PXP,int(j),np.round(i,1),Start_cutoff,End_cutoff))
            #print(data_ave)
            data_errors[:,count]=np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_errors_confidence_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP,Start_cutoff,End_cutoff,n_PXP,int(j),np.round(i,1),Start_cutoff,End_cutoff))
            #print(data_errors)
            count=count+1
        data_errors_fin_0 = data_ave - data_errors[0, :]  # DATA ERRORS NEED TO BE +/- from Null
        data_errors_fin_1 = data_errors[1, :] - data_ave  # DATA ERRORS NEED TO BE +/- from Null
        data_errors_fin = np.empty((2, len(data_errors_fin_0)))
        data_errors_fin[0, :] = data_errors_fin_0/data_ave[0] #Scaled errors as to gamma/gamma_0
        data_errors_fin[1, :] = data_errors_fin_1/data_ave[0] #Scaled errors as to gamma/gamma_0
        scaled_data_ave= data_ave/data_ave[0] #Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin[:, :], color=cmap.colors[int(j-n_TI[0])], marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3, label= '{} TI atoms'.format(int(j)))
        #plt.fill_between(h_c[:], scaled_data_ave[:] - data_errors_fin[0, :], scaled_data_ave[:] + data_errors_fin[1, :])
    plt.title(r'$Z_i$ coupling normalized damping for {} PXP atoms vs $h_c$, cutoff {}-{}'.format(n_PXP,Start_cutoff,End_cutoff))
    plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
    plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
    plt.legend()
    plt.savefig('Figures/True_X_i_Osc/Z_i_damping_{}_PXP_vs_h_c_{}_confi_err_cutoff_{}_{}.png'.format(n_PXP,h_c_max,Start_cutoff,End_cutoff))
    return plt.show()
#Plot_vs_h_c_diff_n_TI_condifence()

def Plot_vs_h_c_diff_n_TI_std():
    '''
    Plot for different TI numbers, a normalized damping graph vs Coupling strength h_c
    :return: a plot of plots
    '''
    n_PXP= n_PXP_feed
    T_start = T_start_feed
    T_max = T_max_feed
    T_step = T_step_feed
    Start_cutoff = Start_cutoff_feed #implement DONT THINK IT IS HERE!
    End_cutoff = End_cutoff_feed #implement DONT THINK IT IS HERE!
    h_c_max = h_c_max_feed
    h_c_step = h_c_step_feed
    n_TI_start = n_PXP
    n_TI_max = n_TI_max_feed
    n_TI= np.arange(n_TI_start,n_TI_max+1,1)
    h_c= np.arange(0, h_c_max+h_c_step, h_c_step)
    data_ave= np.empty((len(h_c)))
    data_errors= np.empty((len(h_c)))
    cmap = cm.get_cmap('plasma',int(n_TI_max-5)) #length of colormap should be about x2 of the number of plots
    for j in np.nditer(n_TI):
        count = 0
        for i in np.nditer(h_c):
            data_ave[count]=np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_ave_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP,Start_cutoff,End_cutoff,n_PXP,int(j),np.round(i,1),Start_cutoff,End_cutoff))
            #print(data_ave)
            data_errors[count]=np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_errors_std_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP,Start_cutoff,End_cutoff,n_PXP,int(j),np.round(i,1),Start_cutoff,End_cutoff))
            #print(data_errors)
            count=count+1
        data_errors_fin = data_errors/data_ave[0] #Scaled errors as to gamma/gamma_0
        #print('data_errors',data_errors)
        scaled_data_ave= data_ave/data_ave[0] #Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin, color=cmap.colors[int(j-n_TI[0])], marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3, label= '{} TI atoms'.format(int(j)))
        plt.fill_between(h_c[:], data_errors_fin, data_errors_fin)
        print(data_ave[11])
    plt.title(r'$Z_i$ coupling normalized damping for {} PXP atoms vs $h_c$, cutoff {}-{}'.format(n_PXP,Start_cutoff,End_cutoff))
    plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
    plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
    plt.legend()
    plt.savefig('Figures/True_X_i_Osc/Z_i_damping_{}_PXP_vs_h_c_{}_std_err_cutoff_{}_{}.png'.format(n_PXP,h_c_max,Start_cutoff,End_cutoff))
    return plt.show()
#Plot_vs_h_c_diff_n_TI_std()

def Plot_vs_h_c_diff_n_TI_Rel_Err_new():
    '''
    Plot for different TI numbers, a normalized damping graph vs Coupling strength h_c with error from pcov
    :return: a plot of plots with error from pcov
    '''
    n_PXP = n_PXP_feed
    T_start = T_start_feed
    T_max = T_max_feed
    T_step = T_step_feed
    Start_cutoff = Start_cutoff_feed  # implement DONT THINK IT IS HERE!
    End_cutoff = End_cutoff_feed  # implement DONT THINK IT IS HERE!
    h_c_max = h_c_max_feed
    h_c_step = h_c_step_feed
    n_TI_start = n_PXP
    n_TI_max = n_TI_max_feed
    n_TI= np.arange(n_TI_start,n_TI_max+1,1)
    h_c= np.arange(0, h_c_max+h_c_step, h_c_step)
    data_ave = np.empty((len(h_c)))
    data_errors = np.empty((len(h_c)))
    cmap = cm.get_cmap('plasma', int(n_TI_max - 5))  # length of colormap should be about x2 of the number of plots
    for j in np.nditer(n_TI):
        count=0
        # for i in np.nditer(h_c): # X_i CASE
        #     data_ave[int(np.round(i, 1) * 10)] = np.load('PXP_{}_True_X_i_Gammas_cutoff_{}_{}/Gamma_ave_True_X_i_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP, Start_cutoff, End_cutoff,n_PXP, int(j), np.round(i, 1),Start_cutoff, End_cutoff))
        #     # print(data_ave)
        #     data_errors[int(np.round(i, 1) * 10)] = np.load('PXP_{}_True_X_i_Gammas_cutoff_{}_{}/Gamma_error_ave_True_X_i_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP, Start_cutoff,End_cutoff, n_PXP,int(j), np.round(i, 1),Start_cutoff,End_cutoff))
        #     # print(data_errors)
        for i in np.nditer(h_c): # Z_i CASE
            data_ave[count] = np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_ave_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP, Start_cutoff, End_cutoff,n_PXP, int(j), np.round(i, 1),Start_cutoff, End_cutoff))
            # print(data_ave)
            data_errors[count] = np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_error_ave_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP, Start_cutoff,End_cutoff, n_PXP,int(j), np.round(i, 1),Start_cutoff,End_cutoff))
            # print(data_errors)
            count=count+1
        data_errors_fin = data_errors / (data_ave[0])  # Scaled errors as to gamma/gamma_0 - STD/2
        # print('data_errors',data_errors)
        scaled_data_ave = data_ave / data_ave[0]  # Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin, color=cmap.colors[int(j - n_TI[0])], marker='s',
                     markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3,
                     label='{} TI atoms'.format(int(j)))
        plt.fill_between(h_c[:], data_errors_fin, data_errors_fin)
        print(data_ave[11])
    plt.title(r'$Z_i$ coupling normalized damping for {} PXP atoms vs $h_c$, cutoff {}-{}'.format(n_PXP, Start_cutoff,End_cutoff))
    #plt.title(r'$X_i$ coupling normalized damping for {} PXP atoms vs $h_c$, cutoff {}-{}'.format(n_PXP, Start_cutoff,End_cutoff))
    plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
    plt.xlabel(r'Coupling Strength $h_c$', fontsize=12)
    plt.legend()
    #plt.savefig('Figures/True_X_i_Osc/Z_i_damping_{}_PXP_vs_h_c_{}_Rel_err_new_{}_{}.png'.format(n_PXP, h_c_max,Start_cutoff,End_cutoff)) #Z_i case
    #plt.savefig('Figures/True_X_i_Osc/X_i_damping_{}_PXP_vs_h_c_{}_Rel_err_new_{}_{}.png'.format(n_PXP, h_c_max,Start_cutoff,End_cutoff)) #X_i case
    return plt.show()
#Plot_vs_h_c_diff_n_TI_Rel_Err_new()



################################################################################################################################################################################
##########################                                                  Fast Gamma FFT CHECK                                                                    ##################
################################################################################################################################################################################

def Lorentzian_function(omega, omega_0, gamma, amp):
    '''
    Lorentzian function shape definer (variables and parameters)
    :param omega: Frequency variable
    :param omega_0: Frequency of oscillations = offset of delta function point
    :param gamma: damping
    :return: Lorentzian "shape" (scalar, just the f(x)= output of lorentzian)
    '''
    return np.divide(amp * gamma, (np.divide(1,2)*gamma)**2 + (omega-omega_0)**2)

def FFT_Compare_fast_plt(n_PXP_1, n_PXP_2, n_TI_1,n_TI_2, h_c_1,h_c_2,fourier_cutoff_div,Optimal_start,Optimal_end,start_cutoff_plt,end_cutoff_plt,T_start,T_max,T_step,sample_1,sample_2):
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
    sandwich_O_z_1_sys=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_1,n_TI_1,T_max,T_step,np.round(h_c_1,2),n_PXP_1,n_TI_1,np.round(h_c_1,2),T_max,T_step,sample_1))
    sandwich_O_z_2_sys=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_2,n_TI_2,T_max,T_step,np.round(h_c_2,2),n_PXP_2,n_TI_2,np.round(h_c_2,2),T_max,T_step,sample_2))
    Fourier_components_O_z_1= rfft(sandwich_O_z_1_sys[:fourier_cutoff_div])
    Fourier_components_O_z_2 = rfft(sandwich_O_z_2_sys[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    popt_1, pcov_1 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    popt_2, pcov_2 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_2[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    sandwich_O_z_gamma_0_sys_1=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_1,n_TI_1,T_max,T_step,0.0,n_PXP_1,n_TI_1,0.0,T_max,T_step,sample_1))
    Fourier_components_O_z_gamma_0_sys_1=rfft(sandwich_O_z_gamma_0_sys_1[:fourier_cutoff_div])
    popt_gamma_0_sys_1,pcov_gamma_0_sys_1= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_1[Optimal_start:Optimal_end]))
    sandwich_O_z_gamma_0_sys_2=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_2,n_TI_2,T_max,T_step,0.0,n_PXP_2,n_TI_2,0.0,T_max,T_step,sample_2))
    Fourier_components_O_z_gamma_0_sys_2=rfft(sandwich_O_z_gamma_0_sys_2[:fourier_cutoff_div])
    popt_gamma_0_sys_2,pcov_gamma_0_sys_2= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_2[Optimal_start:Optimal_end]))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_1[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$'.format(n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_2[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$'.format(n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_1),linestyle='dashed', marker='o', markersize=1,label=r'Lorentzian fit Blue $\gamma={}$, $\Delta\gamma={}$'.format(np.round(popt_1[1]/popt_gamma_0_sys_1[1],5),np.round(pcov_1[1,1]/popt_gamma_0_sys_1[1],8)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_2),linestyle='dashed', marker='o', markersize=1,label=r'Lorentzian fit Orange $\gamma={}$,  $\Delta\gamma={}$'.format(np.round(popt_2[1]/popt_gamma_0_sys_2[1],5),np.round(pcov_2[1,1]/popt_gamma_0_sys_2[1],8)))
    plt.legend()
    plt.title('Comparison of Fourier signals ZZ coupling')
    plt.ylim(0,70)
    plt.xlabel('Frequency')
    plt.ylabel('Fourier Transformed Signal Amplitude')
    return plt.show()


def FFT_Gamma_plot_fast(n_PXP, n_TI, h_c_max,h_c_step,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample):
    '''
    Plots gamma as a function of coupling for given number of atoms ZZ coupling for saved oscillation files!!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c_max: max value of h_c array
    :param h_c_step: step size between the different h_c's
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot of gamma vs h_c strength
    '''
    time=np.linspace(T_start,T_max,T_step)
    h_c_array= np.arange(0.0,h_c_max,h_c_step)
    gamma_array= np.zeros(len(h_c_array))
    gamma_pcov_array= np.zeros(len(h_c_array))
    count=0
    Freq = rfftfreq(fourier_cutoff_div, d=((T_max - T_start)/ T_step))  # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    sandwich_O_z_gamma_0_sys = np.load('PXP_{}_TI_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,0.0,n_PXP,n_TI,0.0,sample))
    Fourier_components_O_z_gamma_0_sys = rfft(sandwich_O_z_gamma_0_sys[:fourier_cutoff_div])
    popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_gamma_0_sys[Optimal_start:Optimal_end]))
    for h_c in np.nditer(h_c_array):
        sandwich_O_z_sys=np.load('PXP_{}_TI_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2),n_PXP,n_TI,np.round(h_c,2),sample))
        Fourier_components_O_z_1= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
        popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
        gamma_array[count]=popt[1]/popt_gamma_0_sys[1]
        gamma_pcov_array[count]=pcov[1,1]/popt_gamma_0_sys[1]
        count=count+1
    plt.plot(h_c_array, gamma_array, marker='o', markersize=1)
    plt.title('Damping coef vs Coup str, ZZ Coup for {} PXP,{} TI, Time {} Steps {}'.format(n_PXP,n_TI,T_max,T_step))
    plt.xlabel('Coupling Strength')
    plt.ylabel('Damping Coefficient (Normalized)')
    #plt.ylim(0.88,2)
    return plt.show()

def FFT_True_X_i_Gamma_plot_fast(n_PXP, n_TI, h_c_max,h_c_step,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample):
    '''
    Plots gamma as a function of coupling for given number of atoms XX coupling for saved oscillation files!!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c_max: max value of h_c array
    :param h_c_step: step size between the different h_c's
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot of gamma vs h_c strength
    '''
    time=np.linspace(T_start,T_max,T_step)
    h_c_array= np.arange(0.0,h_c_max,h_c_step)
    gamma_array= np.zeros(len(h_c_array))
    gamma_pcov_array= np.zeros(len(h_c_array))
    count=0
    Freq = rfftfreq(fourier_cutoff_div, d=((T_max - T_start)/ T_step))  # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    sandwich_O_z_gamma_0_sys = np.load('PXP_{}_TI_{}_True_X_i/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,0.0,n_PXP,n_TI,0.0,sample))
    Fourier_components_O_z_gamma_0_sys = rfft(sandwich_O_z_gamma_0_sys[:fourier_cutoff_div])
    popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_gamma_0_sys[Optimal_start:Optimal_end]))
    for h_c in np.nditer(h_c_array):
        sandwich_O_z_sys=np.load('PXP_{}_TI_{}_True_X_i/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,np.round(h_c,2),n_PXP,n_TI,np.round(h_c,2),sample))
        Fourier_components_O_z_1= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
        popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
        gamma_array[count]=popt[1]/popt_gamma_0_sys[1]
        gamma_pcov_array[count]=pcov[1,1]/popt_gamma_0_sys[1]
        count=count+1
    plt.plot(h_c_array, gamma_array, marker='o', markersize=1)
    plt.title('Damping coef vs Coup str, $XX$ Coup for {} PXP,{} TI, Time {} Steps {}'.format(n_PXP,n_TI,T_max,T_step))
    plt.xlabel('Coupling Strength')
    plt.ylabel('Damping Coefficient (Normalized)')
    #plt.ylim(0.88,2)
    return plt.show()

def Inverse_FFT_Signal_Compare_plt(n_PXP, n_TI, h_c,time_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample):
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
    sandwich_O_z_sys=np.load('PXP_{}_TI_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,h_c,n_PXP,n_TI,h_c,sample))
    Fourier_components_O_z= rfft(sandwich_O_z_sys[:time_cutoff_div])
    Freq = rfftfreq(time_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    Inverse_Fourier_array=np.zeros(len(Freq))
    popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    Inverse_Fourier_array[Optimal_start:Optimal_end]=Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt)
    inverse_transform_O_z_fit= irfft(Inverse_Fourier_array,len(time[:time_cutoff_div]))
    inverse_transform_O_z_full=irfft(np.abs(Fourier_components_O_z),len(time[:time_cutoff_div]))
    offset=np.mean(inverse_transform_O_z_full)
    plt.plot(time[:time_cutoff_div], inverse_transform_O_z_full, marker='o', markersize=3,label=r'Original Time series by inversion')
    plt.plot(time[:time_cutoff_div],offset+inverse_transform_O_z_fit, marker='o', markersize=1,label=r'Inverse transformed fitted signal')
    #plt.plot(time[:time_cutoff_div], sandwich_O_z_sys[:time_cutoff_div], marker='o', markersize=3,label=r'Original Time series')
    plt.legend()
    plt.title('Comparison of Time series to inverse transform of fit ZZ coupling')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.xlim(0,50)
    return plt.show()

def Inverse_FFT_Signal_Compare_True_X_i_plt(n_PXP, n_TI, h_c,time_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample):
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
    sandwich_O_z_sys=np.load('PXP_{}_TI_{}_True_X_i/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,h_c,n_PXP,n_TI,h_c,sample))
    Fourier_components_O_z= rfft(sandwich_O_z_sys[:time_cutoff_div])
    Freq = rfftfreq(time_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    Inverse_Fourier_array=np.zeros(len(Freq))
    popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    Inverse_Fourier_array[Optimal_start:Optimal_end]=Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt)
    inverse_transform_O_z_fit= irfft(Inverse_Fourier_array,len(time[:time_cutoff_div]))
    inverse_transform_O_z_full=irfft(np.abs(Fourier_components_O_z),len(time[:time_cutoff_div]))
    offset=np.mean(inverse_transform_O_z_full)
    plt.plot(time[:time_cutoff_div], inverse_transform_O_z_full, marker='o', markersize=3,label=r'Original Time series by inversion ')
    plt.plot(time[:time_cutoff_div],offset+inverse_transform_O_z_fit, marker='o', markersize=1,label=r'Inverse transformed fitted signal')
    #plt.plot(time[:time_cutoff_div], sandwich_O_z_sys[:time_cutoff_div], marker='o', markersize=3,label=r'Original Time series')
    plt.legend()
    plt.title('Comparison of Time series to inverse transform of fit XX coupling')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.xlim(0,50)
    return plt.show()

# def FFT_Gamma_All_Realizations(n_PXP, n_TI, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample_max,Bootstrap_num):
#     '''
#     Plots gamma as a function of coupling for given number of atoms ZZ coupling, averaged upon realizations
#     :param n_PXP: Size of PXP chain (atoms)
#     :param n_TI: Size of TI chain (atoms)
#     :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
#     :param Optimal_start: start cutoff for fourier signal FIT
#     :param Optimal_end: end cutoff for fourier signal FIT
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :param Sample_max: maximum number of samples
#     :return: plot of gamma vs h_c strength
#     '''
#     time=np.linspace(T_start,T_max,T_step)
#     h_c_array= np.arange(0.0,0.6,0.05)
#     h_c_array_fin=np.append(h_c_array,np.arange(0.6,h_c_max,0.1))
#     gamma_array= np.zeros((sample_max,len(h_c_array_fin)))
#     gamma_pcov_array= np.zeros((sample_max,len(h_c_array_fin)))
#     gamma_array_fin=np.zeros(len(h_c_array_fin))
#     std=np.zeros(len(h_c_array_fin))
#     count=0
#     for h_c in np.nditer(h_c_array_fin):
#         for i in range(1,sample_max+1):
#             sandwich_O_z_gamma_0_sys = np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP,n_TI,T_max,T_step,0.0,n_PXP,n_TI,0.0,T_max,T_step,i))
#             Fourier_components_O_z_gamma_0_sys = rfft(sandwich_O_z_gamma_0_sys[:fourier_cutoff_div])
#             Freq = rfftfreq(fourier_cutoff_div, d=(T_max / T_step))  # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
#             popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_gamma_0_sys[Optimal_start:Optimal_end]))
#             sandwich_O_z_sys=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP,n_TI,T_max,T_step,np.round(h_c,2),n_PXP,n_TI,np.round(h_c,2),T_max,T_step,i))
#             Fourier_components_O_z_1= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
#             popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
#             gamma_array[i-1,count]=popt[1]/popt_gamma_0_sys[1]
#             gamma_pcov_array[i-1,count]=pcov[1,1]/popt_gamma_0_sys[1]
#         bootstrap = np.random.choice(gamma_array[:,count], (2*sample_max,Bootstrap_num),replace=True)# creates [(sample_max) x (Bootstrap_num)] matrix of rows of randomly sampled numbers (with return) from the original sample
#         bootstrap_ave = np.mean(bootstrap, axis=0)  #Vector of averages!! from randomly pulling numbers from 100 realizations of one time instance
#         std[count] = np.std(bootstrap_ave)  # Standard deviation of the different means obtained with bootstrapping, can be divided by sqrt(N) due to averaging
#         gamma_array_fin[count]=np.mean(bootstrap_ave)
#         count=count+1
#     gamma_pcov_array_fin=np.mean(gamma_pcov_array,axis=0)+std
#     plt.errorbar(h_c_array_fin, gamma_array_fin, yerr=gamma_pcov_array_fin, marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3)
#     return h_c_array_fin, gamma_array_fin, gamma_pcov_array_fin
#
# def FFT_Gamma_True_X_i_All_Realizations(n_PXP, n_TI, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample_max,Bootstrap_num):
#     '''
#     Plots gamma as a function of coupling for given number of atoms XX coupling, averaged upon realizations
#     :param n_PXP: Size of PXP chain (atoms)
#     :param n_TI: Size of TI chain (atoms)
#     :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
#     :param Optimal_start: start cutoff for fourier signal FIT
#     :param Optimal_end: end cutoff for fourier signal FIT
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :param Sample_max: maximum number of samples
#     :return: plot of gamma vs h_c strength
#     '''
#     time=np.linspace(T_start,T_max,T_step)
#     h_c_array= np.arange(0.0,0.6,0.05)
#     h_c_array_fin=np.append(h_c_array,np.arange(0.6,h_c_max,0.1))
#     gamma_array= np.zeros((sample_max,len(h_c_array_fin)))
#     gamma_pcov_array= np.zeros((sample_max,len(h_c_array_fin)))
#     gamma_array_fin=np.zeros(len(h_c_array_fin))
#     std=np.zeros(len(h_c_array_fin))
#     count=0
#     for h_c in np.nditer(h_c_array_fin):
#         for i in range(1,sample_max+1):
#             sandwich_O_z_gamma_0_sys = np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP,n_TI,T_max,T_step,0.0,n_PXP,n_TI,0.0,T_max,T_step,i))
#             Fourier_components_O_z_gamma_0_sys = rfft(sandwich_O_z_gamma_0_sys[:fourier_cutoff_div])
#             Freq = rfftfreq(fourier_cutoff_div, d=(T_max / T_step))  # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
#             popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_gamma_0_sys[Optimal_start:Optimal_end]))
#             sandwich_O_z_sys=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP,n_TI,T_max,T_step,np.round(h_c,2),n_PXP,n_TI,np.round(h_c,2),T_max,T_step,i))
#             Fourier_components_O_z_1= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
#             popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
#             gamma_array[i-1,count]=popt[1]/popt_gamma_0_sys[1]
#             gamma_pcov_array[i-1,count]=pcov[1,1]/popt_gamma_0_sys[1]
#         bootstrap = np.random.choice(gamma_array[:,count], (2*sample_max,Bootstrap_num),replace=True)# creates [(sample_max) x (Bootstrap_num)] matrix of rows of randomly sampled numbers (with return) from the original sample
#         bootstrap_ave = np.mean(bootstrap, axis=0)  #Vector of averages!! from randomly pulling numbers from 100 realizations of one time instance
#         std[count] = np.std(bootstrap_ave)  # Standard deviation of the different means obtained with bootstrapping, can be divided by sqrt(N) due to averaging
#         gamma_array_fin[count]=np.mean(bootstrap_ave)
#         count=count+1
#     gamma_pcov_array_fin=np.mean(gamma_pcov_array,axis=0)+std
#     return h_c_array_fin, gamma_array_fin, gamma_pcov_array_fin
#
# def FFT_Gamma_All_Realizations_plt(n_PXP, n_TI, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample_max,Bootstrap_num):
#     '''
#     Plots gamma as a function of coupling for given number of atoms ZZ coupling, averaged upon realizations
#     :param n_PXP: Size of PXP chain (atoms)
#     :param n_TI: Size of TI chain (atoms)
#     :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
#     :param Optimal_start: start cutoff for fourier signal FIT
#     :param Optimal_end: end cutoff for fourier signal FIT
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :param Sample_max: maximum number of samples
#     :param Bootstrap_num: number of bootstrapped datasets (sample SETS!!)
#     :return: plot of gamma vs h_c strength
#     '''
#     time=np.linspace(T_start,T_max,T_step)
#     h_c_array= np.arange(0.0,0.6,0.05)
#     h_c_array_fin=np.append(h_c_array,np.arange(0.6,h_c_max,0.1))
#     gamma_array= np.zeros((sample_max,len(h_c_array_fin)))
#     gamma_pcov_array= np.zeros((sample_max,len(h_c_array_fin)))
#     gamma_array_fin=np.zeros(len(h_c_array_fin))
#     std=np.zeros(len(h_c_array_fin))
#     count=0
#     for h_c in np.nditer(h_c_array_fin):
#         for i in range(1,sample_max+1):
#             sandwich_O_z_gamma_0_sys = np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP,n_TI,T_max,T_step,0.0,n_PXP,n_TI,0.0,T_max,T_step,i))
#             Fourier_components_O_z_gamma_0_sys = rfft(sandwich_O_z_gamma_0_sys[:fourier_cutoff_div])
#             Freq = rfftfreq(fourier_cutoff_div, d=(T_max / T_step))  # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
#             popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_gamma_0_sys[Optimal_start:Optimal_end]))
#             sandwich_O_z_sys=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP,n_TI,T_max,T_step,np.round(h_c,2),n_PXP,n_TI,np.round(h_c,2),T_max,T_step,i))
#             Fourier_components_O_z_1= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
#             popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
#             gamma_array[i-1,count]=popt[1]/popt_gamma_0_sys[1]
#             gamma_pcov_array[i-1,count]=pcov[1,1]/popt_gamma_0_sys[1]
#         bootstrap = np.random.choice(gamma_array[:,count], (2*sample_max,Bootstrap_num),replace=True)# creates [(sample_max) x (Bootstrap_num)] matrix of rows of randomly sampled numbers (with return) from the original sample
#         bootstrap_ave = np.mean(bootstrap, axis=0)  #Vector of averages!! from randomly pulling numbers from 100 realizations of one time instance
#         std[count] = np.std(bootstrap_ave)  # Standard deviation of the different means obtained with bootstrapping, can be divided by sqrt(N) due to averaging
#         gamma_array_fin[count]=np.mean(bootstrap_ave)
#         count=count+1
#     gamma_pcov_array_fin=np.mean(gamma_pcov_array,axis=0)+std
#     plt.errorbar(h_c_array_fin, gamma_array_fin, yerr=gamma_pcov_array_fin, marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3)
#     # plt.plot(h_c_array, gamma_array_fin, marker='o', markersize=1)
#     plt.title('Damping coef vs Coup str, $ZZ$ Coup for {} PXP,{} TI, Time {} Steps {}'.format(n_PXP,n_TI,T_max,T_step))
#     plt.xlabel('Coupling Strength')
#     plt.ylabel('Damping Coefficient (Normalized)')
#     plt.ylim(0.88,2)
#     return plt.show()
#
# def FFT_Gamma_True_X_i_All_Realizations_plt(n_PXP, n_TI, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample_max,Bootstrap_num):
#     '''
#     Plots gamma as a function of coupling for given number of atoms XX coupling, averaged upon realizations
#     :param n_PXP: Size of PXP chain (atoms)
#     :param n_TI: Size of TI chain (atoms)
#     :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
#     :param Optimal_start: start cutoff for fourier signal FIT
#     :param Optimal_end: end cutoff for fourier signal FIT
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :param Sample_max: maximum number of samples
#     :param Bootstrap_num: number of bootstrapped datasets (sample SETS!!)
#     :return: plot of gamma vs h_c strength
#     '''
#     time=np.linspace(T_start,T_max,T_step)
#     h_c_array= np.arange(0.0,0.6,0.05)
#     h_c_array_fin=np.append(h_c_array,np.arange(0.6,h_c_max,0.1))
#     gamma_array= np.zeros((sample_max,len(h_c_array_fin)))
#     gamma_pcov_array= np.zeros((sample_max,len(h_c_array_fin)))
#     gamma_array_fin=np.zeros(len(h_c_array_fin))
#     std=np.zeros(len(h_c_array_fin))
#     count=0
#     for h_c in np.nditer(h_c_array_fin):
#         for i in range(1,sample_max+1):
#             sandwich_O_z_gamma_0_sys = np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP,n_TI,T_max,T_step,0.0,n_PXP,n_TI,0.0,T_max,T_step,i))
#             Fourier_components_O_z_gamma_0_sys = rfft(sandwich_O_z_gamma_0_sys[:fourier_cutoff_div])
#             Freq = rfftfreq(fourier_cutoff_div, d=(T_max / T_step))  # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
#             popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_gamma_0_sys[Optimal_start:Optimal_end]))
#             sandwich_O_z_sys=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP,n_TI,T_max,T_step,np.round(h_c,2),n_PXP,n_TI,np.round(h_c,2),T_max,T_step,i))
#             Fourier_components_O_z_1= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
#             popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
#             gamma_array[i-1,count]=popt[1]/popt_gamma_0_sys[1]
#             gamma_pcov_array[i-1,count]=pcov[1,1]/popt_gamma_0_sys[1]
#         bootstrap = np.random.choice(gamma_array[:,count], (2*sample_max,Bootstrap_num),replace=True)# creates [(sample_max) x (Bootstrap_num)] matrix of rows of randomly sampled numbers (with return) from the original sample
#         bootstrap_ave = np.mean(bootstrap, axis=0)  #Vector of averages!! from randomly pulling numbers from 100 realizations of one time instance
#         std[count] = np.std(bootstrap_ave)  # Standard deviation of the different means obtained with bootstrapping, can be divided by sqrt(N) due to averaging
#         gamma_array_fin[count]=np.mean(bootstrap_ave)
#         count=count+1
#     gamma_pcov_array_fin=np.mean(gamma_pcov_array,axis=0)+std
#     plt.errorbar(h_c_array_fin, gamma_array_fin, yerr=gamma_pcov_array_fin, marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3)
#     plt.fill_between(h_c_array_fin, gamma_pcov_array_fin, gamma_pcov_array_fin)
#     # plt.plot(h_c_array, gamma_array_fin, marker='o', markersize=1)
#     plt.title('Damping coef vs Coup str, $XX$ Coup for {} PXP,{} TI, Time {} Steps {}'.format(n_PXP,n_TI,T_max,T_step))
#     plt.xlabel('Coupling Strength')
#     plt.ylabel('Damping Coefficient (Normalized)')
#     plt.ylim(0.88,1.5)
#     return plt.show()
#
# def FFT_Gamma_All_Realizations_Var_TI_plt(n_PXP, n_TI_max, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample_max,Bootstrap_num):
#     '''
#     Plots gamma as a function of coupling for given number of PXP atoms, and different TI atoms (8-n_TI_max) ZZ coupling, averaged upon realizations
#     :param n_PXP: Size of PXP chain (atoms)
#     :param n_TI_max: maximum size of TI chain (atoms)
#     :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
#     :param Optimal_start: start cutoff for fourier signal FIT
#     :param Optimal_end: end cutoff for fourier signal FIT
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :param Sample_max: maximum number of samples
#     :param Bootstrap_num: number of bootstrapped datasets (sample SETS!!)
#     :return: plot of gamma vs h_c strength
#     '''
#     cmap = cm.get_cmap('plasma', int(n_TI_max - 5))  # length of colormap should be about x2 of the number of plots
#     for j in range(n_PXP,n_TI_max+1):
#         h_c_array, gamma_array, gamma_error_array= FFT_Gamma_All_Realizations(n_PXP, j, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample_max,Bootstrap_num)
#         plt.errorbar(h_c_array, gamma_array, yerr=gamma_error_array, color=cmap.colors[int(j-n_PXP)], marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3, label= '{} TI atoms'.format(int(j)))
#         plt.fill_between(h_c_array, gamma_error_array, gamma_error_array)
#     plt.title(r'Normalized Damping for {} PXP atoms vs $h_c$, $ZZ$ coupling'.format(n_PXP))
#     plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
#     plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
#     plt.legend()
#     plt.xlim(0,h_c_max)
#     plt.ylim(0.9,4)
#     #plt.savefig('Figures/New_Osc_Short_T/Normalized_Damping_For_{}_PXP_vs_h_c_ZZ_Coup_{}_T_Cut.png'.format(n_PXP,int((fourier_cutoff_div*40)/500)))
#     return plt.show()
#
# def FFT_Gamma_True_X_i_All_Realizations_Var_TI_plt(n_PXP, n_TI_max, h_c_max,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step,sample_max,Bootstrap_num):
#     '''
#     Plots gamma as a function of coupling for given number of PXP atoms, and different TI atoms (8-n_TI_max) XX coupling, averaged upon realizations
#     :param n_PXP: Size of PXP chain (atoms)
#     :param n_TI_max: maximum size of TI chain (atoms)
#     :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
#     :param Optimal_start: start cutoff for fourier signal FIT
#     :param Optimal_end: end cutoff for fourier signal FIT
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :param Sample_max: maximum number of samples
#     :param Bootstrap_num: number of bootstrapped datasets (sample SETS!!)
#     :return: plot of gamma vs h_c strength
#     '''
#     cmap = cm.get_cmap('plasma', int(n_TI_max - 5))  # length of colormap should be about x2 of the number of plots
#     for j in range(n_PXP, n_TI_max+1):
#         h_c_array, gamma_array, gamma_error_array = FFT_Gamma_True_X_i_All_Realizations(n_PXP, j, h_c_max,fourier_cutoff_div,Optimal_start, Optimal_end,T_start, T_max, T_step,sample_max, Bootstrap_num)
#         plt.errorbar(h_c_array, gamma_array, yerr=gamma_error_array, color=cmap.colors[int(j-n_PXP)], marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3, label= '{} TI atoms'.format(int(j)))
#         plt.fill_between(h_c_array, gamma_error_array, gamma_error_array)
#     plt.title(r'Normalized Damping for {} PXP atoms vs $h_c$, $XX$ coupling'.format(n_PXP))
#     plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
#     plt.xlabel(r'Coupling Strength $h_c$', fontsize=12)
#     plt.legend()
#     plt.xlim(0,h_c_max)
#     plt.ylim(0.9,4)
#     #plt.savefig('Figures/New_Osc_Short_T/Normalized_Damping_For_{}_PXP_vs_h_c_XX_Coup_{}_T_Cut.png'.format(n_PXP,int((fourier_cutoff_div*40)/500)))
#     return plt.show()
