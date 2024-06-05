import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from Coupling_To_Bath import *
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
#import PXP_E_B_E_Sparse as Ebe

#### the optimal cuts for fit 105 and 235 are for T=400, for other times the formula is x/400 * T_max_new (x/(400/T_max_new)) ####

#T_start=0
#T_max=400
#T_step= 1000
#n_PXP=9
#n_TI=10
#h_c=3.0
#Start_cutoff=105
#End_cutoff= 235
#for sample in range(1,99):
#sample=1

def Lorentzian_function(omega, omega_0, gamma, amp):
    '''
    Lorentzian function shape definer (variables and parameters)
    :param omega: Frequency variable
    :param omega_0: Frequency of oscillations = offset of delta function point
    :param gamma: damping
    :return: Lorentzian "shape" (scalar, just the f(x)= output of lorentzian)
    '''
    return np.divide(amp * gamma, (np.divide(1,2)*gamma)**2 + (omega-omega_0)**2)

# def Osc_graph(T_start, T_max, T_step):
#     '''
#     Gets positive frequency (absolute value of) fourier components of the propagation signal and positive frequencies
#     :param n_PXP: Size of PXP chain (atoms)
#     :param n_TI: Size of TI chain (atoms)
#     :param Initialstate:  Initial Vector state we would like to propagate
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :return: 2 arrays (Positive freq, positive freq fourier components)
#     '''
#     time= np.linspace(T_start, T_max,T_step)
#     #VecProp = np.load('1.4_check/Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,h_c,sample))
#     VecProp = np.load('PXP_{}_Osc_Ave/Sparse_time_propagation_ave_{}_{}_{}.npy'.format(n_PXP,n_PXP,n_TI,h_c))
#     print(VecProp.round(4))
#     plt.plot(time,VecProp.round(4)[:len(time)])
#     plt.show()


def FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step,fourier_cutoff_div, Height_norm,sample):
    '''
    Gets positive frequency (absolute value of) fourier components of the propagation signal and positive frequencies
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 arrays (Positive freq, positive freq fourier components)
    '''
    time= np.linspace(T_start, T_max,T_step)
    VecProp = np.load('PXP_{}_TI_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,h_c,n_PXP,n_TI,h_c,sample))
    #print(VecProp.round(4))
    #plt.plot(time,VecProp.round(4)[:len(time)])
    #plt.show()
    Fourier_components= rfft(VecProp[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=((T_max - T_start)/ T_step)) # Freq * T_max = integer that multiplies 2pi
    return Freq, (Height_norm * np.abs(Fourier_components))


def FFT_X_i(n_PXP, n_TI, h_c,T_start, T_max, T_step, fourier_cutoff_div,Height_norm,sample):
    '''
    Gets positive frequency (absolute value of) fourier components of the propagation signal and positive frequencies
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 arrays (Positive freq, positive freq fourier components)
    '''
    time= np.linspace(T_start, T_max,T_step,endpoint= True)
    #VecProp = np.load('1.4_check/Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,h_c,sample))
    VecProp = np.load('PXP_{}_TI_{}_True_X_i/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,h_c,n_PXP,n_TI,h_c,sample))
    #plt.plot(time,VecProp.round(4)[:len(time)])
    #plt.show()
    Fourier_components= rfft(VecProp[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=((T_max - T_start)/ T_step)) # Freq * T_max = integer that multiplies 2pi
    return Freq, (Height_norm * np.abs(Fourier_components))

def Lorentzian_curvefit(n_PXP, n_TI, h_c,T_start, T_max, T_step, Start_cutoff, End_cutoff,fourier_cutoff_div,sample, Height_norm=1):
    '''
    Fits lorentzian function to Fourier signal, returns gamma (damping coefficient)
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: low cutoff of freq for best fit
    :param End_cutoff: high cutoff of freq for best fit
    :return: optimal coefficients (in the order: Omega_0, gamma, Amplitude)
    '''
    Freq, sig_func = FFT(n_PXP, n_TI, h_c,T_start, T_max, T_step,fourier_cutoff_div, Height_norm,sample)
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    return popt

def Lorentzian_curvefit_plt(n_PXP, n_TI, h_c,T_start, T_max, T_step, Start_cutoff, End_cutoff,fourier_cutoff_div, sample,Height_norm=1):
    #NEW TERMS 400, 1000
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
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: low cutoff of freq for best fit
    :param End_cutoff: high cutoff of freq for best fit
    :return: a plot of data (blue) and fit (red)
    '''
    Freq, sig_func = FFT(n_PXP, n_TI, h_c,T_start, T_max, T_step,fourier_cutoff_div, Height_norm,sample)
    plt.plot(Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff], marker='o', markersize=3,
             color='b', label='data')
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    residuals = sig_func[Start_cutoff:End_cutoff] - Lorentzian_function(Freq[Start_cutoff:End_cutoff],*popt)
    res_sum_squares = np.sum(residuals**2)
    tot_sum_squares= np.sum((sig_func[Start_cutoff:End_cutoff]-np.mean(sig_func[Start_cutoff:End_cutoff]))**2)
    r_squared= 1-(res_sum_squares/tot_sum_squares)
    Rel_std_err=int(np.divide(np.sqrt(np.diag(pcov)[1]),popt[1])*100) #relative standard error of gamma parameter!!
    plt.plot(Freq[Start_cutoff:End_cutoff], Lorentzian_function(Freq[Start_cutoff:End_cutoff],*popt),'r-',
         label=r'fit: $\omega_0$={}, $\gamma$={}, Amp={},$R^2$={}, Rel std err={}%'.format(*np.round(popt,4),np.round(r_squared,4),Rel_std_err))
    plt.title(r'Frequency fit for {} PXP, {} TI, $h_c$ {}, cutoff {}-{}'.format(n_PXP,n_TI,h_c,Start_cutoff,End_cutoff))
    plt.xlabel(r'Frequency [$1/t$]')
    plt.ylabel('Amplitudes of Harmonic Functions')
    plt.legend()
    plt.ylim(0,10)
    #plt.savefig("Figures/Frequency_fit/FFT_Fit_{}_PXP_{}_TI_{}_Coup_{}-{}_cutoff.png".format(n_PXP,n_TI,h_c,Start_cutoff,End_cutoff))
    plt.show()
    return
#Lorentzian_curvefit_plt(9, 10, 4,0, 400, 1000,105, 235,1)

def Lorentzian_curvefit_True_X_i(n_PXP, n_TI, h_c,T_start, T_max, T_step, Start_cutoff, End_cutoff,fourier_cutoff_div,sample,Height_norm=1):
    '''
    Fits lorentzian function to Fourier signal, returns gamma (damping coefficient) for XX coupling!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: low cutoff of freq for best fit
    :param End_cutoff: high cutoff of freq for best fit
    :return: optimal coefficients (in the order: Omega_0, gamma, Amplitude)
    '''
    Freq, sig_func = FFT_X_i(n_PXP, n_TI, h_c,T_start, T_max, T_step,fourier_cutoff_div, Height_norm,sample)
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    return popt

def Lorentzian_curvefit_plt_True_X_i(n_PXP, n_TI, h_c,T_start, T_max, T_step, Start_cutoff, End_cutoff,fourier_cutoff_div,sample, Height_norm=1):

    #NEW TERMS 400, 1000
    # (need to increas t_max and increase t_step so that number of N_tot does not !! go a lot over t_max)
    '''
    Fits lorentzian function to transformed signal, outputs plots of data and fit for XX coupling!!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation; Bigger T_max - smaller frequencies! (N_tot/T_max top freq)
    :param T_step: time step (division) Bigger T_step - bigger frequencies! (N_tot=T_step, N_tot/T_max top freq)
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: low cutoff of freq for best fit
    :param End_cutoff: high cutoff of freq for best fit
    :return: a plot of data (blue) and fit (red)
    '''
    Freq, sig_func = FFT_X_i(n_PXP, n_TI, h_c,T_start, T_max, T_step, fourier_cutoff_div,Height_norm,sample)
    plt.plot(Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff], marker='o', markersize=3,
             color='b', label='data')
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    residuals = sig_func[Start_cutoff:End_cutoff] - Lorentzian_function(Freq[Start_cutoff:End_cutoff],*popt)
    res_sum_squares = np.sum(residuals**2)
    tot_sum_squares= np.sum((sig_func[Start_cutoff:End_cutoff]-np.mean(sig_func[Start_cutoff:End_cutoff]))**2)
    r_squared= 1-(res_sum_squares/tot_sum_squares)
    Rel_std_err=int(np.divide(np.sqrt(np.diag(pcov)[1]),popt[1])*100) #relative standard error of gamma parameter!!
    plt.plot(Freq[Start_cutoff:End_cutoff], Lorentzian_function(Freq[Start_cutoff:End_cutoff],*popt),'r-',
         label=r'fit: $\omega_0$={}, $\gamma$={}, Amp={},$R^2$={}, Rel std err={}%'.format(*np.round(popt,4),np.round(r_squared,4),Rel_std_err))
    plt.title(r'Frequency fit for {} PXP, {} TI, $h_c$ {}, cutoff {}-{}'.format(n_PXP,n_TI,h_c,Start_cutoff,End_cutoff))
    plt.xlabel(r'Frequency [$1/t$]')
    plt.ylabel('Amplitudes of Harmonic Functions')
    plt.legend()
    plt.ylim(0,10)
    #plt.savefig("Figures/Frequency_fit/FFT_Fit_{}_PXP_{}_TI_{}_Coup_{}-{}_cutoff.png".format(n_PXP,n_TI,h_c,Start_cutoff,End_cutoff))
    plt.show()
    return
#Lorentzian_curvefit_plt_True_X_i(9, 10, 4.0,0, 400, 1000,105, 235,1)

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

def FFT_True_X_i_Compare_fast_plt(n_PXP_1, n_PXP_2, n_TI_1,n_TI_2, h_c_1,h_c_2,fourier_cutoff_div,Optimal_start,Optimal_end,start_cutoff_plt,end_cutoff_plt,T_start,T_max,T_step,sample_1,sample_2):
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
    sandwich_O_z_1_sys=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_1,n_TI_1,T_max,T_step,np.round(h_c_1,2),n_PXP_1,n_TI_1,np.round(h_c_1,2),T_max,T_step,sample_1))
    sandwich_O_z_2_sys=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_2,n_TI_2,T_max,T_step,np.round(h_c_2,2),n_PXP_2,n_TI_2,np.round(h_c_2,2),T_max,T_step,sample_2))
    Fourier_components_O_z_1= rfft(sandwich_O_z_1_sys[:fourier_cutoff_div])
    Fourier_components_O_z_2 = rfft(sandwich_O_z_2_sys[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=(T_max/T_step)) # 1/T_max is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    popt_1, pcov_1 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_1[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    popt_2, pcov_2 = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_2[Optimal_start:Optimal_end]))  # popt= parameter optimal values
    sandwich_O_z_gamma_0_sys_1=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_1,n_TI_1,T_max,T_step,0.0,n_PXP_1,n_TI_1,0.0,T_max,T_step,sample_1))
    Fourier_components_O_z_gamma_0_sys_1=rfft(sandwich_O_z_gamma_0_sys_1[:fourier_cutoff_div])
    popt_gamma_0_sys_1,pcov_gamma_0_sys_1= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_1[Optimal_start:Optimal_end]))
    sandwich_O_z_gamma_0_sys_2=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_2,n_TI_2,T_max,T_step,0.0,n_PXP_2,n_TI_2,0.0,T_max,T_step,sample_2))
    Fourier_components_O_z_gamma_0_sys_2=rfft(sandwich_O_z_gamma_0_sys_2[:fourier_cutoff_div])
    popt_gamma_0_sys_2,pcov_gamma_0_sys_2= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_2[Optimal_start:Optimal_end]))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_1[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$'.format(n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_2[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$'.format(n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_1),linestyle='dashed', marker='o', markersize=1,label=r'Lorentzian fit Blue $\gamma={}$, $\Delta\gamma={}$'.format(np.round(popt_1[1]/popt_gamma_0_sys_1[1],5),np.round(pcov_1[1,1]/popt_gamma_0_sys_1[1],8)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_2),linestyle='dashed', marker='o', markersize=1,label=r'Lorentzian fit Orange $\gamma={}$,  $\Delta\gamma={}$'.format(np.round(popt_2[1]/popt_gamma_0_sys_2[1],5),np.round(pcov_2[1,1]/popt_gamma_0_sys_2[1],8)))
    plt.legend()
    plt.title('Comparison of Fourier signals XX coupling')
    plt.xlabel('Frequency')
    plt.ylabel('Fourier Transformed Signal Amplitude')
    plt.ylim(0,70)
    #plt.savefig('Fourier_Transformed_Signal_Thesis.png')
    return plt.show()

def FFT_Compare_ZZ_XX_fast_plt(n_PXP_1, n_PXP_2, n_TI_1,n_TI_2, h_c_1,h_c_2,fourier_cutoff_div,Optimal_start,Optimal_end,start_cutoff_plt,end_cutoff_plt,T_start,T_max,T_step,sample_1,sample_2):
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
    sandwich_O_z_1_sys=np.load('PXP_{}_TI_{}_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_1,n_TI_1,T_max,T_step,np.round(h_c_1,2),n_PXP_1,n_TI_1,np.round(h_c_1,2),T_max,T_step,sample_1))
    sandwich_O_z_2_sys=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_2,n_TI_2,T_max,T_step,np.round(h_c_2,2),n_PXP_2,n_TI_2,np.round(h_c_2,2),T_max,T_step,sample_2))
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
    sandwich_O_z_gamma_0_sys_2=np.load('PXP_{}_TI_{}_True_X_i_T_max_{}_Step_{}/h_c_{}/Sparse_time_propagation_True_X_i_{}_{}_{}_T_max_{}_Step_{}_sample_{}.npy'.format(n_PXP_2,n_TI_2,T_max,T_step,0.0,n_PXP_2,n_TI_2,0.0,T_max,T_step,sample_2))
    Fourier_components_O_z_gamma_0_sys_2=rfft(sandwich_O_z_gamma_0_sys_2[:fourier_cutoff_div])
    popt_gamma_0_sys_2,pcov_gamma_0_sys_2= curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],
                               np.abs(Fourier_components_O_z_gamma_0_sys_2[Optimal_start:Optimal_end]))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_1[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$ ZZ Coupling'.format(n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(Freq[start_cutoff_plt:end_cutoff_plt], np.abs(Fourier_components_O_z_2[start_cutoff_plt:end_cutoff_plt]), marker='o', markersize=3,label=r'PXP {}, TI {}, $h_c={}$ XX Coupling'.format(n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_1),linestyle='dashed', marker='o', markersize=1,label=r'ZZ (blue) $O_z$ fit $\omega_0={}$, $\gamma={}$, $\Delta\gamma={}$'.format(popt_1[0],np.round(popt_1[1]/popt_gamma_0_sys_1[1],5),np.round(pcov_1[1,1]/popt_gamma_0_sys_1[1],8)))
    plt.plot(Freq[Optimal_start:Optimal_end], Lorentzian_function(Freq[Optimal_start:Optimal_end],*popt_2),linestyle='dashed', marker='o', markersize=1,label=r'XX (Orange) $O_z$ fit $\omega_0={}$, $\gamma={}$,  $\Delta\gamma={}$'.format(popt_2[0],np.round(popt_2[1]/popt_gamma_0_sys_2[1],5),np.round(pcov_2[1,1]/popt_gamma_0_sys_2[1],8)))
    plt.legend()
    plt.title('Comparison of Fourier signals ZZ and XX Coupling')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.ylim(0,70)
    return plt.show()