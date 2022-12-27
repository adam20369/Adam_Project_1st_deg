import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Coupling_To_Bath import *
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
#import PXP_E_B_E_Sparse as Ebe

T_start=0
T_max=400
T_step=1000
n_PXP=9
n_TI=10
h_c=0.5
sample=2
Start_cutoff=8
End_cutoff= 1000

def Lorentzian_function(omega, omega_0, gamma, amp):
    '''
    Lorentzian function shape definer (variables and parameters)
    :param omega: Frequency variable
    :param omega_0: Frequency of oscillations = offset of delta function point
    :param gamma: damping
    :return: Lorentzian "shape" (scalar, just the f(x)= output of lorentzian)
    '''
    return np.divide(amp * gamma, gamma**2 + (omega-omega_0)**2)

def FFT(T_start, T_max, T_step, Height_norm):
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
    time= np.linspace(T_start, T_max,T_step)
    #VecProp = np.load('Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,h_c,sample))
    VecProp = np.load('PXP_{}_Osc_Ave/Sparse_time_propagation_ave_{}_{}_{}.npy'.format(n_PXP,n_PXP,n_TI,h_c))
    print(VecProp.round(4))
    plt.plot(time,VecProp.round(4))
    plt.show()
    Fourier_components= rfft(VecProp)
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=(T_max/T_step)) # Freq * T_max = integer that multiplies 2pi
    return Freq, (Height_norm * np.abs(Fourier_components))

def Lorentzian_curvefit(T_start, T_max, T_step, Start_cutoff, End_cutoff,Height_norm=1):
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
    Freq, sig_func = FFT(T_start, T_max, T_step, Height_norm)
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    return popt

def Lorentzian_curvefit_plt(T_start, T_max, T_step, Start_cutoff, End_cutoff, Height_norm=1):

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
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: a plot of data (blue) and fit (red)
    '''
    Freq, sig_func = FFT(T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff], marker='o', markersize=3,
             color='b', label='data')
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    plt.plot(Freq[Start_cutoff:End_cutoff], Lorentzian_function(Freq[Start_cutoff:End_cutoff],*popt),'r-',
         label=r'fit: $\omega_0$={}, $\gamma$={}, Amp={}'.format(*np.round(popt,4)))
    #plt.title('Frequency fit for {} PXP and {} TI atoms, Coupling strength {}'.format(n_PXP,n_TI,h_c))
    plt.xlabel(r'Frequency [$1/t$]')
    plt.ylabel('Amplitudes of Harmonic Functions')
    plt.legend()
    #plt.savefig("Figures/Frequency_fit/Freq_Fit_{}_PXP_{}_TI_{}_Coup.png".format(8,12,0.4))
    return plt.show()
Lorentzian_curvefit_plt(T_start, T_max, T_step,Start_cutoff, End_cutoff)