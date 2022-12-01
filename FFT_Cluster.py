import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import numpy.linalg as la
#import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Coupling_To_Bath import *
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
from scipy.optimize import curve_fit
from O_z_Oscillations import *
from Cluster_Sparse_Osc_Para import *

def FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm):
    '''
    Gets positive frequency (absolute value of) fourier components of the propagation signal and positive frequencies
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (interval!!!!!)
    :return: 2 arrays (Positive freq, positive freq fourier components)
    '''
    time = np.linspace(T_start, T_max, T_step, endpoint=True)
    for i in range(1,seed_max):
        VecProp = np.load('Sparse_time_propagation_sample_{}.npy'.format(i)) #TODO
        Fourier_components= rfft(VecProp) #TODO
        Sig_size= np.size(VecProp) #TODO
        Freq = rfftfreq(Sig_size, d=T_step) # Freq * T_max = integer that multiplies 2pi
        return Freq, (Height_norm * np.abs(Fourier_components))

# def FFT_non_abs(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm):
#     '''
#     Gets positive frequency fourier components (complex!!!) of the propagation signal and positive frequencies
#     :param n_PXP: Size of PXP chain (atoms)
#     :param n_TI: Size of TI chain (atoms)
#     :param Initialstate:  Initial Vector state we would like to propagate
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (interval!!!!!)
#     :return: 2 arrays (Positive freq, positive freq fourier components)
#     '''
#     time, VecProp = Ebe.Run_Time_prop_EBE(n_PXP, n_TI, h_c ,T_start, T_max, int(T_max/T_step)) #Run_Time_prop uses time division
#     Fourier_components= rfft(VecProp)
#     Sig_size= np.size(VecProp)
#     Freq = rfftfreq(Sig_size, d=T_step) # Freq * T_max = integer that multiplies 2pi
#     return Freq, (Height_norm/Sig_size * (Fourier_components))

def Lorentzian_function(omega, omega_0, gamma, amp): #TODO Could it be on the basic
    '''
    Lorentzian function shape definer (variables and parameters)
    :param omega: Frequency variable
    :param omega_0: Frequency of oscillations = offset of delta function point
    :param gamma: damping
    :return: Lorentzian "shpae" (scalar, just the f(x)= output of lorentzian)
    '''
    return np.divide(amp * gamma, gamma**2 + (omega-omega_0)**2)

def Cluster_Lorentzian_curvefit(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Fits lorentzian function to Fourier signal, returns gamma (damping coefficient)
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (interval)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: optimal coefficients (in the order: Omega_0, gamma, Amplitude)
    '''
    Freq, sig_func = FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:], sig_func[Start_cutoff:]) # popt= parameter optimal values
    return popt

