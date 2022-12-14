import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import numpy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Coupling_To_Bath import *
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
from scipy.optimize import curve_fit
from Cluster_Sparse_Osc_Para import *
#from O_z_Oscillations import *


def Cluster_Realizations_FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm=1):
    '''
    creates a matrix of (absolute values of) fourier components of the the different Haar propagation signals,
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: Matrix of fourier transform components (seed_max x len(Time))
    '''
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    Fourier_components= np.empty((seed_max-1,int(len(Time)/2+1)))
    for i in range(1,seed_max):
        VecProp = np.load('Sparse_time_propagation_{}_{}_{}_sample_{}.npy'.format(n_PXP,n_TI,h_c,i))
        Fourier_components[i-1,:]= rfft(VecProp)
    np.save('Fourier_components_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c), Height_norm * np.abs(Fourier_components))

#Cluster_Realizations_FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm=1)

def Cluster_FFT_Freq(T_start, T_max, T_step):
    '''
    Positive frequency components of Fourier transform
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: vector of frequency components
    '''
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    Freq = rfftfreq(len(Time), d=(T_max/T_step)) # Freq * T_max = integer that multiplies 2pi
    np.save('Frequency_T_max_{}_T_step_{}.npy'.format(T_max,T_step), Freq)

#Cluster_FFT_Freq(T_start, T_max, T_step)

def Cluster_Lorentzian_function(omega, omega_0, gamma, amp):
    '''
    Lorentzian function shape definer (variables and parameters)
    :param omega: Frequency variable
    :param omega_0: Frequency of oscillations = offset of delta function point
    :param gamma: damping
    :return: Lorentzian "shpae" (scalar, just the f(x)= output of lorentzian)
    '''
    return np.divide(amp * gamma, gamma**2 + (omega-omega_0)**2)

def Cluster_Lorentzian_curvefit(n_PXP, n_TI, h_c, T_start, T_max, T_step, Start_cutoff): #TODO CHECK IF CURVEFIT SHOULD BE DONE PARRALELISH
    '''
    Fits lorentzian function to Fourier signals, returns array of gammas for different r's (damping coefficient)
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: optimal coefficients (in the order: Omega_0, gamma, Amplitude) matrix (N-1) x 3
    SAVES ONLY THE COL OF GAMMAS!!!!
    '''
    Freq= np.load('Frequency_T_max_{}_T_step_{}.npy'.format(T_max,T_step))
    sig_func = np.load('Fourier_components_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c))
    popt_tot= np.empty(((seed_max)-1,3))
    for i in range(1,seed_max):
        popt, pcov = curve_fit(Cluster_Lorentzian_function, Freq[Start_cutoff:], sig_func[i-1,Start_cutoff:]) # popt= parameter optimal values
        popt_tot[i-1,:]=popt
    np.save('gamma_array_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c), popt_tot[:,1])

#Cluster_Lorentzian_curvefit(n_PXP, n_TI, h_c, T_start, T_max, T_step, Start_cutoff=5)

def Gamma_time_ave():
    '''
    averages over Gamma's of different realizations
    :return: saves average
    '''
    data= np.load('gamma_array_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c))
    data_ave = np.mean(data)
    np.save('Gamma_ave_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c), data_ave)

def Gamma_Bootstrap(Sample_no):
    '''
    Bootstrapping of Gamma samples
    :return: 95% confidence interval upper and lower bounds for each of time steps' average of random samples
    '''
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    lower_upper = np.empty((2))
    data = np.load('gamma_array_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c))
    sample = np.random.choice(data,(seed_max, Sample_no), replace=True) # creates [(seed_max No.) x Sample_no] matrix of randomly sampled arrays (with return) from the original
    sample_ave = np.mean(sample, axis=0)  # vector of averages sampled from one row of propagation data (random)
    lower_mean = np.quantile(sample_ave, 0.025)
    upper_mean = np.quantile(sample_ave, 0.975)
    lower_upper[0] = lower_mean
    lower_upper[1] = upper_mean
    np.save('Gamma_errors_{}_{}_{}.npy'.format(n_PXP,n_TI,h_c),lower_upper)

#Gamma_time_ave()
#Gamma_Bootstrap(Sample_no)

