import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import PXP_E_B_E_Sparse as Ebe
from PXP_Entry_By_Entry import *
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from scipy.linalg import expm
from scipy.signal import find_peaks
from time import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Coupling_To_Bath import *
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq
from scipy.optimize import curve_fit
from matplotlib import colors

#######################################################################################################################################################################
#                                                                Damping calculation (older method, newer are in the FFT cluster files (or FFT)                                                                          #
#######################################################################################################################################################################

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
    time, VecProp = Ebe.Run_Time_prop_EBE(n_PXP, n_TI, h_c ,T_start, T_max, T_step) #Run_Time_prop uses time division
    Fourier_components= rfft(VecProp)
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=((T_max-T_start)/T_step)) # Freq * (T_max-T_start) = integer that multiplies 2pi
    return Freq, (Height_norm * np.abs(Fourier_components))

def FFT_non_abs(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm):
    '''
    Gets positive frequency fourier components (complex!!!) of the propagation signal and positive frequencies
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 arrays (Positive freq, positive freq fourier components)
    '''
    time, VecProp = Ebe.Run_Time_prop_EBE(n_PXP, n_TI, h_c ,T_start, T_max, T_step) #Run_Time_prop uses time division
    Fourier_components= rfft(VecProp)
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=((T_max-T_start)/T_step)) # Freq * T_max = integer that multiplies 2pi
    return Freq, (Height_norm/Sig_size * (Fourier_components))


def Plot_FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff):
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
    plt.plot(Freq[Start_cutoff:], Inverse_sig[Start_cutoff:], marker='o', markersize=3,
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

def Lorentzian_curvefit(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff):
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
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:], sig_func[Start_cutoff:]) # popt= parameter optimal values
    return popt

def Lorentzian_curvefit_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff,End_cutoff):
    #T_max about 100-300 and T_step 1000-2000ish, remember that t_max/t_step = N_tot ; N_tot/t_max = max freq
    # (need to increase t_max and increase t_step so that number of N_tot does not !! go a lot over t_max)
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
    plt.plot(Freq[Start_cutoff:], sig_func[Start_cutoff:], marker='o', markersize=3,
             color='b', label='data')
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:End_cutoff], sig_func[Start_cutoff:End_cutoff]) # popt= parameter optimal values
    plt.plot(Freq[Start_cutoff:], Lorentzian_function(Freq[Start_cutoff:],*popt),'r-',
         label=r'fit: $\omega_0$={}, $\gamma$={}, Amp={}'.format(*np.round(popt,4)))
    #plt.title('Frequency fit for {} PXP and {} TI atoms, Coupling strength {}'.format(n_PXP,n_TI,h_c))
    plt.xlabel(r'Frequency [$1/t$]')
    plt.ylabel('Amplitudes of Harmonic Functions')
    plt.legend()
    plt.savefig("Figures/Frequency_fit/Freq_Fit_{}_PXP_{}_TI_{}_Coup.png".format(n_PXP,n_TI,h_c))
    return plt.show()

def Lorentzian_inverse_curvefit_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff, End_cutoff):
    '''
    Inverse fits lorentzian function with optimal parameters (=fitted data) back to time domain
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: a plot of the original signal
    '''
    Freq = np.arange(0,((T_step)+2)/(2*T_max),1/T_max)
    Lorentzian = Lorentzian_function(Freq,*Lorentzian_curvefit(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff))
    #plt.plot(Freq,Lorentzian, marker= 'x', markersize= 3, color='r')
    #plt.show()
    Time = np.linspace(T_start,T_max,T_step)
    Lorentzian_IFFT = irfft(Lorentzian, n=len(Time))
    plt.plot(Time,Lorentzian_IFFT, marker= 'x', markersize= 3, color='r')
    return plt.show()

def Lorentzian_inverse_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm):
    '''
    Inverse fits FFT Raw Data (complex numbers) back to time domain - non symmetric damping as of
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :return: a plot of the original signal
    '''
    Freq, sig_func = FFT_non_abs(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    Time = np.linspace(T_start,T_max,T_step)
    Lorentzian_IFFT = irfft(sig_func, n=len(Time))
    plt.plot(Time,Lorentzian_IFFT, marker= 'x', markersize= 3, color='r')
    return plt.show()


def Damping_coef_vs_TI_No_plt(n_PXP, n_TI_max, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Plots normalized damping coefficient (damping coef divided by 0 TI atoms case) for different TI atom numbers
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI_max: Max Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: plot of damping coefficient (gamma) Vs TI atom number
    '''
    gamma = np.empty(n_TI_max)
    n_TI_arr = np.arange(0,n_TI_max,1)
    for n_TI in n_TI_arr:
        omega_0, gamma[n_TI], amp = Lorentzian_curvefit(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff)
    plt.plot(n_TI_arr, np.divide(gamma,gamma[0]) ,marker='o', markersize=3,color='c')
    return plt.show()

def Damping_coef_vs_Coup_Str_plt(n_PXP, n_TI, h_c_max, h_c_interval, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Plots normalized damping coefficient (damping coef divided by 0 coupling) for different coupling strengths
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c_max: max coupling strength
    :param h_c_int: interval of coupling strength jumps
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: plot of damping coefficient (gamma) Vs coupling strength
    '''
    gamma = np.empty(int(h_c_max*np.divide(1,h_c_interval)))
    h_c_arr = np.arange(0,h_c_max,h_c_interval)
    for h_c in h_c_arr:
        omega_0, gamma[int(h_c*np.divide(1,h_c_interval))], amp = Lorentzian_curvefit(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff)
    plt.plot(h_c_arr, np.divide(gamma,gamma[0]) ,marker='x', markersize=3,
             color='c')
    #plt.title(r'$\gamma$ Vs Coupling strength for {} PXP atoms, {} TI atoms'.format(n_PXP,n_TI))
    plt.xlabel('Coupling Strength')
    plt.ylabel(r'$\gamma$ (Normalized Damping Coefficient)')
    #plt.savefig("Gamma_vs_Coup_str_{}_PXP_{}_TI.png".format(n_PXP,n_TI))
    return plt.show()

def Residuals(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Calculates residuals (differences) of fit and inverse fourier signal
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: Vector of differences
    '''
    Freq, sig_func = FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:], sig_func[Start_cutoff:])
    residuals = sig_func - Lorentzian_function(Freq, *popt)
    return residuals[Start_cutoff:]

def Residuals_plot(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Plots residuals (differences) of fit and inverse fourier signal
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param Height_norm: controls the amplitude of the frequency graph (default= 1)
    :param Start_cutoff: cutoff of lowest frequencies (they are weird)
    :return: Plot of residuals (0 means full correspondence)
    '''
    Freq, sig_func = FFT(n_PXP, n_TI, h_c, T_start, T_max, T_step, Height_norm)
    popt, pcov = curve_fit(Lorentzian_function, Freq[Start_cutoff:], sig_func[Start_cutoff:])
    residuals = sig_func - Lorentzian_function(Freq, *popt)
    plt.scatter(Freq[Start_cutoff:], residuals[Start_cutoff:], marker='o',
             color='g')
    plt.title('Residuals')
    return plt.show()



######################################################################################################################################################################################################################################################################3
##                                                                                                                    Bath oscillations                                                                                                                              #
######################################################################################################################################################################################################################################################################3


def O_z_Sparse_Time_prop_PXP_OBC_Only(n_PXP, Initialstate, T_start, T_max, T_step):
    '''
    Returns <Neel|O_z(t)|Neel> values and corresponding time values for PXP Hamiltonian only!!, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: vector -  <Neel_pxp|O_z(t)|Neel_pxp>
    '''
    O_z_PXP = Ebe.O_z_PXP_Entry_Sparse(n_PXP, PXP_Subspace_Algo)
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_Ham_OBC_Sparse(n_PXP,PXP_Subspace_Algo),Initialstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ O_z_PXP @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_O_z_Sparse_Time_prop_PXP_OBC_Only(n_PXP ,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter for PXP Hamiltonian only!!, in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param Initialstate: NeelHaar state usually
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = Neel_Subspace_Basis(n_PXP)
    return O_z_Sparse_Time_prop_PXP_OBC_Only(n_PXP, Initialstate, T_start, T_max, T_step)

def Run_O_z_Sparse_Time_prop_PXP_OBC_Only_plt(n_PXP, T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter for PXP Hamiltonian only!!,in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param Initialstate: NeelHaar state usually
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = Neel_Subspace_Basis(n_PXP)
    sandwich = O_z_Sparse_Time_prop_PXP_OBC_Only(n_PXP, Initialstate, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.xlabel('time')
    plt.ylabel('<N|$O_z(t)$|N>')
    plt.ylim(-0.8,0.3)
    plt.title('$O_z$ Oscillations vs time for {} PXP $ZZ$ coupling'.format(n_PXP))
    return plt.show()

def Neel_state_Sparse_Time_prop(n_PXP, Initialstate, T_start, T_max, T_step):
    '''
    Returns |<Neel|Neel(t)>|^2 values and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return:  vector- |<Neel|Neel(t)>|^2 for diffferent times T
    '''
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_Ham_OBC_Sparse(n_PXP,PXP_Subspace_Algo),Initialstate ,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Bra_Initial = Initialstate
    Sandwich = np.abs(Bra_Initial@Propagated_ket_fin)**2
    return Sandwich.round(4).astype('float')

def Run_Neel_state_Sparse_Time_prop(n_PXP, T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation of |<Neel|Neel(t)>|^2 in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return:  vector- |<Neel|Neel(t)>|^2 for diffferent times T
    '''
    Initialstate = Neel_Subspace_Basis(n_PXP)
    sandwich = Neel_state_Sparse_Time_prop(n_PXP, Initialstate, T_start, T_max, T_step)
    return sandwich

def Run_Neel_state_Sparse_Time_prop_plt(n_PXP, T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation of |<Neel|Neel(t)>|^2 in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param Initialstate: NeelHaar state usually
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = Neel_Subspace_Basis(n_PXP)
    sandwich = Neel_state_Sparse_Time_prop(n_PXP, Initialstate, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.xlabel('time')
    plt.ylabel('$|<Neel|Neel(t)>|^2$')
    plt.ylim(-0.1,1)
    plt.title(' Neel state Oscillations vs time for {} PXP'.format(n_PXP))
    return plt.show()

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
    Initialstate = Neel_EBE_Haar(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)

def Run_O_z_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step):
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
    Initialstate = Neel_EBE_Haar(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    sandwich = O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.xlabel('time')
    plt.ylabel('<N|$O_z(t)$|N>')
    plt.ylim(-0.8,0.3)
    plt.title('$O_z$ Oscillations vs time for {} PXP, {} TI, $h_c=${}'.format(n_PXP,n_TI,np.round(h_c,2)))
    return plt.show()


def PXP_TI_Neelstate(n_PXP,n_TI):
    '''
    Combined Neel state of PXP and TI!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :return: kronekered Neel Neel state
    '''
    TI_Neel= TI_Neelstate(n_TI)
    PXP_Neel= Neel_Subspace_Basis(n_PXP)
    PXP_TI_Neel = np.kron(PXP_Neel,TI_Neel)
    return PXP_TI_Neel

def Z_i_TI_only_Sparse_time_prop(n_TI, Initialstate, J, h_x, h_z, T_start, T_max, T_step,i_index):
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
    :return: vector -  <Neel_bath|Z_i(t)|Neel_bath>
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
    return Z_i_TI_only_Sparse_time_prop(n_TI, Initialstate, J, h_x, h_z, T_start, T_max, T_step,i_index)


def Run_Z_i_TI_only_Sparse_Time_prop_plt(n_TI,i_index,T_start, T_max, T_step):
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
    sandwich= Z_i_TI_only_Sparse_time_prop(n_TI, Initialstate, J, h_x, h_z, T_start, T_max, T_step,i_index)
    plt.plot(np.linspace(T_start,T_max,T_step),sandwich)
    plt.title('$<Z_{}>$ vs. time for {} TI atoms (Pure TI)'.format(i_index,n_TI))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-0.8,0.3)
    return plt.show()

def Z_i_Sys_PXP_Only_Sparse_time_prop(n_PXP,i_index, Initialstate,T_start, T_max, T_step):
    '''
    Returns <Neel|Z_i(t)|Neel> values FOR PXP MODEL ONLY (uncoupled to anything) and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param i_index: site of Z_i
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: vector -  <Neel|Z_i(t)|Neel>
    '''
    Z_n = Ebe.Z_i_PXP_Entry_Sparse(n_PXP,i_index, PXP_Subspace_Algo)
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_Ham_OBC_Sparse(n_PXP,PXP_Subspace_Algo),Initialstate,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ Z_n @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_Z_i_Sys_PXP_Only_Sparse_time_prop(n_PXP, i_index, T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation FOR PXP MODEL ONLY (uncoupled to anything) and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param i_index: site of Z_i
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = Neel_Subspace_Basis(n_PXP)
    return Z_i_Sys_PXP_Only_Sparse_time_prop(n_PXP,i_index, Initialstate,T_start, T_max, T_step)


def Run_Z_i_Sys_PXP_Only_Sparse_time_prop_plt(n_PXP,i_index, T_start, T_max, T_step):
    '''
    plots time AVERAGE propagation FOR PXP MODEL ONLY (uncoupled to anything) and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param i_index: site of Z_i
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = Neel_Subspace_Basis(n_PXP)
    sandwich= Z_i_Sys_PXP_Only_Sparse_time_prop(n_PXP,i_index, Initialstate,T_start, T_max, T_step)
    plt.plot(np.linspace(T_start,T_max,T_step),sandwich)
    plt.title('$<Z_{}>$ vs. time for {} PXP atoms (Pure PXP OBC)'.format(i_index,n_PXP))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1,1)
    return plt.show()

def O_z_Bath_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <NeelxNeel|O_z_TI(t)|NeelxNeel>  values and corresponding time values for ZZ coupling!, working with EBE sparse method
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
    :return: vector -   <NeelxNeel|O_z_TI(t)|NeelxNeel>
    '''
    O_z = Ebe.O_z_Spin_Basis_sparse(n_TI)
    O_z_Full = sp.kron(sp.eye(Subspace_basis_count_faster(n_PXP)), O_z)
    Propagated_ket = Ebe.spla.expm_multiply(-1j * Ebe.PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),
                                            Initialstate,
                                            start=T_start, stop=T_max, num=T_step, endpoint=True)
    Propagated_ket_fin = np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ O_z_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_O_z_Bath_Sparse_Time_prop(n_PXP, n_TI, h_c, T_start, T_max, T_step):
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
    return O_z_Bath_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2)

def Run_O_z_Bath_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step):
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
    sandwich = O_z_Bath_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0,m=2)
    #time_cutoff_arr=np.array((250,500,1000))
    #for time_cutoff in np.nditer(time_cutoff_arr):
    plt.plot(np.linspace(T_start,T_max,T_step)[:],sandwich[:])
    plt.title('$<O_z>$ Bath vs. time for {} TI atoms {} PXP atoms $h_c$={} ZZ coupling'.format(n_TI, n_PXP, np.round(h_c, 5)))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1,1)
        # try:
        #     os.mkdir('Bath_Oscillations/PXP_{}'.format(n_PXP,))
        # except:
        #     pass
        # try:
        #     os.mkdir('Bath_Oscillations/PXP_{}/h_c_{}_True_X_i/'.format(n_PXP, np.round(h_c, 2)))
        # except:
        #     pass
        # plt.savefig('Bath_Oscillations/PXP_{}/h_c_{}_True_X_i/Cutoff_{}_Z_1_Bath_Oscillations_TI_{}_PXP_{}.png'.format(n_PXP,np.round(h_c,2),time_cutoff,n_TI,n_PXP))
    plt.show()
    return

def Z_i_Bath_Sparse_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index, h_imp=0, m=2):
    '''
    Returns <NeelxNeel|Z_i(t)TI|NeelxNeel> values and corresponding time values, working with EBE sparse method
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
    :return: vector -<NeelxNeel|Z_i(t)|NeelxNeel>
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
    return Z_i_Bath_Sparse_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index,h_imp=0, m=2)


def Run_Z_i_Bath_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step):
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
    sandwich= (-1)**(i_index+1)*Z_i_Bath_Sparse_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index,h_imp=0, m=2)
    time_cutoff_arr=np.array((250,500,1000))
    for time_cutoff in np.nditer(time_cutoff_arr):
        plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff],sandwich[:time_cutoff])
        plt.title('$<Z_{}>$ vs. time for {} TI atoms {} PXP atoms $h_c$={}'.format(i_index,n_TI,n_PXP,np.round(h_c,5)))
        plt.xlabel('time')
        plt.ylabel('Amplitude')
        plt.ylim(-1,1)
        # try:
        #     os.mkdir('Bath_Oscillations/PXP_{}'.format(n_PXP,))
        # except:
        #     pass
        # try:
        #     os.mkdir('Bath_Oscillations/PXP_{}/h_c_{}/'.format(n_PXP, np.round(h_c, 2)))
        # except:
        #     pass
        # plt.savefig('Bath_Oscillations/PXP_{}/h_c_{}/Cutoff_{}_Z_1_Bath_Oscillations_TI_{}_PXP_{}.png'.format(n_PXP,np.round(h_c,2),time_cutoff,n_TI,n_PXP))
        plt.show()
    return




def Run_Z_i_Bath_Sparse_Time_prop_plt_grid(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    plots grid of y=time, x=(-1)^(i+1) * <Z_i> for running i, and actual values as colors
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
    Time = np.round(np.linspace(T_start,T_max,T_step),2)
    Z_i_array= np.arange(0,n_TI+1,1)
    Sandwich_mat=np.zeros((T_step,n_TI))
    for i_index in range(1,n_TI+1) :
        Sandwich_mat[:,i_index-1]= ((-1)**(i_index-1))*Z_i_Bath_Sparse_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index,h_imp=0, m=2)
    Sandwich_mat_fin= Sandwich_mat
    cmap = cm.get_cmap('plasma') #define cmap as 'plasma' color map (defult 256 elements in string)
    #bounds = np.linspace(-1.5,1.5,180) #define bin values to map regular values to
    #norm = colors.BoundaryNorm(bounds, cmap.N) #attaches the boundarie bin values to colormap values
    plt.pcolor(Z_i_array,Time,Sandwich_mat_fin, shading='auto', cmap=cmap)
    plt.colorbar()
    plt.title('$<Z_i>$ for different times - $ZZ$ coupling strength {}'.format(np.round(h_c,2)))
    plt.ylabel('Time')
    plt.xlabel('$<Z_i>$')
    return plt.show()

def Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, i_index, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <NeelxNeel|Z_i(t)_PXP|NeelxNeel> values for PXP-TI and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param i_index: site of Z_i
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
    :return: vector - <NeelxNeel|Z_i(t)|NeelxNeel>
    '''
    Z_n = Ebe.Z_i_PXP_Entry_Sparse(n_PXP, i_index, PXP_Subspace_Algo)
    Z_n_Full = sp.kron(Z_n,sp.eye(2**n_TI))
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),Initialstate,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ Z_n_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step):
    '''
    runs time AVG propagation for PXP-TI and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param i_index: site of Z_i
    :param J: TI ising term strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param h_x: longtitudinal term strength (TI)
    :param h_z: transverse term strength
    :param h_c: coupling term strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: Plot of Time propagation
    '''
    Initialstate = PXP_TI_Neelstate(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, i_index,Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2)


def Run_Z_i_System_PXP_TI_Sparse_time_prop_plt(n_PXP, n_TI, h_c,i_index, T_start, T_max, T_step):
    '''
    plots time AVERAGE propagation for PXP-TI and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = PXP_TI_Neelstate(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    sandwich= Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, i_index,Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2)
    plt.plot(np.linspace(T_start,T_max,T_step),sandwich)
    plt.title('$<Z_{}>$ vs. time for {} PXP atoms {} TI atoms $h_c$={}'.format(i_index,n_PXP,n_TI,np.round(h_c,2)))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1,1)
    return plt.show()

def O_z_Bath_O_z_Bath_Sparse_Compare_plt(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,T_start,T_max,T_step,step_cutoff):
    '''
    Plot of comparison of 2 bath oscillation graphs (different parameters can be changed) for ZZ coupling!!!!
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param step_cutoff: cutoff of plot
    :return: figure with 2 plots of O_z oscillations (of bath!!)
    '''
    sandwich1=Run_O_z_Bath_Sparse_Time_prop(n_PXP_1, n_TI_1, h_c_1, T_start, T_max, T_step)
    sandwich2=Run_O_z_Bath_Sparse_Time_prop(n_PXP_2, n_TI_2, h_c_2, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich1[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_1, n_TI_1, h_c_1))
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich2[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_2, n_TI_2, h_c_2))
    plt.title(r'Comparison of $ZZ$ coup. $<O_z>$ bath osc. for cutoff {} (see legend)'.format(step_cutoff))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    return plt.show()

def Z_i_Bath_Z_i_Bath_Sparse_Compare_plt(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,Z_index_1,Z_index_2,T_start,T_max,T_step,step_cutoff):
    '''
    Plot of comparison of 2 oscillation graphs (different parameters can be changed)
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param step_cutoff: cutoff of plot
    :return: figure with 2 plots
    '''
    sandwich1=(-1)**(Z_index_1+1)*Run_Z_i_Bath_Sparse_Time_prop(n_PXP_1, n_TI_1, h_c_1, Z_index_1, T_start, T_max, T_step)
    sandwich2=(-1)**(Z_index_2+1)*Run_Z_i_Bath_Sparse_Time_prop(n_PXP_2, n_TI_2, h_c_2, Z_index_2, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich1[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$, $Z_{}$'.format(n_PXP_1, n_TI_1, h_c_1, Z_index_1))
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich2[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$, $Z_{}$'.format(n_PXP_2, n_TI_2, h_c_2, Z_index_2))
    plt.title(r'Comparison of $ZZ$ coup. bath osc. for cutoff {} (see legend)'.format(step_cutoff))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    return plt.show()

def Z_i_System_Z_i_Bath_Sparse_time_prop_plt(n_PXP, n_TI, h_c,i_index_sys,i_index_bath, T_start, T_max, T_step,Cutoff_step):
    '''
    Plot of comparison between Z_i of system to Z_i of Bath for Z coupling
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param i_index: index of Z_i operator of bath and system
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :param Cutoff_step: cutoff of time division for plots
    :return: Plot of Time propagation
    :return: graph with 2 plots
    '''
    sandwich_Z_i_sys=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, h_c,i_index_sys, T_start, T_max, T_step)
    sandwich_Z_i_bath=Run_Z_i_Bath_Sparse_Time_prop(n_PXP, n_TI, h_c,i_index_bath,T_start, T_max, T_step)
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_Z_i_sys[:Cutoff_step], label='$Z_{}$ system'.format(i_index_sys))
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_Z_i_bath[:Cutoff_step],label='$Z_{}$ Bath'.format(i_index_bath))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('$Z_{}$ of Bath, $Z_{}$ of Sys, {} PXP, {} TI, {} $h_c$ XX coup'.format(i_index_bath,i_index_sys,n_PXP,n_TI,h_c))
    plt.ylim(-1,1)
    plt.show()

def Z_i_System_Z_i_System_Sparse_Compare_plt(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,i_index_1,i_index_2,T_start,T_max,T_step,step_cutoff):
    '''
    Plot of comparison of 2 system Z_i oscillation graphs (different parameters can be changed) for ZZ coupling!!!!
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param step_cutoff: cutoff of plot
    :return: figure with 2 plots of O_z oscillations (of bath!!)
    '''
    sandwich_Z_i_1=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP_1, n_TI_1, h_c_1,i_index_1, T_start, T_max, T_step)
    sandwich_Z_i_2=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP_2, n_TI_2, h_c_2,i_index_2, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich_Z_i_1[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_1, n_TI_1, h_c_1))
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich_Z_i_2[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_2, n_TI_2, h_c_2))
    plt.title(r'Comparison of $ZZ$ coup. $<Z_{}>$ $<Z_{}>$ osc, cutoff {} (see legend)'.format(i_index_1,i_index_2,step_cutoff))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    return plt.show()

def Z_i_System_Inf_Time_Avg(n_PXP,n_TI,h_c,i_index,T_start,T_max,T_step,begin_cutoff):
    '''
     System Z_i oscillation infinite time average value (different parameters can be changed) for ZZ coupling!!!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param Z_index: index of Z_i operator
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param begin_cutoff: where to begin taking values for average
    :return: values of infinite time average
    '''
    sandwich_Z_i=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step)
    print(r'{} PXP, {} TI, {} $h_c$, {} Z index ZZ:'.format(n_PXP, n_TI, h_c,i_index), np.mean(sandwich_Z_i[begin_cutoff:]))
    return np.mean(sandwich_Z_i[begin_cutoff:])

def Z_i_System_Inf_Time_Avg_plot(n_PXP,n_TI,h_c_max,i_index,T_start,T_max,T_step,begin_cutoff):
    '''
    Plots System Z_i oscillation infinite time average values vs coupling strength (different parameters can be changed) ZZ coupling
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c_max: coupling strength
    :param Z_index: index of Z_i operator
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param begin_cutoff: where to begin taking values for average
    :return: plot of Infinite Time average vs h_c ZZ coupling
    '''
    h_c_array= np.arange(0.0,h_c_max+0.1,0.1)
    Z_i_time_averaged=np.zeros((len(h_c_array)))
    count=0
    for h_c in h_c_array:
        sandwich_Z_i_ZZ=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, np.round(h_c,2),i_index, T_start, T_max, T_step)
        Z_i_time_averaged[count]=np.mean(sandwich_Z_i_ZZ[begin_cutoff:])
        count=count+1
    plt.plot(h_c_array,Z_i_time_averaged)
    plt.xlabel('$h_c$ (coupling strength)')
    plt.ylabel(r'$Z_i^{Inf}$')
    plt.title('Infinite Time Avg of $Z_{}$ vs. Coupling Str {} PXP, {} TI ZZ Coupling'.format(i_index,n_PXP,n_TI))
    return plt.show()

def Z_i_System_O_z_System_Sparse_Compare_plt(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,i_index,T_start,T_max,T_step,step_cutoff):
    '''
    Plot of comparison of Z_i and O_z system oscillation graphs (different parameters can be changed) for ZZ coupling!!!!
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param step_cutoff: cutoff of plot
    :return: figure with 2 plots of O_z oscillations (of bath!!)
    '''
    sandwich_O_z=Run_O_z_Sparse_Time_prop(n_PXP_1, n_TI_1, h_c_1, T_start, T_max, T_step)
    sandwich_Z_i=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP_2, n_TI_2, h_c_2,i_index, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich_O_z[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_1, n_TI_1, h_c_1))
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich_Z_i[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_2, n_TI_2, h_c_2))
    plt.title(r'Comparison of $ZZ$ coup. $<O_z>$ and $<Z_{}>$ osc, cutoff {} (see legend)'.format(i_index,step_cutoff))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    return plt.show()

def Z_i_System_Neel_Sparse_Compare_plt(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,i_index,T_start,T_max,T_step,step_cutoff):
    '''
    Plot of comparison of Z_i and Neel system oscillation graphs (different parameters can be changed) for ZZ coupling!!!!
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param step_cutoff: cutoff of plot
    :return: figure with 2 plots of O_z oscillations (of bath!!)
    '''
    sandwich_Neel=Run_Neel_state_Sparse_Time_prop(n_PXP_1, T_start, T_max, T_step)
    sandwich_Z_i=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP_2, n_TI_2, h_c_2,i_index, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich_Neel[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_1, n_TI_1, h_c_1))
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich_Z_i[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_2, n_TI_2, h_c_2))
    plt.title(r'Comparison of $ZZ$ coup. $<O_z>$ and $<Z_{}>$ osc, cutoff {} (see legend)'.format(i_index,step_cutoff))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    return plt.show()

def Run_Z_i_system_Sparse_Time_prop_plt_grid(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    plots grid of y=time, x=(-1)^(i+1) * <Z_i> for running i, and actual values as colors
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Time = np.round(np.linspace(T_start,T_max,T_step),2)
    Z_i_array= np.arange(0,n_PXP+1,1)
    Sandwich_mat=np.zeros((T_step,n_PXP))
    for i_index in range(1,n_PXP+1) :
        Sandwich_mat[:,i_index-1]=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step)
    Sandwich_mat_fin= Sandwich_mat
    cmap = cm.get_cmap('plasma') #define cmap as 'plasma' color map (defult 256 elements in string)
    #bounds = np.linspace(-1.5,1.5,180) #define bin values to map regular values to
    #norm = colors.BoundaryNorm(bounds, cmap.N) #attaches the boundarie bin values to colormap values
    plt.pcolor(Z_i_array,Time,Sandwich_mat_fin, shading='auto', cmap=cmap,vmin=-1,vmax=1)
    plt.colorbar()
    plt.title('$<Z_i>$ for different times - $ZZ$ coupling strength {}'.format(np.round(h_c,2)))
    plt.ylabel('Time')
    plt.xlabel('$<Z_i>$')
    plt.savefig('Figures/Grid_Z_i_Osc_Sys/ZZ/Grid_Plot_{}_PXP_{}_TI_{}_h_c.png'.format(n_PXP,n_TI,np.round(h_c,2)))
    return plt.show()

def Run_Z_i_system_Sparse_Time_prop_plt_deducted_grid(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    plots grid of y=time, x= <Z_i> for running i, and actual values as colors, while deducting the coupling 0 oscillations from the output of each Z_i
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Grid plot with oscillation deducted values
    '''
    Time = np.round(np.linspace(T_start,T_max,T_step),2)
    Z_i_array= np.arange(0,n_PXP+1,1)
    Sandwich_mat=np.zeros((T_step,n_PXP))
    for i_index in range(1,n_PXP+1):
        Zero_Coupling_Sandwich_mat = Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, 0.0, i_index, T_start, T_max,T_step)
        Sandwich_mat[:,i_index-1]=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step)-Zero_Coupling_Sandwich_mat
    Sandwich_mat_fin= Sandwich_mat
    cmap = cm.get_cmap('plasma') #define cmap as 'plasma' color map (defult 256 elements in string)
    #bounds = np.linspace(-1.5,1.5,180) #define bin values to map regular values to
    #norm = colors.BoundaryNorm(bounds, cmap.N) #attaches the boundarie bin values to colormap values
    plt.pcolor(Z_i_array,Time,Sandwich_mat_fin, shading='auto', cmap=cmap,vmin=-1,vmax=1)
    plt.colorbar()
    plt.title('$<Z_i>$ for different times - $ZZ$ coupling strength {}'.format(np.round(h_c,2)))
    plt.ylabel('Time')
    plt.xlabel('$<Z_i>$')
    #plt.savefig('Figures/Grid_Z_i_Osc_Sys/ZZ/Grid_Plot_{}_PXP_{}_TI_{}_h_c.png'.format(n_PXP,n_TI,np.round(h_c,2)))
    return plt.show()

#######################################################################################################################################################################
####                                                             True X_i Oscillations Part                                                                        ####
#######################################################################################################################################################################



def O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <NeelxHaar|O_z(t)|NeelxHaar> values and corresponding time values, FOR EXTENDED PXP BASIS, working with EBE sparse method
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
    Initialstate = Neel_EBE_Haar_X_i_Extended(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)

def Run_O_z_Sparse_True_X_i_Time_prop_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step):
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
    Initialstate = Neel_EBE_Haar_X_i_Extended(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    sandwich = O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.xlabel('time')
    plt.ylabel('<N|$O_z(t)$|N>')
    plt.title('$O_z$ True $X_i$ Oscillations vs time for {} PXP, {} TI, $h_c=${}'.format(n_PXP,n_TI,np.round(h_c,2)))
    plt.ylim(-0.8,0.3)
    return plt.show()

# def O_z_Sparse_True_X_i_Time_prop_PXP_OBC_Only(n_PXP, Initialstate, T_start, T_max, T_step):
#     '''
#     Returns <Neel|O_z(t)|Neel> values and corresponding time values for PXP Hamiltonian only!! FOR EXTENDED PXP BASIS, working with EBE sparse method
#     :param n_PXP: No. of PXP atoms
#     :param Initialstate:  Initial Vector state we would like to propagate
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :return: vector -  <Neel_pxp|O_z(t)|Neel_pxp>
#     '''
#     O_z_PXP = Ebe.O_z_PXP_Entry_Sparse(n_PXP, PXP_Subspace_Algo_extended_X_i)
#     Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_Ham_OBC_Sparse_True_X_i(n_PXP,PXP_Subspace_Algo_extended_X_i),Initialstate ,
#                                         start= T_start , stop=T_max ,num = T_step ,endpoint = True)
#     Propagated_ket_fin= np.transpose(Propagated_ket)
#     Propagated_bra_fin = np.conjugate(Propagated_ket)
#     Sandwich = np.diag(Propagated_bra_fin @ O_z_PXP @ Propagated_ket_fin)
#     return Sandwich.round(4).astype('float')
#
# def Run_O_z_Sparse_True_X_i_Time_prop_PXP_OBC_Only(n_PXP ,T_start, T_max, T_step):
#     '''
#     Runs time AVERAGE propagation plotter for PXP Hamiltonian only!! FOR EXTENDED PXP BASIS, in EBE sparse method
#     :param n_PXP: No. of PXP atoms
#     :param Initialstate: NeelHaar state usually
#     :param T_start: start time
#     :param T_max: end time
#     :param T_step: time division
#     :return: Plot of Time propagation
#     '''
#     Initialstate = Neel_X_i_Extended_Subspace_Basis(n_PXP)
#     return O_z_Sparse_True_X_i_Time_prop_PXP_OBC_Only(n_PXP, Initialstate, T_start, T_max, T_step)
#
# def Run_O_z_Sparse_True_X_i_Time_prop_PXP_OBC_Only_plt(n_PXP, T_start, T_max, T_step):
#     '''
#     Runs time AVERAGE propagation plotter for PXP Hamiltonian only!! FOR EXTENDED PXP BASIS,in EBE sparse method
#     :param n_PXP: No. of PXP atoms
#     :param Initialstate: NeelHaar state usually
#     :param T_start: start time
#     :param T_max: end time
#     :param T_step: time division
#     :return: Plot of Time propagation
#     '''
#     Initialstate = Neel_X_i_Extended_Subspace_Basis(n_PXP)
#     sandwich = O_z_Sparse_True_X_i_Time_prop_PXP_OBC_Only(n_PXP, Initialstate, T_start, T_max, T_step)
#     plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
#     plt.xlabel('time')
#     plt.ylabel('<N|$O_z(t)$|N>')
#     plt.ylim(-0.8,0.3)
#     plt.title('<$O_z$> Oscillations vs time for {} PXP $XX$ coupling'.format(n_PXP))
#     return plt.show()


def PXP_TI_Neelstate_True_X_i(n_PXP,n_TI):
    '''
    Combined Neel state of PXP and TI! for extended X_i PXP basis
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :return: kronekered Neel state
    '''
    TI_Neel= TI_Neelstate(n_TI)
    PXP_Neel= Neel_X_i_Extended_Subspace_Basis(n_PXP)
    PXP_TI_Neel = np.kron(PXP_Neel,TI_Neel)
    return PXP_TI_Neel

# def Z_n_PXP_True_X_i_Only_Sparse_time_prop(n_PXP, Initialstate,T_start, T_max, T_step):
#     '''
#     Returns <Neel|Z_n(t)|Neel> values FOR PXP extended X_i MODEL ONLY (uncoupled to anything) and corresponding time values, working with EBE sparse method
#     :param n_PXP: No. of PXP atoms
#     :param Initialstate:  Initial Vector state we would like to propagate
#     :param T_start: Start Time of propagation
#     :param T_max: Max Time of propagation
#     :param T_step: time step (division)
#     :return: vector -  <Neel|Z_n(t)|Neel>
#     '''
#     Z_n = Ebe.Z_i_PXP_Entry_Sparse(n_PXP, n_PXP, PXP_Subspace_Algo_extended_X_i)
#     Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_Ham_OBC_Sparse_True_X_i(n_PXP,PXP_Subspace_Algo_extended_X_i),Initialstate,
#                                         start= T_start , stop=T_max ,num = T_step ,endpoint = True)
#     Propagated_ket_fin= np.transpose(Propagated_ket)
#     Propagated_bra_fin = np.conjugate(Propagated_ket)
#     Sandwich = np.diag(Propagated_bra_fin @ Z_n @ Propagated_ket_fin)
#     return Sandwich.round(4).astype('float')
#
# def Run_Z_n_PXP_True_X_i_Only_Sparse_time_prop(n_PXP,T_start, T_max, T_step):
#     '''
#     Runs time AVERAGE propagation FOR PXP extended X_i MODEL ONLY (uncoupled to anything) and corresponding time values, working with EBE sparse method
#     :param n_PXP: No. of PXP atoms
#     :param T_start: start time
#     :param T_max: end time
#     :param T_step: time division
#     :return: Plot of Time propagation
#     '''
#     Initialstate = Neel_X_i_Extended_Subspace_Basis(n_PXP)
#     return Z_n_PXP_True_X_i_Only_Sparse_time_prop(n_PXP, Initialstate,T_start, T_max, T_step)
#
#
# def Run_Z_n_PXP_True_X_i_Only_Sparse_time_prop_plt(n_PXP,T_start, T_max, T_step):
#     '''
#     plots time AVERAGE propagation FOR PXP extended X_i MODEL ONLY (uncoupled to anything) and corresponding time values, working with EBE sparse method
#     :param n_PXP: No. of PXP atoms
#     :param T_start: start time
#     :param T_max: end time
#     :param T_step: time division
#     :return: Plot of Time propagation
#     '''
#     Initialstate = Neel_X_i_Extended_Subspace_Basis(n_PXP)
#     sandwich= Z_n_PXP_True_X_i_Only_Sparse_time_prop(n_PXP, Initialstate,T_start, T_max, T_step)
#     plt.plot(np.linspace(T_start,T_max,T_step),sandwich)
#     plt.title('$<Z_n>$ vs. time for {} PXP atoms (Pure PXP OBC)'.format(n_PXP))
#     plt.xlabel('time')
#     plt.ylabel('Amplitude')
#     plt.ylim(-1,1)
#     return plt.show()
def O_z_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <NeelxNeel|O_z_TI(t)|NeelxNeel>  values and corresponding time values FOR X_i coupling!, working with EBE sparse method
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
    :return: vector -   <NeelxNeel|O_z_TI(t)|NeelxNeel>
    '''
    O_z = Ebe.O_z_Spin_Basis_sparse(n_TI)
    O_z_Full = sp.kron(sp.eye(Extended_X_i_Subspace_basis_count_faster(n_PXP)), O_z)
    Propagated_ket = Ebe.spla.expm_multiply(-1j * Ebe.PXP_TI_coupled_Sparse_Xi(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),
                                            Initialstate,
                                            start=T_start, stop=T_max, num=T_step, endpoint=True)
    Propagated_ket_fin = np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ O_z_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_O_z_Bath_True_X_i_Sparse_Time_prop(n_PXP, n_TI, h_c, T_start, T_max, T_step):
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
    return O_z_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2)

def Run_O_z_Bath_True_X_i_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, T_start, T_max, T_step):
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
    sandwich = O_z_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0,m=2)
    #time_cutoff_arr=np.array((250,500,1000))
    #for time_cutoff in np.nditer(time_cutoff_arr):
    plt.plot(np.linspace(T_start,T_max,T_step)[:],sandwich[:])
    plt.title('$<O_z>$ Bath vs. time for {} TI atoms {} PXP atoms $h_c$={} X_i coupling'.format(n_TI, n_PXP, np.round(h_c, 5)))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1,1)
        # try:
        #     os.mkdir('Bath_Oscillations/PXP_{}'.format(n_PXP,))
        # except:
        #     pass
        # try:
        #     os.mkdir('Bath_Oscillations/PXP_{}/h_c_{}_True_X_i/'.format(n_PXP, np.round(h_c, 2)))
        # except:
        #     pass
        # plt.savefig('Bath_Oscillations/PXP_{}/h_c_{}_True_X_i/Cutoff_{}_Z_1_Bath_Oscillations_TI_{}_PXP_{}.png'.format(n_PXP,np.round(h_c,2),time_cutoff,n_TI,n_PXP))
    plt.show()
    return

def Z_i_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0, m=2):
    '''
    Returns <NeelxNeel|Z_i(t)|NeelxNeel>  values and corresponding time values FOR X_i coupling!, working with EBE sparse method
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
    :return: vector -  <NeelxNeel|Z_i(t)|NeelxNeel>
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
    return Z_i_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0, m=2)

def Run_Z_i_Bath_True_X_i_Sparse_Time_prop_var(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method, different parameters
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
    J = 1.1
    h_x = np.sin(0.475 * np.pi)
    h_z = np.cos(0.475 * np.pi)
    return Z_i_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0, m=2)

def Run_Z_i_Bath_True_X_i_Sparse_Time_prop_plt(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step):
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
    sandwich = (-1)**(i_index+1)*Z_i_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, i_index, h_imp=0,
                                  m=2)
    time_cutoff_arr=np.array((250,500,1000))
    for time_cutoff in np.nditer(time_cutoff_arr):
        plt.plot(np.linspace(T_start,T_max,T_step)[:time_cutoff],sandwich[:time_cutoff])
        plt.title('$<Z_{}>$ vs. time for {} TI atoms {} PXP atoms $h_c$={} X_i coupling'.format(i_index, n_TI, n_PXP, np.round(h_c, 5)))
        plt.xlabel('time')
        plt.ylabel('Amplitude')
        plt.ylim(-1,1)
        # try:
        #     os.mkdir('Bath_Oscillations/PXP_{}'.format(n_PXP,))
        # except:
        #     pass
        # try:
        #     os.mkdir('Bath_Oscillations/PXP_{}/h_c_{}_True_X_i/'.format(n_PXP, np.round(h_c, 2)))
        # except:
        #     pass
        # plt.savefig('Bath_Oscillations/PXP_{}/h_c_{}_True_X_i/Cutoff_{}_Z_1_Bath_Oscillations_TI_{}_PXP_{}.png'.format(n_PXP,np.round(h_c,2),time_cutoff,n_TI,n_PXP))
        plt.show()
    return



def Run_Z_i_Bath_True_X_i_Sparse_Time_prop_plt_grid(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    ''' #TODO add Vmin Vmax to pcolor
    plots grid of y=time, x=<Z_i> running i, and actual values as colors for extended X_i basis!!!
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
    :return: Grid Plot
    '''
    Initialstate = PXP_TI_Neelstate_True_X_i(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    Time = np.round(np.linspace(T_start,T_max,T_step),2)
    Z_i_array= np.arange(0,n_TI+1,1)
    Sandwich_mat=np.zeros((T_step,n_TI))
    for i_index in range(1,n_TI+1) :
        Sandwich_mat[:,i_index-1]=Z_i_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index,h_imp=0, m=2)
    Sandwich_mat_fin= Sandwich_mat
    cmap = cm.get_cmap('plasma') #define cmap as 'plasma' color map (defult 256 elements in string)
    plt.pcolor(Z_i_array,Time,Sandwich_mat_fin, shading='auto', cmap=cmap)
    plt.colorbar()
    plt.title('$<Z_i>$ for different times - $XX$ coupling strength {}'.format(np.round(h_c,2)))
    plt.ylabel('Time')
    plt.xlabel('$<Z_i>$')
    return plt.show()


def Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, i_index, Initialstate,J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <NeelxNeel|Z_i(t)_PXP|NeelxNeel> values for PXP-TI Extended X_i basis and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param i: index of Z_i
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
    :return: vector - <NeelxNeel|Z_n(t)|NeelxNeel> extended X_i
    '''
    Z_n = Ebe.Z_i_PXP_Entry_Sparse(n_PXP, i_index, PXP_Subspace_Algo_extended_X_i)
    Z_n_Full = sp.kron(Z_n,sp.eye(2**n_TI))
    Propagated_ket = Ebe.spla.expm_multiply(-1j*Ebe.PXP_TI_coupled_Sparse_Xi(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),Initialstate,
                                        start= T_start , stop=T_max ,num = T_step ,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ Z_n_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, h_c, i, T_start, T_max, T_step):
    '''
    runs time AVG propagation for Z_i(t) PXP of PXP-TI coupled HAM Extended X_i basis!!!! and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param i: index of Z_i
    :param J: TI ising term strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param h_x: longtitudinal term strength (TI)
    :param h_z: transverse term strength
    :param h_c: coupling term strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: Plot of Time propagation
    '''
    Initialstate = PXP_TI_Neelstate_True_X_i(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    return Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, i, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2)


def Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop_plt(n_PXP, n_TI, h_c, i, T_start, T_max, T_step):
    '''
    plots time AVERAGE propagation for PXP-TI Extended X_i basis!!!! and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param h_c: coupling term strength
    :param i: index of Z_i
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = PXP_TI_Neelstate_True_X_i(n_PXP, n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    sandwich= Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, i, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2)
    plt.plot(np.linspace(T_start,T_max,T_step),sandwich)
    plt.title('$<Z_{}>$ vs. time for {} PXP atoms {} TI atoms $h_c$={} $X_i$ coup'.format(i,n_PXP,n_TI,np.round(h_c,2)))
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1,1)
    return plt.show()

def Run_Z_i_System_True_X_i_Sparse_Time_prop_plt_grid(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    plots grid of y=time, x= <Z_i> running i, and actual values as colors for extended XX basis!!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: grid plot
    '''
    Time = np.round(np.linspace(T_start,T_max,T_step),2)
    Z_i_array= np.arange(0,n_PXP+1,1)
    Sandwich_mat=np.zeros((T_step,n_PXP))
    for i_index in range(1,n_PXP+1) :
        Sandwich_mat[:,i_index-1]= Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, h_c,i_index, T_start, T_max, T_step)
    Sandwich_mat_fin= Sandwich_mat
    cmap = cm.get_cmap('plasma') #define cmap as 'plasma' color map (defult 256 elements in string)
    plt.pcolor(Z_i_array,Time,Sandwich_mat_fin, shading='auto', cmap=cmap,vmin=-1,vmax=1)
    plt.colorbar()
    plt.title('$<Z_i>$ for different times - $XX$ coupling strength {}'.format(np.round(h_c,2)))
    plt.ylabel('Time')
    plt.xlabel('$<Z_i>$')
    #plt.savefig('Figures/Grid_Z_i_Osc_Sys/XX/Grid_Plot_{}_PXP_{}_TI_{}_h_c.png'.format(n_PXP,n_TI,np.round(h_c,2)))
    return plt.show()

def Run_Z_i_System_True_X_i_Sparse_Time_prop_plt_deducted_grid(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    plots grid of y=time, x= <Z_i> running i, and actual values as colors for extended XX basis, while deducting the h_c=0 case from each Z_i
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: grid plot
    '''
    Time = np.round(np.linspace(T_start,T_max,T_step),2)
    Z_i_array= np.arange(0,n_PXP+1,1)
    Sandwich_mat=np.zeros((T_step,n_PXP))
    for i_index in range(1,n_PXP+1) :
        Zero_Coupling_Sandwich_mat = Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, 0.0, i_index, T_start, T_max,
                                                                            T_step)
        Sandwich_mat[:,i_index-1]= Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, h_c,i_index, T_start, T_max, T_step) -Zero_Coupling_Sandwich_mat
    Sandwich_mat_fin= Sandwich_mat
    cmap = cm.get_cmap('plasma') #define cmap as 'plasma' color map (defult 256 elements in string)
    plt.pcolor(Z_i_array,Time,Sandwich_mat_fin, shading='auto', cmap=cmap,vmin=-1,vmax=1)
    plt.colorbar()
    plt.title('$<Z_i>$ for different times - $XX$ coupling strength {}'.format(np.round(h_c,2)))
    plt.ylabel('Time')
    plt.xlabel('$<Z_i>$')
    #plt.savefig('Figures/Grid_Z_i_Osc_Sys/XX/Grid_Plot_{}_PXP_{}_TI_{}_h_c.png'.format(n_PXP,n_TI,np.round(h_c,2)))
    return plt.show()

def Z_i_System_True_X_i_Inf_Time_Avg(n_PXP,n_TI,h_c,i_index,T_start,T_max,T_step,begin_cutoff):
    '''
    System Z_i oscillation infinite time average value (different parameters can be changed) XX coupling
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param Z_index: index of Z_i operator
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param begin_cutoff: where to begin taking values for average
    :return:  values of infinite time average for XX coupling
    '''
    sandwich_Z_i_XX=Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, h_c,i_index, T_start, T_max, T_step)
    print(r'{} PXP, {} TI, {} $h_c$, {} Z index XX:'.format(n_PXP, n_TI, h_c,i_index), np.mean(sandwich_Z_i_XX[begin_cutoff:]))
    return np.mean(sandwich_Z_i_XX[begin_cutoff:])

def Z_i_System_True_X_i_Inf_Time_Avg_plot(n_PXP,n_TI,h_c_max,i_index,T_start,T_max,T_step,begin_cutoff):
    '''
    Plots System Z_i oscillation infinite time average values vs coupling strength (different parameters can be changed) XX coupling
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c_max: coupling strength
    :param Z_index: index of Z_i operator
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param begin_cutoff: where to begin taking values for average
    :return: plot of Infinite Time average vs h_c
    '''
    h_c_array= np.arange(0.0,h_c_max+0.1,0.1)
    Z_i_time_averaged=np.zeros((len(h_c_array)))
    count=0
    for h_c in h_c_array:
        sandwich_Z_i_XX=Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, np.round(h_c,2),i_index, T_start, T_max, T_step)
        Z_i_time_averaged[count]=np.mean(sandwich_Z_i_XX[begin_cutoff:])
        count=count+1
    plt.plot(h_c_array,Z_i_time_averaged)
    plt.xlabel('$h_c$ (coupling strength)')
    plt.ylabel(r'$Z_i^{Inf}$')
    plt.title('Infinite Time Avg of $Z_{}$ vs. Coupling Str {} PXP, {} TI XX Coupling'.format(i_index,n_PXP,n_TI))
    return plt.show()

def O_z_Bath_O_z_Bath_True_X_i_Sparse_Compare_plt(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,T_start,T_max,T_step,step_cutoff):
    '''
    Plot of comparison of 2 bath oscillation graphs (different parameters can be changed) for XX coupling!!!!
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param step_cutoff: cutoff of plot
    :return: figure with 2 plots of O_z oscillations (of bath!!)
    '''
    sandwich1=Run_O_z_Bath_True_X_i_Sparse_Time_prop(n_PXP_1, n_TI_1, h_c_1, T_start, T_max, T_step)
    sandwich2=Run_O_z_Bath_True_X_i_Sparse_Time_prop(n_PXP_2, n_TI_2, h_c_2, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich1[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_1, n_TI_1, h_c_1))
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich2[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$'.format(n_PXP_2, n_TI_2, h_c_2))
    plt.title(r'Comparison of $X_i$ coup. $<O_z>$ bath osc. for cutoff {} (see legend)'.format(step_cutoff))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    return plt.show()

def Z_i_Bath_Z_i_Bath_True_X_i_Sparse_Compare_plt(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,Z_index_1,Z_index_2,T_start,T_max,T_step,step_cutoff):
    '''
    Plot of comparison of 2 oscillation graphs (different parameters can be changed) for X_i coupling!!!!
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param step_cutoff: cutoff of plot
    :return: figure with 2 plots
    '''
    sandwich1=(-1)**(Z_index_1+1)*Run_Z_i_Bath_True_X_i_Sparse_Time_prop(n_PXP_1, n_TI_1, h_c_1, Z_index_1, T_start, T_max, T_step)
    sandwich2=(-1)**(Z_index_2+1)*Run_Z_i_Bath_True_X_i_Sparse_Time_prop(n_PXP_2, n_TI_2, h_c_2, Z_index_2, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich1[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$, $Z_{}$'.format(n_PXP_1, n_TI_1, h_c_1, Z_index_1))
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich2[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$, $Z_{}$'.format(n_PXP_2, n_TI_2, h_c_2, Z_index_2))
    plt.title(r'Comparison of $X_i$ coup. bath osc. for cutoff {} (see legend)'.format(step_cutoff))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    return plt.show()

def Z_i_Bath_ZZ_Coup_Z_i_Bath_XX_Coup_Sparse_Compare_plt(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,Z_index_1,Z_index_2,T_start,T_max,T_step,step_cutoff):
    '''
    Plot of comparison of 2 oscillation graphs (different parameters can be changed) one for ZZ and other for XX coupling!!!!
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param step_cutoff: cutoff of plot
    :return: figure with 2 plots
    '''
    sandwich1=(-1)**(Z_index_1+1)*Run_Z_i_Bath_Sparse_Time_prop(n_PXP_1, n_TI_1, h_c_1, Z_index_1, T_start, T_max, T_step)
    sandwich2=(-1)**(Z_index_2+1)*Run_Z_i_Bath_True_X_i_Sparse_Time_prop(n_PXP_2, n_TI_2, h_c_2, Z_index_2, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich1[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$, $Z_{}$, $ZZ$ coupling'.format(n_PXP_1, n_TI_1, h_c_1, Z_index_1))
    plt.plot(np.linspace(T_start, T_max, T_step)[:step_cutoff], sandwich2[:step_cutoff],label=r'{} PXP, {} TI, {} $h_c$, $Z_{}$, $XX$ coupling'.format(n_PXP_2, n_TI_2, h_c_2, Z_index_2))
    plt.title(r'Comparison of $ZZ$ coup. and $XX$ coup. bath osc. (see legend)'.format(step_cutoff))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.savefig('Bath_Oscillations_thesis_present.png')
    plt.ylim(-1, 1)
    return plt.show()



def Z_i_System_Z_i_Bath_True_X_i_Sparse_Compare_plt(n_PXP, n_TI, h_c, i_index_system, i_index_bath, T_start, T_max, T_step,Cutoff_step):
    '''
    Plot of comparison between Z_ן of system to Z_i of Bath for X_i coupling nature!!!
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
    sandwich_Z_i_sys=(-1)**(i_index_system+1)*Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, h_c, i_index_system, T_start, T_max, T_step)
    sandwich_Z_i_bath=(-1)**(i_index_bath+1)*Run_Z_i_Bath_True_X_i_Sparse_Time_prop(n_PXP, n_TI, h_c,i_index_bath,T_start, T_max, T_step)
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_Z_i_sys[:Cutoff_step], label='$Z_{}$ system'.format(i_index_system))
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_Z_i_bath[:Cutoff_step],label='$Z_{}$ Bath'.format(i_index_bath))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('$Z_{}$ of Bath, $Z_{}$ of Sys, {} PXP, {} TI, {} $h_c$ X_i coup.'.format(i_index_bath,i_index_system,n_PXP,n_TI,h_c))
    plt.ylim(-1,1)
    plt.show()

def O_z_System_Z_i_Bath_True_X_i_Sparse_Compare_plt(n_PXP, n_TI, h_c,i_index, T_start, T_max, T_step,Cutoff_step):
    '''
    Plot of comparison between O_z of system to Z_i of bath for X_i coupling nature!!!
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
    sandwich_O_z_sys=Run_O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, h_c,T_start, T_max, T_step)
    sandwich_Z_i_bath=Run_Z_i_Bath_True_X_i_Sparse_Time_prop(n_PXP, n_TI, h_c,i_index,T_start, T_max, T_step)
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_O_z_sys[:Cutoff_step], label='$O_z$ system')
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_Z_i_bath[:Cutoff_step],label='$Z_{}$ Bath'.format(i_index))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('$Z_{}$ of Bath, $O_z$ of Sys, {} PXP, {} TI, {} $h_c$ X_i coup.'.format(i_index,n_PXP,n_TI,h_c))
    plt.ylim(-1,1)
    plt.show()

def Z_i_System_O_z_System_True_X_i_Sparse_Compare_plt(n_PXP, n_TI, h_c,i_index_sys, T_start, T_max, T_step,Cutoff_step):
    '''
    Plot of comparison between Z_i of system to O_z of system for X_i coupling nature!!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :param Cutoff_step: cutoff of steps in plots
    :return: Plot of Time propagation
    :return: graph with 2 plots
    '''
    sandwich_Z_i_sys=Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, h_c, i_index_sys,T_start, T_max, T_step)
    sandwich_O_z_sys=Run_O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, h_c,T_start, T_max, T_step)
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_Z_i_sys[:Cutoff_step], label='$Z_i$ system'.format(i_index_sys))
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_O_z_sys[:Cutoff_step],label='$O_z$ system')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('$O_z$ of Sys, $Z_{}$ of Sys, {} PXP, {} TI, {} $h_c$ X_i coup.'.format(i_index_sys,n_PXP,n_TI,h_c))
    plt.ylim(-1,1)
    plt.show()

def O_z_System_O_z_System_True_X_i_Sparse_Compare_plt(n_PXP_1, n_PXP_2, n_TI_1, n_TI_2, h_c_1,h_c_2, T_start, T_max, T_step,Cutoff_step):
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
    sandwich_O_z_1_sys=Run_O_z_Sparse_True_X_i_Time_prop(n_PXP_1, n_TI_1, h_c_1,T_start, T_max, T_step)
    sandwich_O_z_2_sys=Run_O_z_Sparse_True_X_i_Time_prop(n_PXP_2, n_TI_2, h_c_2,T_start, T_max, T_step)
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_O_z_1_sys[:Cutoff_step], label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_O_z_2_sys[:Cutoff_step],label=r'$O_z$ sys {} PXP, {} TI, $h_c={}$'.format(n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Compare $O_z$ of Sys (see legend)')
    plt.ylim(-0.8,0.2)
    plt.show()

def Z_i_System_Z_i_System_True_X_i_Sparse_Compare_plt(n_PXP_1, n_PXP_2, n_TI_1, n_TI_2, i_index_sys_1, i_index_sys_2, h_c_1,h_c_2, T_start, T_max, T_step,Cutoff_step):
    '''
    Plot of comparison between Z_i of system to Z_i of system for X_i coupling nature!!!
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param i_index: indeces of Z_i operator
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :param Cutoff_step: cutoff of steps in plots
    :return: graph with 2 plots
    '''
    sandwich_Z_i_1_sys=(-1)**(i_index_sys_1+1)*Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP_1, n_TI_1, h_c_1, i_index_sys_1,T_start, T_max, T_step)
    sandwich_Z_i_2_sys=(-1)**(i_index_sys_2+1)*Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP_2, n_TI_2, h_c_2,i_index_sys_2, T_start, T_max, T_step)
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_Z_i_1_sys[:Cutoff_step], label=r'$Z_{}$ sys {} PXP, {} TI, $h_c={}$'.format(i_index_sys_1,n_PXP_1,n_TI_1,np.round(h_c_1,2)))
    plt.plot(np.linspace(T_start,T_max,T_step)[:Cutoff_step],sandwich_Z_i_2_sys[:Cutoff_step],label=r'$Z_{}$ sys {} PXP, {} TI, $h_c={}$'.format(i_index_sys_2,n_PXP_2,n_TI_2,np.round(h_c_2,2)))
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Comparison of $Z_i$ of Sys (see legend)')
    plt.ylim(-1,1)
    plt.show()

def Z_i_System_Inf_Time_Avg_ZZ_XX_Compare(n_PXP_1,n_PXP_2,n_TI_1,n_TI_2,h_c_1,h_c_2,i_index_1,i_index_2,T_start,T_max,T_step,begin_cutoff):
    '''
    comparison of 2 system Z_i oscillation infinite time average values (different parameters can be changed) ZZ XX coupling compare
    :param n_PXP_1: No. of PXP atoms 1st graph
    :param n_PXP_2: No. of PXP atoms 2nd graph
    :param n_TI_1: No. of TI atoms 1st graph
    :param n_TI_2: No. of TI atoms 2nd graph
    :param h_c_1: coupling strength 1st graph
    :param h_c_2: coupling strength 2nd graph
    :param Z_index_1: index of Z_i operator of 1st graph
    :param Z_index_2: index of Z_i operator of 2nd graph
    :param T_start: start time
    :param T_max: end time
    :param T_step: number of steps
    :param begin_cutoff: where to begin taking values for average
    :return: Two values of infinite time average for different couplings
    '''
    sandwich_Z_i_ZZ=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP_1, n_TI_1, h_c_1,i_index_1, T_start, T_max, T_step)
    sandwich_Z_i_XX=Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP_2, n_TI_2, h_c_2,i_index_2, T_start, T_max, T_step)
    print(r'{} PXP, {} TI, {} h_c, {} Z index ZZ:'.format(n_PXP_1, n_TI_1, h_c_1, i_index_1), np.mean(sandwich_Z_i_ZZ[begin_cutoff:]))
    print(r'{} PXP, {} TI, {} h_c {} Z index XX:'.format(n_PXP_2, n_TI_2, h_c_2,i_index_2), np.mean(sandwich_Z_i_XX[begin_cutoff:]))
    return np.mean(sandwich_Z_i_ZZ[begin_cutoff:]),np.mean(sandwich_Z_i_XX[begin_cutoff:])


def Run_Z_i_System_Z_i_Bath_True_X_i_Sparse_Time_prop_plt_grid(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    plots grid of y=time, x=(-1)^(i+1) * <Z_i> running i FOR SYSTEM and BATH, and actual values as colors for extended X_i basis!!!
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
    Time = np.round(np.linspace(T_start,T_max,T_step),2)
    Z_i_array_full= np.arange(0,n_TI+n_PXP+1,1)
    Sandwich_mat=np.zeros((T_step,n_TI+n_PXP))
    for i_index in range(1,n_TI+n_PXP+1):
        if i_index<(n_PXP+1):
            Sandwich_mat[:, i_index - 1] = ((-1) ** (i_index + 1)) * Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, i_index, Initialstate, J, h_x, h_z, h_c, T_start,T_max, T_step,h_imp=0, m=2)
        else:
            Sandwich_mat[:,i_index-1]= ((-1)**(n_PXP+i_index+1))*Z_i_Bath_True_X_i_time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,i_index-n_PXP,h_imp=0, m=2)
    Sandwich_mat_fin= Sandwich_mat
    print(Sandwich_mat_fin)
    cmap = cm.get_cmap('plasma') #define cmap as 'plasma' color map (defult 256 elements in string)
    #bounds = np.linspace(-1.5,1.5,180) #define bin values to map regular values to
    #norm = colors.BoundaryNorm(bounds, cmap.N) #attaches the boundary bin values to colormap values
    plt.pcolor(Z_i_array_full,Time,Sandwich_mat_fin, shading='auto', cmap=cmap)
    plt.colorbar()
    plt.title('$<Z_i>$ for different times - $XX$ coupling strength {}'.format(np.round(h_c,2)))
    plt.ylabel('Time')
    plt.xlabel('$<Z_i>$')
    return plt.show()




############################################################################################################################################################################################
####                                                                              OVERLAP                                                                                                ###
############################################################################################################################################################################################


def Neel_Overlap_calc_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp = 0, m=2):
    '''
    Calculates different overlaps of PXP-TI Hamiltonian eigenstates with the Neel state (Neel matrix x identity)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: log10 of overlap vector and evals (ordered same manner)
    '''
    eval, evec = la.eigh(Ebe.PXP_TI_coupled_Sparse(n_PXP, n_TI,J ,h_x ,h_z ,h_c ,h_imp ,m).todense())
    Neel_state_coupled_mat= np.kron(np.outer(Neel_Subspace_Basis(n_PXP),Neel_Subspace_Basis(n_PXP)),np.identity(2**n_TI))
    overlap_vec_Neel_outer= np.diag(np.matmul(np.conjugate(np.transpose(evec)),np.matmul(Neel_state_coupled_mat,evec)))
    return eval, np.log10(overlap_vec_Neel_outer.round(32))

def Neel_Overlap_calc_PXP_TI_plt(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp = 0, m=2):
    '''
    Calculates different overlaps of PXP-TI Hamiltonian eigenstates with the Neel state (Neel matrix x identity)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: evals, log of overlap of each eigenvector with Neel state
    '''
    eval, log_overlap = Neel_Overlap_calc_PXP_TI(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp = 0, m=2)
    plt.scatter(eval,log_overlap)
    plt.xlabel('Energy')
    plt.ylabel(r'$log_{10}(|\langle\mathbb{Z}_{2}|\psi\rangle|)^{2}$')
    plt.title('Overlap of Neel State with Eigenstates vs. Energies {} PXP ZZ'.format(n_PXP))
    plt.xlim(-8,8)
    plt.ylim(-10,0)
    #plt.savefig('Neel_Overlap_{}_PXP_{}_TI_{}_h_c_ZZ.pdf'.format(n_PXP,n_TI,h_c))
    return plt.show()

def Neel_Overlap_calc_PXP_TI_True_X_i(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp = 0, m=2):
    '''
    Calculates different overlaps of PXP-TI Hamiltonian eigenstates with the Neel state (Neel matrix x identity)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: log10 of overlap vector and evals (ordered same manner)
    '''
    eval, evec = la.eigh(Ebe.PXP_TI_coupled_Sparse_Xi(n_PXP, n_TI,J ,h_x ,h_z ,h_c ,h_imp ,m).todense())
    Neel_state_coupled_mat= np.kron(np.outer(Neel_X_i_Extended_Subspace_Basis(n_PXP),Neel_X_i_Extended_Subspace_Basis(n_PXP)),np.identity(2**n_TI))
    overlap_vec_Neel_outer= np.diag(np.matmul(np.conjugate(np.transpose(evec)),np.matmul(Neel_state_coupled_mat,evec)))
    return eval, np.log10(overlap_vec_Neel_outer.round(32))

def Neel_Overlap_calc_PXP_TI_True_X_i_plt(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp = 0, m=2):
    '''
    Calculates different overlaps of PXP-TI Hamiltonian eigenstates with the Neel state (Neel matrix x identity)
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: evals, log of overlap of each eigenvector with Neel state
    '''
    eval, log_overlap = Neel_Overlap_calc_PXP_TI_True_X_i(n_PXP, n_TI, h_c, J=1, h_x=np.sin(0.485 * np.pi), h_z=np.cos(0.485 * np.pi), h_imp = 0, m=2)
    plt.scatter(eval,log_overlap)
    plt.xlim(-8,8)
    plt.ylim(-10,0)
    plt.xlabel('Energy')
    plt.ylabel(r'$log_{10}(|\langle\mathbb{Z}_{2}|\psi\rangle|^{2})$')
    plt.title('Overlap of Neel State with Eigenstates vs. Energies {} PXP XX'.format(n_PXP))
    #plt.savefig('Neel_Overlap_{}_PXP_{}_TI_{}_h_c_XX.pdf'.format(n_PXP,n_TI,h_c))
    return plt.show()


def Neel_Max_Overlap_Delta(n_PXP, n_TI, h_c):
    '''
    finds delta of energy of highest overlap states
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :anomalous_eigenstates: number is n_PXP + 1
    :return: delta of energies for highest and second highest, and second highest and third (of same side of spectrum of course)
    '''
    eval, overlap_vec = Neel_Overlap_calc_PXP_TI(n_PXP, n_TI, h_c)
    if n_PXP%2==0:
        overlap_max_vals=np.flip(np.sort(overlap_vec))[:5*(2**n_TI)] #take first 5 (*2**n_TI) anomalous eigenvalues, EXCLUDING 0
    else:
        overlap_max_vals=np.flip(np.sort(overlap_vec))[:4*(2**n_TI)] #take first 4 (*2**n_TI) anomalous eigenvalues
    overlap_max_indeces=np.squeeze(np.nonzero(np.isin(overlap_vec,overlap_max_vals)))
    max_overlap_evals=eval[overlap_max_indeces]
    overlap_max_vals_eval_ordered= overlap_vec[overlap_max_indeces]
    return max_overlap_evals



##############################################################################################################################################################################################
#                                                                  Fourier Transform - Sys and BATH!!!! WITH FAST GAMMA CALCULATING METHODS!!!!                                                                                             #
##############################################################################################################################################################################################

def Run_O_z_Sparse_Time_prop_alternate_param(n_PXP, n_TI, h_c,J,h_x,h_z,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method, with option to alternate parameters
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
    return O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)

def Run_O_z_Sparse_Time_prop_alternate_param_plt(n_PXP, n_TI, h_c,h_x,h_z,T_start, T_max, T_step):
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
    Initialstate = Neel_EBE_Haar(n_PXP, n_TI)
    J = 1
    sandwich = O_z_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.xlabel('time')
    plt.ylabel('<N|$O_z(t)$|N>')
    plt.ylim(-0.8,0.3)
    plt.title('$O_z$ Oscillations vs time for {} PXP, {} TI, $h_c=${}'.format(n_PXP,n_TI,np.round(h_c,2)))
    return plt.show()


def FFT_sys(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm):
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
    Time = np.linspace(T_start, T_max, T_step)
    VecProp = Run_O_z_Sparse_Time_prop_alternate_param(n_PXP, n_TI, h_c,J,h_x,h_z ,T_start, T_max, T_step) #Run_Time_prop uses time division
    Fourier_components= rfft(VecProp)
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=((T_max-T_start)/T_step)) # Freq * T_max = integer that multiplies 2pi
    return Freq, (np.divide(2,Height_norm) * np.abs(Fourier_components))


def Plot_FFT_sys(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Plots positive frequency fourier components as a function of (positive) frequency for O_z operator!!!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot
    '''
    Freq, Inverse_sig = FFT_sys(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_cutoff:], Inverse_sig[Start_cutoff:], marker='o', markersize=3)
    #plt.ylim(0,0.05)
    plt.xlim(0,2)
    plt.title('FFT for ZZ coup. {} PXP, {} TI, {} $h_c$'.format(n_PXP,n_TI,h_c))
    return plt.show()

def Run_O_z_Sparse_True_X_i_Time_prop_alternate_param(n_PXP, n_TI, h_c ,J,h_x,h_z,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method, with option to alternate parameters  True X_i!!
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
    return O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)

def Run_O_z_Sparse_True_X_i_Time_prop_alternate_param_plt(n_PXP, n_TI, h_c,h_x,h_z,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method True X_i!!
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
    sandwich = O_z_Sparse_True_X_i_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step,h_imp=0, m=2)
    plt.plot(np.linspace(T_start, T_max, T_step), sandwich)
    plt.xlabel('time')
    plt.ylabel('<N|$O_z(t)$|N>')
    plt.ylim(-0.8,0.3)
    plt.title('$O_z$ Oscillations vs time for {} PXP, {} TI, $h_c=${} True X_i'.format(n_PXP,n_TI,np.round(h_c,2)))
    return plt.show()

def FFT_True_X_i_sys(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm):
    '''
    Gets positive frequency (absolute value of) fourier components of the propagation signal and positive frequencies for O_z Operator!!!
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 arrays (Positive freq, positive freq fourier components)
    '''
    Time = np.linspace(T_start, T_max, T_step)
    VecProp = Run_O_z_Sparse_True_X_i_Time_prop_alternate_param(n_PXP, n_TI, h_c,J,h_x,h_z ,T_start, T_max, T_step) #Run_Time_prop uses time division
    Fourier_components= rfft(VecProp)
    #print(np.round(Fourier_components,3))
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=((T_max-T_start)/T_step)) # Freq * (T_max-T_start) = integer that multiplies 2pi
    #return Freq, (Height_norm * (np.abs(np.real(Fourier_components))+np.abs(np.imag(Fourier_components))))
    return Freq, (np.divide(2,Height_norm) * (np.abs(Fourier_components)))


def Plot_FFT_True_X_i_sys(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Plots positive frequency fourier components as a function of (positive) frequency for O_z Operator!!!:
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot
    '''
    Freq, Inverse_sig = FFT_True_X_i_sys(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_cutoff:], Inverse_sig[Start_cutoff:], marker='o', markersize=3)
    plt.title('FFT for XX coup. {} PXP, {} TI, {} $h_c$'.format(n_PXP,n_TI,h_c))
    #plt.ylim(0,0.05)
    plt.xlim(0,2)
    return plt.show()

def Plot_FFT_XX_ZZ_Compare(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Plots comparison of positive frequency fourier components as a function of (positive) frequency XX and ZZ coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: comparison plot
    '''
    Freq_ZZ, Inverse_sig_ZZ = FFT_sys(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm)
    Freq_XX, Inverse_sig_XX = FFT_True_X_i_sys(n_PXP, n_TI, h_c,J,h_x,h_z, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq_ZZ[Start_cutoff:], Inverse_sig_ZZ[Start_cutoff:], marker='o', markersize=3,
             label='ZZ Coupling')
    plt.plot(Freq_XX[Start_cutoff:], Inverse_sig_XX[Start_cutoff:], marker='o', markersize=3,
             label='XX Coupling')
    plt.ylim(0,20)
    plt.xlim(0,2)
    plt.legend()
    plt.title('FFT Comparison for ZZ XX coup. {} PXP, {} TI, {} $h_c$ '.format(n_PXP,n_TI,h_c))
    return plt.show()

def Plot_FFT_TI_param_Compare(n_PXP, n_TI, h_c,J_1,h_x_1,h_z_1,J_2,h_x_2,h_z_2, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Plots comparison of positive frequency fourier components as a function of (positive) frequency for different TI parameters, XX coup
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: comparison plot
    '''
    Freq_1, Inverse_sig_1= FFT_sys(n_PXP, n_TI, h_c,J_1,h_x_1,h_z_1, T_start, T_max, T_step, Height_norm)
    Freq_2, Inverse_sig_2 = FFT_sys(n_PXP, n_TI, h_c,J_2,h_x_2,h_z_2, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq_1[Start_cutoff:], Inverse_sig_1[Start_cutoff:], marker='o', markersize=3,
             label='{} J, {} $h_x$ {} $h_z$'.format(J_1, h_x_1, h_z_1))
    plt.plot(Freq_2[Start_cutoff:], Inverse_sig_2[Start_cutoff:], marker='o', markersize=3,
             label='{} J, {} $h_x$ {} $h_z$'.format(J_2,h_x_2,h_z_2))
    plt.ylim(0,20)
    plt.xlim(0,2)
    plt.legend()
    plt.title('FFT Comparison varying TI parameters, {} PXP, {} TI, {} $h_c$ '.format(n_PXP,n_TI,h_c))
    return plt.show()


def Plot_FFT_True_X_i_TI_param_Compare(n_PXP, n_TI, h_c,J_1,h_x_1,h_z_1,J_2,h_x_2,h_z_2, T_start, T_max, T_step, Height_norm, Start_cutoff):
    '''
    Plots comparison of positive frequency fourier components as a function of (positive) frequency for different TI parameters, XX coup
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: comparison plot
    '''
    Freq_1, Inverse_sig_1= FFT_True_X_i_sys(n_PXP, n_TI, h_c,J_1,h_x_1,h_z_1, T_start, T_max, T_step, Height_norm)
    Freq_2, Inverse_sig_2 = FFT_True_X_i_sys(n_PXP, n_TI, h_c,J_2,h_x_2,h_z_2, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq_1[Start_cutoff:], Inverse_sig_1[Start_cutoff:], marker='o', markersize=3,
             label='{} J, {} $h_x$ {} $h_z$'.format(J_1, h_x_1, h_z_1))
    plt.plot(Freq_2[Start_cutoff:], Inverse_sig_2[Start_cutoff:], marker='o', markersize=3,
             label='{} J, {} $h_x$ {} $h_z$'.format(J_2,h_x_2,h_z_2))
    plt.ylim(0,20)
    plt.xlim(0,2)
    plt.legend()
    plt.title('FFT Comparison varying TI parameters, {} PXP, {} TI, {} $h_c$ '.format(n_PXP,n_TI,h_c))
    return plt.show()

def FFT_sys_Z_i(n_PXP, n_TI, h_c,i_index, T_start, T_max, T_step, Height_norm):
    '''
    Gets positive frequency (absolute value of) fourier components of <Z_i> time signal and positive frequencies ZZ coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 arrays total, consisting of: Positive freq, positive freq fourier components
    '''
    Time = np.linspace(T_start, T_max, T_step)
    VecProp = Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP, n_TI, h_c,i_index ,T_start, T_max, T_step) #Run_Time_prop uses time division
    Fourier_components= rfft(VecProp)
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=((T_max-T_start)/T_step)) # Freq * T_max = integer that multiplies 2pi
    return Freq, (Height_norm * np.abs(Fourier_components))

def Plot_FFT_sys_Z_i(n_PXP, n_TI, h_c,i_index,Start_Cutoff,End_Cutoff,T_start, T_max, T_step, Height_norm):
    '''
    Plots positive frequency fourier components of <Z_i> as a function of (positive) frequency ZZ Coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot
    '''
    Freq, Inverse_sig = FFT_sys_Z_i(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_Cutoff:End_Cutoff], Inverse_sig[Start_Cutoff:End_Cutoff], marker='o', markersize=3)
    plt.title('FFT for ZZ coup. {} PXP, {} TI, {} $h_c$, site {}'.format(n_PXP,n_TI,h_c,i_index))
    return plt.show()


def FFT_Z_i_Gamma_plot_fast(n_PXP, n_TI, h_c_max,i_index,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step):
    '''
    Plots gamma as a function of coupling for given number of atoms ZZ coupling NOT LOADING, CALCULATING FROM SCARTCH
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c_max: max value of coupling strength
    :param i_index: index of Z_i operator
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
    sandwich_Z_i_gamma_0_sys = Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP,n_TI,0.0,i_index,T_start,T_max,T_step)
    Fourier_components_Z_i_gamma_0_sys = rfft(sandwich_Z_i_gamma_0_sys[:fourier_cutoff_div])
    Freq = rfftfreq(fourier_cutoff_div, d=((T_max-T_start) / T_step))  # 1/(T_max-T_start) is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_Z_i_gamma_0_sys[Optimal_start:Optimal_end]))
    for h_c in np.nditer(h_c_array):
        sandwich_Z_i_sys=Run_Z_i_System_PXP_TI_Sparse_time_prop(n_PXP,n_TI,np.round(h_c,2),i_index,T_start,T_max,T_step)
        Fourier_components_Z_i= rfft(sandwich_Z_i_sys[:fourier_cutoff_div])
        popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_Z_i[Optimal_start:Optimal_end]))  # popt= parameter optimal values
        gamma_array[count]=popt[1]/popt_gamma_0_sys[1]
        gamma_pcov_array[count]=pcov[1,1]/popt_gamma_0_sys[1]
        count=count+1
    plt.plot(h_c_array, gamma_array, marker='o', markersize=1)
    plt.title('Damping coef vs Coup str, ZZ Coup for {} PXP, {} TI, site {}, {} Time, {} Steps'.format(n_PXP,n_TI,i_index,T_max,T_step))
    plt.xlabel('Coupling Strength')
    plt.ylabel('Damping Coefficient (Normalized)')
    plt.ylim(0.5,3)
    return plt.show()

def FFT_Z_i_True_X_X_Gamma_plot_fast(n_PXP, n_TI, h_c_max,i_index,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step):
    '''
    Plots gamma as a function of coupling for given number of atoms XX coupling NOT LOADING, CALCULATING FROM SCARTCH
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c_max: max value of coupling strength
    :param i_index: index of Z_i operator
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal
    :param Optimal_start: fit start cutoff for fourier signal FIT
    :param Optimal_end: fit end cutoff for fourier signal FIT
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
    Freq = rfftfreq(fourier_cutoff_div, d=((T_max-T_start)/ T_step))  # 1/(T_max-T_start) is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    sandwich_Z_i_gamma_0_sys = Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP,n_TI,0.0,i_index,T_start,T_max,T_step)
    Fourier_components_Z_i_gamma_0_sys = rfft(sandwich_Z_i_gamma_0_sys[:fourier_cutoff_div])
    popt_gamma_0_sys, pcov_gamma_0_sys = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_Z_i_gamma_0_sys[Optimal_start:Optimal_end]))
    for h_c in np.nditer(h_c_array):
        sandwich_Z_i_sys= Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP,n_TI,np.round(h_c,2),i_index,T_start,T_max,T_step)
        Fourier_components_Z_i= rfft(sandwich_Z_i_sys[:fourier_cutoff_div])
        popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_Z_i[Optimal_start:Optimal_end]))  # popt= parameter optimal values
        gamma_array[count]=popt[1]/popt_gamma_0_sys[1]
        gamma_pcov_array[count]=pcov[1,1]/popt_gamma_0_sys[1]
        count=count+1
    plt.plot(h_c_array, gamma_array, marker='o', markersize=1)
    plt.title('Damping coef vs Coup str, $XX$ Coup for {} PXP, {} TI, site {}, {} Max Time, {} Steps'.format(n_PXP,n_TI,i_index, T_max,T_step))
    plt.xlabel('Coupling Strength')
    plt.ylabel('Damping Coefficient (Normalized)')
    plt.ylim(0.5,3)
    return plt.show()

def FFT_True_X_i_sys_Z_i(n_PXP, n_TI, h_c,i_index, T_start, T_max, T_step, Height_norm):
    '''
    Gets positive frequency (absolute value of) fourier components of <Z_i> time signal and positive frequencies XX Coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: 2 arrays total, consisting of Positive freq, positive freq fourier components
    '''
    Time = np.linspace(T_start, T_max, T_step)
    VecProp = Run_Z_i_System_PXP_TI_True_X_i_Sparse_time_prop(n_PXP, n_TI, h_c,i_index ,T_start, T_max, T_step) #Run_Time_prop uses time division
    Fourier_components= rfft(VecProp)
    Sig_size= np.size(VecProp)
    Freq = rfftfreq(Sig_size, d=((T_max-T_start)/T_step)) # Freq * (T_max-T_start) = integer that multiplies 2pi
    return Freq, (Height_norm * np.abs(Fourier_components))

def Plot_FFT_True_X_i_sys_Z_i(n_PXP, n_TI, h_c,i_index,Start_Cutoff,End_Cutoff, T_start, T_max, T_step, Height_norm):
    '''
    Plots positive frequency fourier components of <Z_i> as a function of (positive) frequency XX Coupling
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c: coupling strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot
    '''
    Freq, Inverse_sig = FFT_True_X_i_sys_Z_i(n_PXP, n_TI, h_c, i_index, T_start, T_max, T_step, Height_norm)
    plt.plot(Freq[Start_Cutoff:End_Cutoff], Inverse_sig[Start_Cutoff:End_Cutoff], marker='o', markersize=3)
    plt.title('FFT for XX coup. {} PXP, {} TI, {} $h_c$, site {}'.format(n_PXP,n_TI,h_c, i_index))
    return plt.show()

def FFT_Gamma_plot_fast(n_PXP, n_TI, h_c_max,h_c_interval,h_x,h_z,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step):
    '''
    Plots gamma as a function of coupling for given number of atoms ZZ coupling (for O_z full operator)
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c_max: max value of coupling strength
    :param h_c_interval: interval of h_c jumps
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal)
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot of gamma vs h_c strength
    '''
    time=np.linspace(T_start,T_max,T_step)
    h_c_array= np.arange(0.0,h_c_max,h_c_interval)
    J=1
    gamma_array= np.zeros(len(h_c_array))
    gamma_pcov_array= np.zeros(len(h_c_array))
    count=0
    Freq = rfftfreq(fourier_cutoff_div, d=((T_max-T_start) / T_step))  # 1/(T_max-T_start) is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    for h_c in np.nditer(h_c_array):
        sandwich_O_z_sys=Run_O_z_Sparse_Time_prop_alternate_param(n_PXP, n_TI, np.around(h_c,2),J,h_x,h_z,T_start, T_max, T_step)
        Fourier_components_O_z= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
        popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z[Optimal_start:Optimal_end]))  # popt= parameter optimal values
        gamma_array[count]=popt[1]
        gamma_pcov_array[count]=pcov[1,1]
        count=count+1
    plt.plot(h_c_array, gamma_array, marker='o', markersize=1)
    plt.title('Damping coef vs Coup str, ZZ Coup for {} PXP, {} TI, {} Time, {} Steps'.format(n_PXP,n_TI,T_max,T_step))
    plt.xlabel('Coupling Strength')
    plt.ylabel('Damping Coefficient (Normalized)')
    #plt.ylim(0.5,3)
    return plt.show()

def FFT_True_X_X_Gamma_plot_fast(n_PXP, n_TI, h_c_max,h_c_interval,h_x,h_z,fourier_cutoff_div,Optimal_start,Optimal_end,T_start,T_max,T_step):
    '''
    Plots gamma as a function of coupling for given number of atoms XX coupling (for O_z full operator) WITHOUT LOADING OSC Realization
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param h_c_max: max value of coupling strength
    :param h_c_interval: interval of h_c jumps
    :param fourier_cutoff_div: cutoff of timeseries for fourier transform (if we want to use only part of the signal)
    :param Optimal_start: start cutoff for fourier signal FIT
    :param Optimal_end: end cutoff for fourier signal FIT
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :return: plot of gamma vs h_c strength
    '''
    time=np.linspace(T_start,T_max,T_step)
    h_c_array= np.arange(0.0,h_c_max,h_c_interval)
    J=1
    gamma_array= np.zeros(len(h_c_array))
    gamma_pcov_array= np.zeros(len(h_c_array))
    count=0
    Freq = rfftfreq(fourier_cutoff_div, d=((T_max-T_start) / T_step))  # 1/(T_max-T_start) is step sample distance (interval). max freq is 1/(2*d), there are T_step/2 frequency samples.
    for h_c in np.nditer(h_c_array):
        sandwich_O_z_sys=Run_O_z_Sparse_True_X_i_Time_prop_alternate_param(n_PXP, n_TI, np.around(h_c,2),J,h_x,h_z,T_start, T_max, T_step)
        Fourier_components_O_z= rfft(sandwich_O_z_sys[:fourier_cutoff_div])
        popt, pcov = curve_fit(Lorentzian_function, Freq[Optimal_start:Optimal_end],np.abs(Fourier_components_O_z[Optimal_start:Optimal_end]))  # popt= parameter optimal values
        gamma_array[count]=popt[1]
        gamma_pcov_array[count]=pcov[1,1]
        count=count+1
    plt.plot(h_c_array, gamma_array, marker='o', markersize=1)
    plt.title('Damping coef vs Coup str, XX Coup for {} PXP, {} TI, {} Time, {} Steps'.format(n_PXP,n_TI,T_max,T_step))
    plt.xlabel('Coupling Strength')
    plt.ylabel('Damping Coefficient (Normalized)')
    #plt.ylim(0.5,3)
    return plt.show()

##############################################################################################################################################################################################
##                                                                  OLD METHODS - TO BE DELETED?                                                                                             #
##############################################################################################################################################################################################



def ZiSandwichCheck2(Ham, n_PXP, n_TI,  Initialstate,
             T_start, T_max, T_step, i, Color, Marker):
    '''
    returns 2x 1D arrays: time propagated output for every delta_t, and the corresponding vector of t we defined
    :param Ham: Hamiltonian for propagation
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (interval)
    :param i: Z_i site choice
    :param Color:
    :param Marker:
    :return: vector of V (time propagated output for every delta t) and the corresponding vector of t we defined
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    v_ket= Initialstate
    v_bra= Initialstate
    t = np.arange(T_start, T_max, T_step)
    VecProp=np.zeros((np.size(t)))
    for ti in np.nditer(t):
        v_ket = np.dot(U,v_ket) # propagation in iterations from here
        v_bra = np.dot(U,v_bra) # propagation in iterations from here
        VecProp[np.argwhere(t == ti)] = np.round(np.vdot(v_bra,np.dot(Z_generali(n_PXP+n_TI,i),v_ket)),4)
    return t, VecProp

def RunZiSandwichCheck2(n_PXP, n_TI, i, Coupl=Z_i, J=1 , h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_c=0, T_start=0, T_max=100, T_step=0.05, h_imp=0.01, m=1 ):# 1 Time propagation of PXP TI COUPLED
    """
    Runs Z1SandwichCheck
    """
    H_full = PXPBathHam2(n_PXP, n_TI, Coupl, J, h_x, h_z, h_c, h_imp, m)
    InitVecstate = NeelHaar(n_PXP,n_TI)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.array(('s', '^', 'o', 'd'))
    return ZiSandwichCheck2(H_full, n_PXP, n_TI, InitVecstate, T_start, T_max, T_step, i, Color, np.random.choice(markers))

def ZiSandwichCheck2plt(Ham, n_PXP, n_TI,  Initialstate,
             T_start, T_max, T_step, i, Color, Marker):
    '''
    plots <Neel|Z_i(t)|Neel> with respect to time and returns 2x 1D arrays
    :param Ham: Hamiltonian for propagation
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (interval)
    :param i: Z_i site choice
    :param Color:
    :param Marker:
    :return: plot and vector of V (time propagated output for every delta t) + the corresponding vector of t we defined
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    v_ket= Initialstate
    v_bra= Initialstate
    t = np.arange(T_start, T_max, T_step)
    VecProp=np.zeros((np.size(t)))
    for ti in np.nditer(t):
        v_ket = np.dot(U,v_ket) # propagation in iterations from here
        v_bra = np.dot(U,v_bra) # propagation in iterations from here
        VecProp[np.argwhere(t == ti)] = np.round(np.vdot(v_bra,np.dot(Z_generali(n_PXP+n_TI,i),v_ket)),4)
    plt.plot(t, VecProp, marker=Marker, markersize=3,
             color=Color)
    return t, VecProp, Color

def RunZiSandwichCheck2plt(n_PXP, n_TI, i, Coupl=Z_i, J=1 , h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_c=0, T_start=0, T_max=100, T_step=0.05, h_imp=0.01, m=1 ):# 1 Time propagation of PXP TI COUPLED
    """
    Runs Z1SandwichCheck plotter!
    """
    H_full = PXPBathHam2(n_PXP, n_TI, Coupl, J, h_x, h_z, h_c, h_imp, m)
    InitVecstate = NeelHaar(n_PXP,n_TI)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.array(('s', '^', 'o', 'd'))
    return ZiSandwichCheck2plt(H_full, n_PXP, n_TI, InitVecstate, T_start, T_max, T_step, i, Color, np.random.choice(markers))

def OzPXPOBConlySandwichplt(Ham, n_PXP, Initialstate, T_start, T_max, T_step, Color, Marker):
    '''
    plots <Neel|O_z(t)|Neel> with respect to time, FOR PXP OBC only
    :param Ham: Hamiltonian for propagation
    :param n_PXP: Size of PXP chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (interval)
    :param Color:
    :param Marker:
    :return: plot only
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    V= Initialstate
    t = np.arange(T_start, T_max, T_step)
    VecProp=np.zeros((np.size(t)))
    for ti in np.nditer(t):
        if ti==0:
            VecProp[np.argwhere(t == ti)] = np.round(np.vdot(V, np.dot(O_znew(n_PXP),V)), 4)
        else:
            V = np.dot(U,V) # propagation in iterations from here
            VecProp[np.argwhere(t == ti)] = np.round(np.vdot(V,np.dot(O_znew(n_PXP),V)),4)
    plt.plot(t, VecProp, marker=Marker, markersize=3,
             color=Color)
    return plt.show()


def RunOzPXPOBConlySandwichplt(n_PXP, T_start, T_max, T_step, j=2, st=0):
    '''
    Runs O_z time propagation for PXP model only (plotter)
    :param n_PXP: No of PXP atoms
    :param j: Impurity site for PXP model
    :param st: Impurity strength
    :param T_start: Start propagation time
    :param T_max: Max time
    :param T_step: Step interval of time
    :return: OzPXPOBConlySandwichplt
    '''
    Ham = PXPOBCNew2_Impure(n_PXP, j, st)
    InitVecstate = Neelstate(n_PXP)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.array(('s', '^', 'o', 'd'))
    return OzPXPOBConlySandwichplt(Ham, n_PXP, InitVecstate, T_start, T_max, T_step, Color, np.random.choice(markers))


def NeelstateFidelityplt(Ham, Initialstate, T_start, T_max, T_step, Color, Marker):
    '''
    Plots |<N(t)|N(0)>|^2 as function of time (quantum fidelity) ONLY PXP
    :param Ham: Hamiltonian
    :param Initialstate: Initial state input
    :param T_start: Start time
    :param T_max: Max time
    :param T_step: Time step interval
    :param Color:
    :param Marker:
    :return: plot
    '''
    U = expm(-1j*Ham*T_step)
    V_t = Initialstate
    V = Initialstate
    t = np.arange(T_start, T_max, T_step)
    VecProp=np.zeros((np.size(t)))
    for ti in np.nditer(t):
        if ti==0:
            InitSandwich = np.vdot(V_t,V)
            VecProp[np.argwhere(t == ti)] = np.round(np.multiply(np.conjugate(InitSandwich), InitSandwich), 4)
        else:
            V_t= np.dot(U,V_t) # propagation in iterations from here
            Sandwich = np.vdot(V_t,V)
            VecProp[np.argwhere(t == ti)] = np.round(np.multiply(Sandwich, np.conjugate(Sandwich)), 4)
    plt.plot(t, VecProp, marker=Marker, markersize=3,
             color=Color)
    return plt.show()


def RunNeelstateFidelityplt(n_PXP, T_start=0, T_max=2000, T_step=0.05, j=2, st=0):
    '''
    Runs Neel state Fidelity plotter ONLY PXP
    :param n_PXP: No of PXP atoms
    :param j: Impurity site for PXP model
    :param st: Impurity strength
    :param T_start: Start propagation time
    :param T_max: Max time
    :param T_step: Step interval of time
    :return: NeelstateFidelityplt
    '''
    Ham = PXPOBCNew2_Impure(n_PXP, j, st)
    InitVecstate = Neelstate(n_PXP)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.array(('s', '^', 'o', 'd'))
    return NeelstateFidelityplt(Ham, InitVecstate, T_start, T_max, T_step, Color, np.random.choice(markers))

def OzPXPOBConlySandwich(Ham, n_PXP, Initialstate, T_start, T_max, T_step):
    '''
    Returns <Neel|O_z(t)|Neel> values and corresponding time values, FOR PXP OBC only
    :param Ham: Hamiltonian for propagation
    :param n_PXP: Size of PXP chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (interval)
    :return: 2 vectors -  <Neel|O_z(t)|Neel> values and corresponding time values
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    V= Initialstate
    t = np.arange(T_start, T_max, T_step)
    VecProp=np.zeros((np.size(t)))
    for ti in np.nditer(t):
        if ti==0:
            VecProp[np.argwhere(t == ti)] = np.round(np.vdot(V, np.dot(O_znew(n_PXP),V)), 4)
        else:
            V = np.dot(U,V) # propagation in iterations from here
            VecProp[np.argwhere(t == ti)] = np.round(np.vdot(V,np.dot(O_znew(n_PXP),V)),4)
    return t, VecProp


def RunOzPXPOBConlySandwich(n_PXP, T_start, T_max, T_step, j=2, st=0):
    '''
    Runs O_z time propagation for PXP model only
    :param n_PXP: No of PXP atoms
    :param j: Impurity site for PXP model
    :param st: Impurity strength
    :param T_start: Start propagation time
    :param T_max: Max time
    :param T_step: Step interval of time
    :return: OzPXPOBConlySandwich
    '''
    Ham = PXPOBCNew2_Impure(n_PXP, j, st)
    InitVecstate = Neelstate(n_PXP)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.array(('s', '^', 'o', 'd'))
    return OzPXPOBConlySandwich(Ham, n_PXP, InitVecstate, T_start, T_max, T_step)


def OzSandwichTotHam(Ham, n_PXP, n_TI,  Initialstate,
             T_start, T_max, T_step):
    '''
    Returns <Neel|O_z(t)|Neel> values and corresponding time values
    :param Ham: Hamiltonian for propagation
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (interval)
    :return: 2 vectors -  <Neel|O_z(t)|Neel> values and corresponding time values
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    V= Initialstate
    t = np.arange(T_start, T_max, T_step)
    VecProp=np.zeros((np.size(t)))
    for ti in np.nditer(t):
        if ti==0:
            VecProp[np.argwhere(t == ti)] = np.round(np.vdot(V, np.dot(np.kron(O_znew(n_PXP),np.identity(2**n_TI)),V)), 4)
        else:
            V = np.dot(U,V) # propagation in iterations from here
            VecProp[np.argwhere(t == ti)] = np.round(np.vdot(V,np.dot(np.kron(O_znew(n_PXP),np.identity(2**n_TI)),V)),4)
    return t, VecProp

def RunOzSandwichTotHam(n_PXP, n_TI, h_c, T_start, T_max, T_step, Coupl=Z_i, J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0, m=2):# Time propagation of PXP TI COUPLED
    '''
     Runs OzSandwichTotHam!
    :param n_PXP: No of PXP atoms
    :param n_TI: No of TI atoms
    :param Coupl:Coupling nature (type of matrix)
    :param J: TI Ising part strength
    :param h_x: Trasverse field
    :param h_z: Longtitudinal field
    :param h_c: Coupling strength
    :param T_start: Start time of measurement
    :param T_max: End time of measurement
    :param T_step: time step interval
    :param h_imp: Impurity strength in TI model
    :param m: Site of impurity of TI model (should NOT be 1)
    :return: t array and array of corresponding <Neel|O_z(t)|Neel> values
    '''
    H_full = PXPBathHam2(n_PXP, n_TI, Coupl, J, h_x, h_z, h_c, h_imp, m)
    InitVecstate = NeelHaar(n_PXP,n_TI)
    return OzSandwichTotHam(H_full, n_PXP, n_TI, InitVecstate, T_start, T_max, T_step)

def OzSandwichTotHamplt(Ham, n_PXP, n_TI,  Initialstate,
             T_start, T_max, T_step, Color, Marker, h_c):
    '''
    plots <Neel|O_z(t)|Neel> with respect to time
    :param Ham: Hamiltonian for propagation
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (interval!!!!!!!!!!)
    :param Color:
    :param Marker:
    :return: plot only
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    V= Initialstate
    t = np.arange(T_start, T_max, T_step)
    VecProp=np.zeros((np.size(t)))
    for ti in np.nditer(t):
        if ti==0:
            VecProp[np.argwhere(t == ti)] = np.round(np.vdot(V, np.dot(np.kron(O_znew(n_PXP),np.identity(2**n_TI)),V)), 4)
        else:
            V = np.dot(U,V) # propagation in iterations from here
            VecProp[np.argwhere(t == ti)] = np.round(np.vdot(V,np.dot(np.kron(O_znew(n_PXP),np.identity(2**n_TI)),V)),4)
    plt.plot(t, VecProp, marker=Marker, markersize=3,
             color=Color)
    plt.title('{} PXP atoms, {} TI atoms, {} Coupling strength'.format(n_PXP,n_TI, h_c))
    return plt.show()

def RunOzSandwichTotHamplt(n_PXP, n_TI, h_c, T_start, T_max, T_step, Coupl=Z_i, J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0, m=2):# Time propagation of PXP TI COUPLED
    '''
     Runs OzSandwichCheck plotter!
    :param n_PXP: No of PXP atoms
    :param n_TI: No of TI atoms
    :param Coupl:Coupling nature (type of matrix)
    :param J: TI Ising part strength
    :param h_x: Trasverse field
    :param h_z: Longtitudinal field
    :param h_c: Coupling strength
    :param T_start: Start time of measurement
    :param T_max: End time of measurement
    :param T_step: time step interval!!!!!!!
    :param h_imp: Impurity strength in TI model
    :param m: Site of impurity of TI model (should NOT be 1)
    :return: OzSandwichCheckplt
    '''
    H_full = PXPBathHam2(n_PXP, n_TI, Coupl, J, h_x, h_z, h_c, h_imp, m)
    InitVecstate = NeelHaar(n_PXP,n_TI)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.array(('s', '^', 'o', 'd'))
    return OzSandwichTotHamplt(H_full, n_PXP, n_TI, InitVecstate, T_start, T_max, T_step, Color, np.random.choice(markers),h_c)


def Averagesig(n_PXP, n_TI, h_c, T_start=0, T_max=100, T_step=1):
    '''
    Arithmetic mean signal amplitude calculation
    :param n_PXP:number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :return: average of samples
    '''
    t, VecProp = RunOzSandwichTotHam(n_PXP, n_TI, h_c, T_start, T_max, T_step, Coupl=Z_i, J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0, m=2)
    return np.average(VecProp)

def Peakfinder(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    Finding peaks (maximum) of <Neel|O_z(t)|Neel> graph
    :param n_PXP:number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :return: t values of peaks and corresponding peak values.
    '''
    t, VecProp = RunOzSandwichTotHam(n_PXP, n_TI, h_c, T_start, T_max, T_step, Coupl=Z_i, J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0, m=2)
    Peakindeces, heights = find_peaks(VecProp, height= -0.35) #TODO change to avgsig?
    height_array = heights['peak_heights']
    time_allpeaks = np.take(t,Peakindeces)
    return time_allpeaks, height_array

def PeakfinderPXPOBC(n_PXP, T_start, T_max, T_step):
    '''
    Finding peaks (maximum) of <Neel|O_z(t)|Neel> graph for PXP OBC only
    :param n_PXP:number of PXP atoms
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :return: t values of peaks and corresponding peak values.
    '''
    t, VecProp = RunOzPXPOBConlySandwich(n_PXP, T_start, T_max, T_step, j=2, st=0)
    Peakindeces, heights = find_peaks(VecProp, height= -0.35) #TODO change to avgsig?
    height_array = heights['peak_heights']
    time_allpeaks = np.take(t,Peakindeces)
    return time_allpeaks, height_array

def Peakfinderplt(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    Plotting peak values vs t values of these peaks
    :param n_PXP:number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :return: Plot
    '''
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    t, VecProp = RunOzSandwichTotHam(n_PXP, n_TI, h_c, T_start, T_max, T_step, Coupl=Z_i, J=1, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0, m=2)
    Peakindeces, heights = find_peaks(VecProp, height=-0.35)  # TODO change to avgsig?
    height_array = heights['peak_heights']
    time_allpeaks = np.take(t, Peakindeces)
    plt.plot(time_allpeaks, height_array, color=Color, marker='o')
    plt.title('{} PXP atoms, {} TI atoms, {} Coupling strength'.format(n_PXP,n_TI, h_c))
    return plt.show()


def MinimumPeak(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    finds the minimum peak value and t value
    :param n_PXP:number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :return: index of Minimum Peak , array of times of peaks and of peak heights
    '''
    Time_peaks, Height_array = Peakfinder(n_PXP, n_TI, h_c, T_start, T_max, T_step)
    Minpeak = np.argmin(Height_array)
    return Time_peaks, Height_array, Minpeak



def DampingCoef(n_PXP, n_TI, h_c, T_start, T_max, T_step):
    '''
    calculates damping coefficient as |(Y_fin-Y_init)/(X_fin-X_init)| - the linear slope of graph
    :param n_PXP:number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :return: damping coefficient
    '''
    T_peaks, Height_peaks, Min_peak= MinimumPeak(n_PXP, n_TI, h_c, T_start, T_max, T_step)
    Coef = np.absolute(np.divide(Height_peaks[Min_peak]-Height_peaks[0],T_peaks[Min_peak]-T_peaks[0]))
    return Coef

def DampingStr(n_PXP, n_TI, h_c, T_start, T_max, T_step, T_cap):
    '''
    Calculates damping between t=0 and t=x (fixed interval) as Height differences (absolute value)
    :param n_PXP:number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :param T_cap: Time cap for damping measurement
    :return: scalar, absolute value of peak heights
    '''
    Time_peaks, Height_array = Peakfinder(n_PXP, n_TI, h_c, T_start, T_max, T_step)
    T_cap_index = (np.absolute(Time_peaks-T_cap)).argmin()
    Dampingstrength = np.absolute(Height_array[T_cap_index]-Height_array[0])
    return Dampingstrength

def PXPOnlyDampingStr(n_PXP, T_start, T_max, T_step, T_cap):
    '''
    Calculates damping between t=0 and t=x (fixed interval) as Height differences (absolute value)
    :param n_PXP:number of PXP atoms
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :param T_cap: Time cap for damping measurement
    :return: scalar, absolute value of peak heights
    '''
    Time_peaks, Height_array = PeakfinderPXPOBC(n_PXP, T_start, T_max, T_step)
    T_cap_index = (np.absolute(Time_peaks-T_cap)).argmin()
    PXPOnlyDampingstrength = np.absolute(Height_array[T_cap_index]-Height_array[0])
    return PXPOnlyDampingstrength

def ScaledDampingStr(n_PXP, n_TI, h_c, T_start, T_max, T_step, T_cap):
    '''
    Calculates Scaled (by the pxp_OBC case) of damping between t=0 and t=x (fixed interval) as Height differences (absolute value)
    :param n_PXP:number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :param T_cap: Time cap for damping measurement
    :return: scalar, absolute value of peak heights
    '''
    Damping_strength = DampingStr(n_PXP, n_TI, h_c, T_start, T_max, T_step, T_cap)
    PXP_Only_Damping_strength = PXPOnlyDampingStr(n_PXP, T_start, T_max, T_step, T_cap)
    Scaled_Damping = np.divide(Damping_strength,PXP_Only_Damping_strength)
    return Scaled_Damping

def ThresholdPeak(n_PXP, n_TI, h_c, T_start, T_max, T_step, Threshold=-0.3):
    '''
    Finds closest peak to treshold (above or below..) and returns index
    :param n_PXP:number of PXP atoms
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: step interval
    :param Threshold: Threshold setting
    :return: Array of times of peaks and of peak heights, and Treshold peak index
    '''
    Time_peaks, Height_array = Peakfinder(n_PXP, n_TI, h_c, T_start, T_max, T_step)
    Threshold_array = np.argwhere(Height_array < Threshold)
    Threshold_index = Threshold_array[0]
    return Time_peaks, Height_array, Threshold_index

def DampingLength(n_PXP, n_TI, h_c, T_start, T_max, T_step,Threshold=-0.3):
    '''
    NEED TO INPUT!
    :param n_PXP:
    :param n_TI:
    :param h_c:
    :param T_start:
    :param T_max:
    :param T_step:
    :param Threshold:
    :return:
    '''
    Time_peaks, Height_array, Threshold_index = ThresholdPeak(n_PXP, n_TI, h_c, T_start, T_max, T_step, Threshold)
    Length= np.absolute(Time_peaks[Threshold_index]-Time_peaks[0])
    return print(Length)

def PXPonlyDampingLength(n_PXP, T_start, T_max, T_step,Threshold=-0.3):
    '''
    Need TO INPUT
    :param n_PXP:
    :param T_start:
    :param T_max:
    :param T_step:
    :param Threshold:
    :return:
    '''
    Time_peaks, Height_array, Threshold_index = ThresholdPeak(n_PXP, 0, 0, T_start, T_max, T_step, Threshold)
    Length= np.absolute(Time_peaks[Threshold_index]-Time_peaks[0])
    return Length

def ScaledDampingLength(n_PXP, n_TI, h_c, T_start, T_max, T_step,Threshold=-0.3): #TODO problem with PXP length finder
    '''
    Need TO INPUT
    :param n_PXP:
    :param n_TI:
    :param h_c:
    :param T_start:
    :param T_max:
    :param T_step:
    :param Threshold:
    :return:
    '''
    PXP_only_Length = DampingLength(n_PXP, n_TI, h_c, T_start, T_max, T_step, Threshold=-0.3) #TODO MAKBILI
    Length = PXPonlyDampingLength(n_PXP, T_start, T_max, T_step, Threshold=-0.3)
    Scaled_length = np.divide(Length,PXP_only_Length)
    return Scaled_length




# def ScaledDampinglength(n_PXP, n_TI, h_c, T_start, T_max, T_step, Cap=5):
#     '''
#     Scaled damping length (can be used for comparing different TI atom numbers/ different coupling strength)
#     :param n_PXP: No. of PXP atoms
#     :param n_TI: No. of TI atoms
#     :param Cap: peak cap for taking length interval
#     :param i: Z_i site
#     :return: Scalar from 0 to 1 (indicating the relative damping length to the pure PXP one)
#     '''
#     PurePXPlength = Dampinglength(n_PXP, 0, Cap, 0, T_start, T_max, T_step)
#     Damplength = Dampinglength(n_PXP, n_TI, Cap, h_c, T_start, T_max, T_step)
#     Scaledlength= np.divide(Damplength,PurePXPlength)
#     return Scaledlength

# def PlotDampinglengthTIno(n_PXP, n_TI, h_c, Cap=5, i=1):
#     '''
#
#     :param n_PXP: No. of PXP atoms
#     :param n_TI: No. of TI atoms
#     :param Cap: peak cap for taking length interval
#     :param i: Z_i site
#     :return: Plot
#
#     '''
#     for n in np.nditer(n_TI):
#         Scaledlength= ScaledDampinglength(n_PXP, n, h_c, Cap, i)
#         plt.plot(n,Scaledlength, color='black', marker='o')
#         plt.xlabel('No. of TI atoms')
#         plt.ylabel('Damping Length')
#     plt.show()
#     return

# def PlotDampinglengthCoupstr(n_PXP, n_TI, h_c, Cap=5, i=1):
#     '''
#     :param n_PXP: No. of PXP atoms
#     :param n_TI: No. of TI atoms
#     :param Cap: peak cap for taking length interval
#     :param i: Z_i site
#     :return:
#     '''
#     for h in np.nditer(h_c):
#         Scaledlength= ScaledDampinglength(n_PXP, n_TI, h, Cap, i)
#         plt.plot(h ,Scaledlength, color='blue', marker='o')
#         plt.xlabel('No. of TI atoms')
#         plt.ylabel('Damping Length')
#     plt.show()
#     return

####### New fit method - FFT #######
