import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
from scipy.linalg import expm
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from Coupling_To_Bath import *

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

# TODO think about the problem with the values for different Impurity strength and different atom chain sizes?

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
     Runs Z1SandwichCheck plotter!
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
    :return: OzSandwichCheckplt
    '''
    H_full = PXPBathHam2(n_PXP, n_TI, Coupl, J, h_x, h_z, h_c, h_imp, m)
    InitVecstate = NeelHaar(n_PXP,n_TI)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.array(('s', '^', 'o', 'd'))
    return OzSandwichTotHamplt(H_full, n_PXP, n_TI, InitVecstate, T_start, T_max, T_step, Color, np.random.choice(markers),h_c)

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

########################## Damping coefficient measuring ##################################

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
    :param n_TI: number of TI atoms
    :param h_c: coupling strength
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
    :param Treshold: Threshold setting
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
#TODO Write the scaled damping length differently so that the PXP length will only be calculated once.

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

############################## END OF OSCILLATIONS FINDING SECTION #################################
