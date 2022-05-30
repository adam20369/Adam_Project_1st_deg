import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from scipy.linalg import expm
from matplotlib.lines import Line2D
from scipy.signal import find_peaks


# ========================== definitions Dim=1 ======================
g = np.array([0, 1])  # ground state
# print("ground=\n", g)

r = np.array([1, 0])  # rydberg state
# print("rydberg=\n", r)

X_i = np.array([[0, 1], [1, 0]])  # flips ground -> rydberg/ rydberg-> ground |g><r|+|r><g|
# print("X_i = \n", X_i)

Q_i = np.array([[1, 0], [0, 0]])  # projector on rydberg state (density of excited states) |r><r|
# print("Q_i = \n", Q_i)

P_i = np.array([[0, 0], [0, 1]])  # projector on ground state (density of ground states) |g><g|
# print("P_i = \n",P_i)

Z_i = np.array([[1, 0], [0, -1]])  # pauli Z


# ======================== declarations of some operators in D=2**n Hilbert (Kronecker) space ===========================

def Z_1(n):
    Z_1 = np.kron(Z_i, np.identity(2**(n-1)))
    return Z_1
# print(Z_1(3))

def Z_generali(n,i): # calculates Z for general i
    '''

    :param n: dimension 2**n
    :param i: atom index of Z operator
    :return: Z_i in dimension 2**n
    '''
    if i <= n:
        Z_geni = np.kron(np.identity(2**(i-1)),np.kron(Z_i, np.identity(2**(n-i))))
    else:
        Z_geni= np.identity(2**n)
    return Z_geni
# print(Z_1(3))

def O_znew(n): #mean of Z_i's
    '''
    Arithmetic Mean of Z_i's
    :param n: number of atoms (2**n is dimension)
    :return: O_z
    '''
    d = 2 ** n
    Z_sum= np.array(0)
    for i in range(1,n+1):
        Z_sum= Z_sum +Z_generali(n,i)
    O_zop= (1/n) * Z_sum
    return O_zop

def Neelstate(n): # GENERAL NEELSTATE
    d = 2 ** n
    Neel= np.array(1)
    Even= np.arange(0,n,2)
    for i in range(0,n):
        if i in Even:
            Neel = np.kron(Neel, r)
        else:
            Neel = np.kron(Neel, g)
    return Neel

def Haarstate(n): # GENERAL HAAR STATE
    '''
    General Haar state, A Haar state is a state that simulates an "average" eigenstate of the system,
    since it replaces an arithmetic mean over all eigenstates of an observable
    :param n: number of atoms
    :return: Haar Vector
    '''
    if n == 0:
         HaarVec= np.array(1)
    else:
        d = 2 ** n
        alpha= np.random.normal(0, 1, 2 ** n)
        betta= np.random.normal(0, 1, 2 ** n)
        v= alpha + 1j*betta
        HaarVec= np.divide(v, la.norm(v))
    return HaarVec

def NeelHaar(n_PXP, n_TI):
    """
    Combination of the Neel and Haar states
    :param n_PXP:  Number of PXP chain atoms
    :param n_TI: number of TI chain atoms
    :return: Neel-Haar combined state
    """
    NeelHaarstate = np.kron(Neelstate(n_PXP), Haarstate(n_TI))
    return NeelHaarstate


# ===========================   Declarations of the separate Hamiltonians, coupling, and full coupled hamiltonian (basis of 2**ntot) ==========================================


def PXPOBCNew2(n): # faster method
    '''
    Faster way of PXP OBC
    :param n: number of atoms
    :return: PXP model hamiltonian OBC
    '''
    d = 2 ** n # full dimension
    Pi_minus1 = P_i  # notation convenience
    Xi = X_i  # notation convenience
    Pi_plus1 = P_i  # notation convenience
    PXPleftbound = np.zeros((d, d))
    PXPrightbound = np.zeros((d, d))
    PXPnobound = np.zeros((d, d))
    for i in range(1, n+1):  # i marks the number of atom from 1 to n
        if i == 1:
            PXPleftbound = np.kron(Xi,np.kron(Pi_plus1,np.identity(2**(n-2))))
        elif i == n:
            PXPrightbound = np.kron(np.identity(2**(n-2)),np.kron(Pi_minus1,Xi))
        else:
            PXPnobound = PXPnobound + np.kron(np.identity(2**(i-2)),np.kron(Pi_minus1,np.kron(Xi,np.kron(Pi_plus1,np.identity(2**(n-1-i))))))
    pxp_fin = PXPleftbound + PXPnobound + PXPrightbound
    return pxp_fin

def TIOBCNew2(n_TI, J, h_x, h_z): # faster method
    """
    Tilted Ising Hamiltonian OBC, faster method
    :param n_TI: No of Tilted ising atoms MUST BE =>2
    :param J: Ising coupling parameter
    :param h_x: transverse field strength
    :param h_z: Z field (Longtitudinal) strength
    :return: Tilted Ising Hamiltonian (for i=>2)
    """
    d = 2 ** n_TI # dimesion
    Zi = Z_i # Notation convenience
    Zi_plus1 = Z_i # Notation convenience
    Xi = X_i  # Notation convenience
    Ising_part = np.zeros((d, d))
    Transverse = np.zeros((d, d))
    Longtitude = np.zeros((d, d))
    for i in range(1, n_TI+1): # i = atom number
        Transverse= Transverse + np.kron(np.identity(2 ** (i-1)),np.kron(Xi,np.identity(2 ** (n_TI-i))))
        Longtitude= Longtitude + np.kron(np.identity(2 ** (i-1)),np.kron(Zi,np.identity(2 ** (n_TI-i))))
        if i != n_TI:
            Ising_part= Ising_part + np.kron(np.identity(2 ** (i-1)),np.kron(Zi,np.kron(Zi_plus1, np.identity(2 ** (n_TI-1-i)))))
    TI_fin = ((h_x) * Transverse) + ((h_z) * Longtitude) + ((J) * Ising_part)
    return TI_fin

def TIOBCNewImpure2(n, J, h_x, h_z, h_imp, m): # faster method
    """
    Tilted Ising Hamiltonian OBC with impurity on the m'th site
    :param n: No of Tilted ising atoms MUST BE =>2
    :param J: Ising coupling parameter
    :param h_x: transverse field strength
    :param h_z: Z field (Longtitudinal) strength
    :param h_imp: impurity strength
    :param m: impurity site! (from 1 to n_TI)
    :return: Tilted Ising Hamiltonian WITH IMPURITY (for i=>2)
    """
    d = 2 ** n #dimension
    TI = TIOBCNew2(n, J, h_x, h_z)
    Z_impure= (h_imp) * Z_generali(n,m)
    TI_impure = np.add(TI, Z_impure)
    return TI_impure


def Coupling2(n_PXP, n_TI , Coupmat, h_c): #faster way
    """
    faster way of
    2 site coupling matrix in TOTAL Hamiltonian dimension (2**(n_PXP + n_TI))
    :param n_PXP: Number of PXP atoms (0 to whatever)
    :param n_TI: Number of TI atoms (0 to whatever)
    :param Coupmat: Coupling 2x2 base matrix ( one site matrix)
    :param h_c: coupling strength parameter
    :return:
    """
    n_tot = np.add(n_PXP, n_TI)  # Total number of atoms
    d_pxp = 2 ** n_PXP #dimension of PXP
    d_TI = 2 ** n_TI #dimension of TI
    d_TOT = 2 ** n_tot  #Total dimension
    Coupling = (h_c) * np.kron(Coupmat, Coupmat)
    if n_TI == 0 or n_PXP == 0 or h_c == 0:
        Coupterm = np.zeros((d_TOT,d_TOT))
    else:
        Coupterm = np.kron(np.kron(np.identity(2 ** (n_PXP - 1)), Coupling), np.identity(2 ** (n_TI-1)))
    return Coupterm


def PXPBathHam2(n_PXP, n_TI, Coupmat, J, h_x, h_z, h_c, h_imp, m=1):
    """
    FULL 2**(n_tot) dimension COUPLED PXP and TI hamiltonian Builder
    :param n_PXP: PXP Atom number
    :param n_TI: TI Atom number
    :param Coupmat: 2x2 base matrix of coupling
    :param h_x: transverse field strength
    :param h_z: Z field strength
    :param h_c: coupling strength
    :param h_imp: impurity strength
    :param m: impurity site (default is 1)
    :return: Full coupled Hamiltonian
    """
    n_tot= np.add(n_PXP,n_TI)
    d_PXP = 2 ** n_PXP
    d_TI = 2 ** n_TI
    d_tot= 2 ** n_tot
    PXP = PXPOBCNew2(n_PXP)
    TI = TIOBCNewImpure2(n_TI, J, h_x, h_z, h_imp, m)
    HamNoCoupl = np.add(np.kron(PXP, np.identity(d_TI)), np.kron(np.identity(d_PXP), TI))
    TotalHam = np.add(HamNoCoupl, Coupling2(n_PXP, n_TI, Coupmat, h_c))
    return TotalHam

def PXPBathHamNoCoupl2(n_PXP, n_TI, J, h_x, h_z, h_imp, m=1):
    """
    FULL 2**(n_tot) dimension COUPLED PXP and TI hamiltonian Builder
    :param n_PXP: PXP Atom number
    :param n_TI: TI Atom number
    :param h_x: transverse field strength
    :param h_z: Z field strength
    :param h_imp: impurity strength
    :param m: impurity site (default is 1)
    :return: Full coupled Hamiltonian
    """
    n_tot= np.add(n_PXP,n_TI)
    d_PXP = 2 ** n_PXP
    d_TI = 2 ** n_TI
    d_tot= 2 ** n_tot
    PXP = PXPOBCNew2(n_PXP)
    TI = TIOBCNewImpure2(n_TI, J, h_x, h_z, h_imp, m)
    HamNoCoupl = np.add(np.kron(PXP, np.identity(d_TI)), np.kron(np.identity(d_PXP), TI))
    return HamNoCoupl


# ============================== Declarations of metrics- functions to check quantities  ==========================================


def EvecEval(Mat):
    '''
    calculates eigenvalues and eigenstates of a HERMITIAN matrix
    :param Mat: Any Hermitian matrix
    :return: eigenvalues and eigenstates
    '''
    eval, evec = la.eigh(Mat)
    return np.real(np.round(eval, 5)), np.round(evec, 5)

def EigenSpan(Mat,
              VecState):
    """
    Spans some vector (VecState) in the eigenbasis of a matrix.
    :param Mat: Input matrix for eigenstate basis decomposition
    :param VecState: Some vector state (initial state usually) that we want to span in eigenstate basis
    :return: vector of weights (in Eigenstate basis)
    """
    Eval, Evec = EvecEval(Mat)
    W = np.dot(np.transpose(Evec),VecState)
    return W

def EigenCombine(Mat,VecState):
    '''
    checks tat multiplying the weights back with the eigenstates gives the original vector
    :param Mat: Any matrix that we will use the eigenbasis of
    :param VecState: vector we want to span in the eigenbasis of the matrix
    :return: recombined vector
    '''
    W = EigenSpan(Mat,VecState)
    Eval, Evec = EvecEval(Mat) #Evec is matrix!
    Recombine= np.round(np.dot(Evec,W),4)
    return Recombine

def NewTimeProp2(Ham, n_PXP, n_TI,  Initialstate,
             T_max, T_step, Color, Marker):
    '''
    NEW METHOD 10.5.22
    :param Ham: Hamiltonian for propagation
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_max: Max Time of propagation
    :param T_step: time step (interval)
    :param Color:
    :param Marker:
    :return: plot of |<Z_2|Z_2(t)>|^2 as a func of time t
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    v= Initialstate
    t = np.arange(1, T_max, T_step)
    for t in np.nditer(t):
        v = np.dot(U,v)
        plt.plot(t, np.round(np.vdot(Initialstate,v)**2,4), marker=Marker, markersize=3, color=Color)

def RunNewTimeProp2(n_PXP, n_TI, Coupl=Z_i, J=1 ,h_x=1, h_z=1, h_c=1, T_max=20, T_step=0.05, h_imp=0.01, m=1):# 1 Time propagation of PXP TI COUPLED
    """
    Runs NewTimeProp2
    """
    H_full = PXPBathHam2(n_PXP, n_TI, Coupl, J, h_x, h_z, h_c, h_imp, m)
    InitVecstate = NeelHaar(n_PXP,n_TI)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.random.choice(np.array(('s', '^', 'o', 'X')))
    NewTimeProp2(H_full, n_PXP, n_TI, InitVecstate, T_max, T_step, Color, markers)

def Run4TimePropPxpConserve2(n_totArray, n_PXP, Coupl=Z_i, J=1 , h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_c=1, h_imp=0.01, m=1, T_max=20, T_step=0.05):
    """
    time propagation of 4 different TOTAL atom sizes, PXP number conserved
    :param n_totArray: Array of 4 different TOTAL atom numbers
    :param n_PXP: number of PXP atoms
    :param Coupl: Nature of coupling (2x2 matrix)
    :param h_x: transverse field strength
    :param h_z: Z field strength
    :param h_c: coupling strength
    :param T_max: max time
    :param T_step: step size (between steps)
    :return: Timeprop graph x4
    """
    markers = np.array(('s', '^', 'o', 'd'))
    colors = np.array(('b','r','y','k'))
    n_TIarray = n_totArray - n_PXP # Vector!
    n_start = n_TIarray[0]
    for n_TI in n_TIarray: #running over n_tots
        H = PXPBathHam2(n_PXP,n_TI, Coupl, J, h_x, h_z, h_c, h_imp, m)
        Initialstate = NeelHaar(n_PXP, n_TI)
        Color = colors[n_TI-n_start]
        marker = markers[n_TI-n_start]
        NewTimeProp2(H, n_PXP, n_TI, Initialstate, T_max, T_step, Color, marker)
    custom_lines = [Line2D([0], [0], color=colors[0], marker=markers[0]),
                    Line2D([0], [0], color=colors[1], marker=markers[1]),
                    Line2D([0], [0], color=colors[2], marker=markers[2]),
                    Line2D([0], [0], color=colors[3], marker=markers[3])] #LEGEND DEFINITIONS
    plt.legend(custom_lines, ['{} total atoms'.format(n_totArray[0]), '{} total atoms'.format(n_totArray[1]), '{} total atoms'.format(n_totArray[2]),'{} total atoms'.format(n_totArray[3])])
    plt.xlabel('$t$')
    plt.ylabel(r'$|\langle\mathbb{Z}_{2}|\mathbb{Z}_{2}(t)\rangle|^{2}$')
    #plt.savefig('new_fidelity_for_different_numbers_of_total_atoms.pdf')
    plt.title('Quantum Fidelity of {}-PXP Neel State with coupling strength {}'.format(n_PXP,h_c))


def ZiSandwichCheck(Ham, n_PXP, n_TI,  Initialstate,
             T_max, T_step, i, Color, Marker):
    '''
    NEW METHOD 10.5.22
    :param Ham: Hamiltonian for propagation
    :param n_PXP: Size of PXP chain (atoms)
    :param n_TI: Size of TI chain (atoms)
    :param Initialstate:  Initial Vector state we would like to propagate
    :param T_max: Max Time of propagation
    :param T_step: time step (interval)
    :param i: Z_i site choice
    :param Color:
    :param Marker:
    :return: plot of |<Z_2|Z_2(t)>|^2 as a func of time t
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    v_ket= Initialstate
    v_bra= Initialstate
    t = np.arange(1, T_max, T_step)
    for t in np.nditer(t):
        v_ket = np.dot(U,v_ket)
        v_bra = np.dot(U,v_bra)
        plt.plot(t, np.round(np.vdot(v_bra,np.dot(Z_generali(n_PXP+n_TI,i),v_ket)),4), marker=Marker, markersize=3, color=Color)
    return

def RunZiSandwichCheck(n_PXP, n_TI, i, Coupl=Z_i, J=1 ,h_x=1, h_z=1, h_c=0, T_max=20, T_step=0.05, h_imp=0.01, m=1 ):# 1 Time propagation of PXP TI COUPLED
    """
    Runs Z1SandwichCheck
    """
    H_full = PXPBathHam2(n_PXP, n_TI, Coupl, J, h_x, h_z, h_c, h_imp, m)
    InitVecstate = NeelHaar(n_PXP,n_TI)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.random.choice(np.array(('s', '^', 'o', 'X')))
    ZiSandwichCheck(H_full, n_PXP, n_TI, InitVecstate, T_max, T_step, i, Color, markers)

# TODO think about the problem with the values for different Impurity strength and different atom chain sizes?

def RNewMeanMetricTI(eval):
    """
    mean R metric calculation
    :param eval: EigenValues (size-ordered: smallest to biggest)
    :return: r mean value - r= 0.39 poisson, r=0.536 W-D
    """
    S = np.diff(eval)  # returns  array of n-1 (Non-Negative) differences
    r = np.zeros([S.shape[0] - 1])
    for i in range(1, S.shape[0]):
        r[i - 1] = np.divide(min(S[i], S[i - 1]), max(S[i], S[i - 1]))
    R = np.around(r, 5)
    N = np.count_nonzero(R)
    Rmean = np.divide(np.sum(R), N)  # Averaging over No. of non-zero R contributions
    return Rmean

def RunRNewMeanMetricTI(n_TI, h_x, h_z, h_imp, m = 1):
    '''
    Runs the RMeanMetric function on Tilted Ising model
    :param n_TI:
    :param h_x:
    :param h_z:
    :param h_imp:
    :param m:
    :return: Runs the RMeanMetric function on Tilted Ising model
    '''
    eval, evec = EvecEval(TIOBCNewImpure2(n_TI, 1, h_x, h_z, h_imp, m))
    return RNewMeanMetricTI(eval)

# RunRNewMeanMetricTI(9, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0.1)

def RNewMeanMetricTICUT(eval):
    """
    Cut version of 1/8 of eigenvalues on every side of the spectrum
    :param eval: EigenValues (size-ordered: smallest to biggest)
    :return: r mean value - r= 0.39 poisson, r=0.536 W-D
    """
    S = np.diff(eval)  # returns  array of n-1 (Non-Negative) differences
    r = np.zeros([S.shape[0] - 1])
    for i in range(int(np.divide(S.shape[0],8)), int(7*np.divide(S.shape[0],8))):
        r[i - 1] = np.divide(min(S[i], S[i - 1]), max(S[i], S[i - 1]))
    R = np.around(r, 5)
    N = np.count_nonzero(R)
    Rmean = np.divide(np.sum(R), N)  # Averaging over No. of non-zero R contributions
    return Rmean

def RunRNewMeanMetricTICUT(n_TI, h_x, h_z, h_imp, m = 1): #Run the RMeanMetric function on Tilted Ising model
    '''
    Runs the cut version of RMeanMetric function on Tilted Ising model
    :param n_TI:
    :param h_x:
    :param h_z:
    :param h_imp:
    :param m:
    :return: Runs the cut version of RMeanMetric function on Tilted Ising model
    '''
    eval, evec =EvecEval(TIOBCNewImpure2(n_TI, 1, h_x, h_z, h_imp, m))
    return RNewMeanMetricTICUT(eval)

#RunRNewMeanMetricTICUT(9, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_imp=0.1)


# TODO fix the different scripts that run functions from here?


#-------------------------- NEW REGIME----------------------------

def Findpeaksosc():
    find_peaks()