import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from scipy.linalg import expm
from matplotlib.lines import Line2D

#================-------------------------------------------Declarations (only to avoid errors)

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
#------------------------------------------------------------

def O_z(n): #mean of Z_i's
    d = 2 ** n
    Zi_sum = np.zeros((d,d))
    for j in range(0,n):
        Zi_sum= Zi_sum + np.kron(np.kron(np.identity(2**j),Z_i), np.identity(2**(n-(j+1))))
    O_zop= (1/n) * Zi_sum
    return O_zop

def PXPOBCNew(n):
    '''
    OBC PXP HAMILTONIAN
    :param n: number of atoms
    :return: PXP model hamiltonian OBC
    '''
    d = 2 ** n
    pxp_fin = np.zeros((d, d))
    for i in range(0, n):  # goes over all atoms (i [from 0 to n-1]= the atom) for total sum in the end
        piminus1 = P_i  # initial declaration
        xi = X_i  # initial declaration
        piplus1 = P_i  # initial declaration
        if i == 0:
            piminus1 = np.identity(d)  # boundary (X_1*P_2)
            piplus1 = np.kron(np.identity(2 ** (i + 1)),
                              np.kron(piplus1, np.identity(2 ** (n - (i + 2)))))  # general P_i+1 term
        elif i == n - 1:
            piplus1 = np.identity(d)  # boundary (P_N-1*X_N)
            piminus1 = np.kron(np.identity(2 ** (i - 1)),
                               np.kron(piminus1, np.identity(2 ** (n - i))))  # general P_i-1 term
        else:
            piminus1 = np.kron(np.identity(2 ** (i - 1)),
                               np.kron(piminus1, np.identity(2 ** (n - i))))  # general P_i-1 term
            piplus1 = np.kron(np.identity(2 ** (i + 1)),
                          np.kron(piplus1, np.identity(2 ** (n - (i + 2)))))  # general P_i+1 term
        xi = np.kron(np.identity(2 ** (i)), np.kron(xi, np.identity(2 ** (n - (i + 1)))))  # general X_i term
        pxp_ar = np.matmul(piminus1, np.matmul(xi, piplus1))  # calculates hamiltonian PER i
        pxp_fin = np.add(pxp_fin, pxp_ar)  # cumulative sum over i
    return pxp_fin

def TIOBCNew(n_TI, h_x, h_z):
    """
    Tilted Ising Hamiltonian OBC n= no of atoms (n must be =>2)
    :param n_TI: No of Tilted ising atoms MUST BE =>2
    :param h_x: transverse field strength
    :param h_z: Z field (parallel) strength
    :return: Tilted Ising Hamiltonian (for i=>2)
    """
    d = 2 ** n_TI #dimesion
    TI_fin = np.zeros((d, d))
    for i in range(0, n_TI):
        zi = Z_i  # inital declaration
        ziplus1 = Z_i  # inital declaration
        xi = X_i  # inital declaration
        zi = np.kron(np.identity(2 ** i), np.kron(zi, np.identity(2 ** (n_TI - (i + 1)))))  # Z_i term
        if i == n_TI - 1:
            ziplus1 = np.zeros((d, d))  # Z_i+1 boundary (kills boundary)
        else:
            ziplus1 = np.kron(np.identity(2 ** (i + 1)),
                              np.kron(ziplus1, np.identity(2 ** (n_TI - (i + 2)))))  # Z_i+1 term
        xi = np.kron(np.identity(2 ** (i)), np.kron(xi, np.identity(2 ** (n_TI - (i + 1)))))  # X_i term
        TI_ar = np.add(np.add(np.matmul(zi, ziplus1), (h_z) * zi), (h_x) * xi)  # calculates hamiltonian PER i
        TI_fin = np.add(TI_fin, TI_ar)
    return TI_fin

def TIOBCNewImpure(n_TI, h_x, h_z, h_i): #same Tilted Ising only with impurity at the Z_1 site!!
    d = 2 ** n_TI
    TI_fin = np.zeros((d, d))
    for i in range(0, n_TI):
        zi = Z_i  # inital declaration
        ziplus1 = Z_i  # inital declaration
        xi = X_i  # inital declaration
        zi = np.kron(np.identity(2 ** i), np.kron(zi, np.identity(2 ** (n_TI - (i + 1)))))  # Z_i term
        if i == n_TI - 1:
            ziplus1 = np.zeros((d, d))  # Z_i+1 boundary (kills boundary)
        else:
            ziplus1 = np.kron(np.identity(2 ** (i + 1)),
                              np.kron(ziplus1, np.identity(2 ** (n_TI - (i + 2)))))  # Z_i+1 term
        xi = np.kron(np.identity(2 ** (i)), np.kron(xi, np.identity(2 ** (n_TI - (i + 1)))))  # X_i term
        TI_ar = np.add(np.add(np.matmul(zi, ziplus1), (h_z) * zi), (h_x) * xi)  # calculates hamiltonian PER i
        TI_fin = np.add(TI_fin, TI_ar)
    TI_fin_impure = np.add(TI_fin, h_i*np.kron(Z_i,np.identity(int(np.divide(d,2)))))
    return TI_fin_impure

def Coupling(n_tot, n, Coupmat, h_c): # 2 site coupling matrix in TOTAL Hamiltonian size (2**n_totx2**n_tot )
    """
    :param n_tot: Total number of atoms in hamiltonian (PXP+ TI)
    :param n: Number of PXP atoms
    :param Coupmat: Coupling 2x2 base matrix ( one site matrix)
    :param h_c: coupling strength parameter
    :return:
    """
    d_pxp = 2 ** n #dimension of PXP
    d_TI = 2 ** np.subtract(n_tot, n) #dimension of TI
    n_TI = np.subtract(n_tot, n)
    d_TOT = 2 ** n_tot  #Total dimension
    Coupling = (h_c) * np.kron(Coupmat, Coupmat)
    if np.subtract(n_tot, n) == 0 or np.subtract(n_tot, n) == n_tot or h_c == 0:
        Coupterm = np.zeros((d_TOT,d_TOT))
    else:
        Coupterm = np.kron(np.kron(np.identity(2 ** (n - 1)), Coupling), np.identity(2 ** (n_TI-1)))
    return Coupterm


def Couplingalt(n_tot, n, Coupmat, h_c): #alternative way to write coupling (slower!!!, the older one)
    """
    :param n_tot: Total number of atoms in hamiltonian (PXP+ TI)
    :param n: Number of PXP atoms
    :param Coupmat: Coupling matrix (2x2- on one site first)
    :param h_c: strength parameter
    :return:
    """
    d_pxp = 2 ** n
    d_TI = 2 ** np.subtract(n_tot, n)
    d_TOT = 2 ** n_tot
    # d_tot = 2 ** n_tot
    Couplmat = (h_c) * Coupmat  # the coupling nature (2x2 matrix- usually pauli)
    if np.subtract(n_tot,n)==0 or np.subtract(n_tot,n)==n_tot or h_c==0:
        Coupterm = np.zeros((d_TOT,d_TOT))
    else:
        CoupMat_n = np.kron(np.kron(np.identity(2 ** (n - 1)), Couplmat), np.identity(d_TI))
        CoupMat_nplus1 = np.kron(np.kron(np.identity(d_pxp), Couplmat), np.identity(2 ** (n_tot - (n + 1))))
        Coupterm = np.matmul(CoupMat_n, CoupMat_nplus1)
    return Coupterm #returns hilbert space matrix of dimension 2**n_tot


def PXPBathHam(n_tot, n, Coupmat, h_x, h_z, h_c, h_i):
    """
    FULL 2**(n_tot) dimension COUPLED PXP and TI hamiltonian Builder
    :param n_tot: Total atom number
    :param n: PXP Atom number
    :param Coupmat: 2x2 matrix of coupling nature
    :param h_x: transverse field strength
    :param h_z: Z field strength
    :param h_c: coupling strength
    :return: Full Hamiltonian
    """
    d_pxp = 2 ** n
    d_TI = 2 ** np.subtract(n_tot, n)
    # d_tot = 2 ** n_tot
    PXP = PXPOBCNew(n)
    TI = TIOBCNewImpure(np.subtract(n_tot, n), h_x, h_z, h_i)
    HamNoCoup = np.add(np.kron(PXP, np.identity(d_TI)), np.kron(np.identity(d_pxp), TI))
    TotHam = np.add(HamNoCoup, Coupling(n_tot, n, Coupmat, h_c))
    return TotHam

#-------------------------------------------- METRICS FOR CHAOS---------------------------------------#

def EvecEval(Mat): ###    NEW FUNCTION! JUST HERE TO AVOID ERRORS!###

    '''
    calculates eigenvalues and eigenstates of a HERMITIAN matrix
    :param Mat: Any Hermitian matrix
    :return: eigenvalues and eigenstates
    '''
    eval, evec = la.eigh(Mat)
    return np.real(np.round(eval, 5)), np.round(evec, 5)

def EigenSpan(Mat, VecState): ###    NEW FUNCTION! JUST HERE TO AVOID ERRORS!###
    """
    Spans some vector (VecState) in the eigenbasis of a matrix.
    :param Mat: Input matrix for eigenstate basis decomposition
    :param VecState: Some vector state (initial state usually) that we want to span in eigenstate basis
    :return: vector of weights (in Eigenstate basis)
    """
    Eval, Evec = EvecEval(Mat)
    W = np.dot(np.transpose(Evec),VecState)
    return W


def Fig3B(EigenEnVecs, Nstate):  # ONLY EVEN NUM OF ATOMS
    Eval, Evec = EigenEnVecs
    for j in range(0, np.size(Eval)):
        # print("\n <Operator(", Eval[j], ")>:", np.log10((np.absolute(np.dot(Nstate,Evec[:,j])))**2))
        plt.plot(Eval[j], np.log10((np.absolute(np.dot(Nstate, Evec[:, j]))) ** 2), marker='.', color='C2')
    plt.xlabel('$E$')
    plt.ylabel(r'$log_{10}(|\langle\mathbb{Z}_{2}|\psi\rangle|)^{2}$')
    # plt.title('Overlap of Neel State with Eigenstates vs. Eigenstate Energy')
    plt.savefig('Overlap_12_atoms.pdf')
    return plt.show()
#TODO will need this later?

def EigenSpanAlt(Mat,
              VecState):  # alternative way to define the Eigenspan function (slower!!, the older one)
    """
    :param Mat: Input matrix for eigenstate decomposition
    :param VecState: Some vector state (initial state usually) that we want to span in eigenstate basis
    :return: vector of weights
    """
    Eval, Evec = EvecEval(Mat)
    Z_2 = VecState
    y = np.zeros(np.size(Eval)).astype('complex')
    for j in range(0, np.size(Eval)):
        y[j] = np.vdot(Z_2, Evec[:, j]) #with! Complex conjugation
    # print("\n array of <Z_2|EigenVec(j)>:", np.real(y))
    return y

def TimeProp(Ham, n_tot, VecState,
             T_max, T_step, Color, marker):  # time propagation of initial state by decomposing in eigenvectors basis & propagating w/ respect to corresponding eigenenergy
    """
    :param Ham: Hamiltonian for propagation
    :param n_tot: Size of dimension *(size of Hamiltonian)
    :param VecState: Initial Vector state we would like to propagate
    :param T_max: Time of propagation
    :param Color:
    :param marker:
    :return:
    """
    Eval, Evec = EvecEval(Ham)
    w = EigenSpan(Ham, VecState) #weights vector of projection of Vecstate onto Eigenbasis
    t = np.arange(1, T_max, T_step)
    for t in np.nditer(t): # builds Z_2t from scratch for every time t in the total time interval
        Z_2t = np.zeros(2 ** n_tot)
        for j in range(0, np.size(Eval)):
            Z_2t = Z_2t + np.dot(np.dot((np.exp(-1j * Eval[j] * t)), w[j]),
                                 np.transpose(Evec)[j, :])  # Z_2(t) spanned in eigenstate basis as Cols of a matrix
        Fidel = np.absolute(np.vdot(VecState,(Z_2t)))**2
        plt.plot(t, np.round(Fidel, 4), marker=marker, markersize=3, color=Color)
    # plt.show()

def RunTimeProp(n_tot, n, Vecstate, Coupl=Z_i, h_x=1, h_z=1, h_c=1, T_max=20, T_int=0.05):# 1 Time propagation of PXP TI COUPLED
    """
    Runs TimeProp
    """
    H_pxp= PXPOBCNew(n_tot)
    #H_tot = PXPBathHam(n_tot, n, Coupl, h_x, h_z, h_c, h_i=0.4)
    # H= PXPBathHamUncoup(n_tot, n, Coupl, h_x, h_z) # Uncoupled version
    InitVecstate = Vecstate(n_tot)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.random.choice(np.array(('s', '^', 'o', 'X')))
    #TimeProp(H_pxp, n_tot, InitVecstate, T_max, T_int, Color, markers)
    TimeProp(H_pxp, n_tot, InitVecstate, T_max, T_int, Color, markers)

def Sandwichcheck(Op, VecState, n_tot, n_pxp, T_max, Coupmat=Z_i, h_x=1, h_z=1, h_c=0, h_i=0):
    Eval, Evec = EvecEval(PXPBathHam(n_tot, n_pxp, Coupmat, h_x, h_z, h_c, h_i))
    w = EigenSpan(PXPBathHam(n_tot, n_pxp, Coupmat, h_x, h_z, h_c, h_i), VecState(n_tot,n_pxp)) #weights vector of projection of Vecstate onto Eigenbasis
    t = T_max
    Z_2t = np.zeros(2 ** n_tot)
    for j in range(0, np.size(Eval)):
        Z_2t = Z_2t + np.dot(np.dot((np.exp(-1j * Eval[j] * t)), w[j]),
                             np.transpose(Evec)[j, :])  # Z_2(t) calculated from spanning in eigenstate basis and propagating in time
    Sandwich= np.dot(np.dot(np.conjugate(Z_2t),Op(n_tot, 2)),Z_2t)
    return Sandwich

def Sandwichcheckpxp(Op, VecState, n_pxp, T_max):
    Eval, Evec = EvecEval(PXPOBCNew(n_pxp))
    w = EigenSpan(PXPOBCNew(n_pxp), VecState(n_pxp)) #weights vector of projection of Vecstate onto Eigenbasis
    t = T_max
    Z_2t = np.zeros(2 ** n_pxp)
    for j in range(0, np.size(Eval)):
        Z_2t = Z_2t + np.dot(np.dot((np.exp(-1j * Eval[j] * t)), w[j]),
                             np.transpose(Evec)[j, :])  # Z_2(t) calculated from spanning in eigenstate basis and propagating in time
    Sandwich= np.dot(np.dot(np.conjugate(Z_2t),Op(n_pxp, 2)),Z_2t)
    return Sandwich

def ZiSandwichCheck(Ham, n_PXP, n_TI,  Initialstate,
             T_start, T_max, T_step, i, Color, Marker):
    '''
    Newer version of code (10.5.22)
    plotting <Neel|Z_i(t)|Neel> with respect to time
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
    :return: plot of V=<Neel|Z_i(t)|Neel> with respect to time
    '''
    U= expm(-1j*Ham*T_step)
    U_dag= expm(1j*Ham*T_step)
    v_ket= Initialstate
    v_bra= Initialstate
    t = np.arange(T_start, T_max, T_step)
    for t in np.nditer(t):
        v_ket = np.dot(U,v_ket)
        v_bra = np.dot(U,v_bra)
        plt.plot(t, np.round(np.vdot(v_bra,np.dot(Z_generali(n_PXP+n_TI,i),v_ket)),4), marker=Marker, markersize=3, color=Color)
    return

def RMeanMetric(EV):  #Mean R metric, r= 0.39 poisson, r=0.536 W-D
    """
    :param EV: EigenValues (size-ordered: smallest to biggest)
    :return: r mean value
    """
    S = np.diff(EV)  # returns  array of n-1 (Non-Negative) differences between Eigenvalues
    r = 0
    #c = 0  #counts the r's that don't contribute
    for i in range(1, S.shape[0]):
        r = r + np.divide(min(S[i], S[i - 1]), max(S[i], S[i - 1])) #???out=np.zeros((1)), where=max(S[i], S[i - 1]) != 0)
        #print(max(S[i], S[i - 1]))
        # c = c + np.array((0,1))[int(max(S[i], S[i - 1]) == 0)]  # counts the r's that don't contribute
    r = r / (S.shape[0] - 1)  # n-1 minus c+1 more (n-2-c total)
    return r

def RunRmetric(n_TI, h_x, h_z, h_i, Ham): #Run the RMeanMetric function on Tilted Ising model
    H = Ham(n_TI, h_x, h_z, h_i)
    EV = la.eigvalsh(H) #outputs vector of eigenvalues, from smallest to biggest
    # print(EV)
    return RMeanMetric(EV)


########EE_part###################
def Densitymat(n_PXP, j, st): #todo CHECK
    '''
    calculates all rho's of eigenstates
    :param n_PXP: number of PXP atoms
    :param j: site of impurity
    :param st: strength of impurity
    :return: 3D array where every rho[:,:,i] is the density matrix of i'th eigenstate
    '''
    eval, evec = Diagonalize(Subspace_reduced_PXP(n_PXP,j,st))
    rho_tot=np.zeros([Block_dim(n_PXP,j,st),Block_dim(n_PXP,j,st),Block_dim(n_PXP,j,st)])
    for i in range(Block_dim(n_PXP,j,st)):
        rho_tot[:,:,i]=np.outer(evec[:,i],evec[:,i])
    return rho_tot

def Partition(n_PXP, j, st):
    '''
    Subspace dimensions for bipartition  dim(A)*dim(B)=Tot_dim
    :param n_PXP: number of PXP atoms
    :param j: site of impurity
    :param st: strength of impurity
    :return: 2 scalars of dimensions  of reduced subspaces
    '''
    dim = Block_dim(n_PXP, j, st)
    partition = (int(dim / 2)) #half of log2 (int) of Hilbert dimension
    complement = dim - partition #complement of the partition (due to it being int)
    SubspcA= 2**partition
    SubspcB= 2**complement
    return SubspcA, SubspcB


def Partialtrace(n_PXP, j, st):
    '''
    taking partial trace and getting reduced density matrix by partition defined before
    :param n_PXP: number of PXP atoms
    :param j: site of impurity
    :param st: strength of impurity
    :return: 3D array of subspace A partition reduced density matrices (ordered by eigenstates)
    '''
    rho_tot= Densitymat(n_PXP, j, st)
    tot_dim= Block_dim(n_PXP,j,st)                 #DO I NEED THIS?
    dim_A, dim_B= Partition(n_PXP, j, st) #reduced partition dimensions (base of 2)
    identity_dim_A= np.identity(dim_A)
    basis_vec_B = np.zeros([dim_B])
    Reduced_rho_A = np.zeros([dim_B,dim_B,tot_dim])
    for j in range(tot_dim):
        for i in range(dim_B):
            Traceterm=np.kron(identity_dim_A,basis_vec_B[i])
            Reduced_rho_A[:,:,j]= np.dot(np.dot(np.conjugate(Traceterm),rho_tot[:,:,j]),Traceterm)
    return Reduced_rho_A

def EntanglementEntropy(n_PXP, j, st):
    '''
    Calculating Entanglement entropy per eigenstate
    :param n_PXP: number of PXP atoms
    :param j: site of impurity
    :param st: strength of impurity
    :return:
    '''
    reduced_rho_A=Partialtrace(n_PXP, j, st)
    dim_A, dim_B= Partition(n_PXP, j, st) #reduced partition dimensions (base of 2)
    basis_vec = np.zeros([dim_B])


