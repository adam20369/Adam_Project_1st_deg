import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate

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


# ======================== declarations of operators in D=2**n Hilbert (Kronecker) space ===========================

def Z_1_old(n):
    Z_1 = Z_i  # pauli Z matrix
    for i in range(0, n - 1):
        Z_1 = np.kron(Z_1, np.identity(2)).astype('int32')
    return Z_1

def Z_1(n):
    Z_1 = np.kron(Z_i, np.identity(2**(n-1)))
    return Z_1

# print(Z_1(3))

def O_z_old(n):
    d = 2 ** n
    O_zsum = np.zeros((d, d))
    for i in range(0, n):
        Z_i = np.array([[1, 0], [0, -1]])  # pauli Z matrix
        for j in range(0, n - 1):
            if j < i:
                Z_i = np.kron(np.identity(2), Z_i)
            else:
                Z_i = np.kron(Z_i, np.identity(2))
        O_zsum = np.add(O_zsum, Z_i)  # summation over n pauli Z matrices of dimension d
    O_zop = (1 / n) * O_zsum
    return O_zop

def O_z(n):
    d = 2 ** n
    O_zsum = np.zeros((d,d))
    for i in range(0,n):
        z_i= np.kron(np.kron(np.identity(2**i),Z_i), np.identity(2**(n-(i+1))))
        O_zsum= np.add(O_zsum, z_i)
    O_zop= (1/n) * O_zsum
    return O_zop



# print(O_z(3))

# def T_op(n): #translation operator for n sites
#     d= 2**n
#     r=np.array((d/2), dtype=int)
#     T= np.zeros((d,d), dtype=int)
#     for m in range(0, d):
#         for i in range(0, d):
#             if i == 2*m:
#                 T[m,i]= 1
#             elif i== 2*m+1:
#                 T[m+r,i]= 1
#     return T
# print (T_op(4))

def SubspaceP(n):  # Projector on the relevant constrained OBC hilbert subspace
    d = 2 ** n
    ProjP = np.identity(d)
    for i in range(0, n - 1):
        qi = Q_i
        qiplus1 = Q_i
        for t in range(0, n - 1):  # for kronecker product of qi
            if t < i:
                qi = np.kron(np.identity(2), qi)
            else:
                qi = np.kron(qi, np.identity(2))
        for m in range(0, n - 1):  # for kronecker product of qiplus1
            if m < i + 1:
                qiplus1 = np.kron(np.identity(2), qiplus1)
            else:
                qiplus1 = np.kron(qiplus1, np.identity(2))
        ProjP = np.matmul(ProjP, (np.subtract(np.identity(d), np.matmul(qi, qiplus1))))
    return ProjP.astype('int')


# print("\n Constrained subspace:\n", SubspaceP(4))

def SubspcNeelstate(n):  # outputs the neel state (Z_2) in standard basis in subspace dim!! ONLY FOR EVEN n!!!!
    d = 2 ** n
    k = np.array(n / 2).astype('int32')
    Neelproj = np.kron(Q_i, P_i).astype('int32')
    for i in range(0, k - 1):
        Neelproj = np.kron(np.kron(Neelproj, Q_i), P_i)
    SubspcNproj = SubspaceMat(n, Neelproj)
    Nstate = np.squeeze(SubspcNproj[np.nonzero(SubspcNproj), :])[0, :]
    return Nstate


def Neelstate(n): # GENERAL NEELSTATE
    d = 2 ** n
    k= np.array(n)
    Neel= np.array(1)
    Even= np.arange(0,n,2)
    for i in range(0,n):
        if i in Even:
            Neel = np.kron(Neel, r)
        else:
            Neel = np.kron(Neel, g)
    return Neel

def Haarstate(n): # GENERAL HAAR STATE
    if n == 0:
         HaarVec= np.array(1)
    else:
        d = 2 ** n
        alpha= np.random.normal(0, 1, 2**n)
        betta= np.random.normal(0, 1, 2 ** n)
        v= alpha + 1j*betta
        HaarVec= np.divide(v, la.norm(v))
    return HaarVec

def NeelHaar(n_tot, n):
    NeelHaarstate = np.kron(Neelstate(n), Haarstate(n_tot - n))
    return NeelHaarstate

def PXPHamOBC(n):
    d = 2 ** n
    pxp_fin = np.zeros((d, d))
    for i in range(1, n + 1):  # goes over all atoms for total sum in the end
        piminus1 = P_i
        xi = X_i
        piplus1 = P_i
        for m in range(1, n):  # for kronecker product of P_(i-1)
            if i == 1:
                piminus1 = np.identity(d)
            elif m < i - 1:
                piminus1 = np.kron(np.identity(2), piminus1)
            else:
                piminus1 = np.kron(piminus1, np.identity(2))
        for t in range(1, n):  # for kronecker product of X_(i)
            if t < i:
                xi = np.kron(np.identity(2), xi)
            else:
                xi = np.kron(xi, np.identity(2))
        for c in range(1, n):  # for kronecker product of P_(i+1)
            if i == n:
                piplus1 = np.identity(d)
            elif c < i + 1:
                piplus1 = np.kron(np.identity(2), piplus1)
            else:
                piplus1 = np.kron(piplus1, np.identity(2))
        pxp_ar = np.matmul(np.matmul(piminus1, xi), piplus1)  # calculates PXP form for a given site
        pxp_fin = np.add(pxp_fin, pxp_ar)  # cumulative sum of PXP's
    return pxp_fin.astype('int32')


# print("\n PXP= \n", PXPHamOBC(4))
def PXPOBCNew(n):  # OBC
    d = 2 ** n
    pxp_fin = np.zeros((d, d))
    for i in range(0, n):  # goes over all atoms (i [from 0 to n-1]= the atom) for total sum in the end
        piminus1 = P_i  # initial declaration
        xi = X_i  # initial declaration
        piplus1 = P_i  # initial declaration
        if i == 0:
            piminus1 = np.identity(d)  # boundary (X_1*P_2)
        else:
            piminus1 = np.kron(np.identity(2 ** (i - 1)),
                               np.kron(piminus1, np.identity(2 ** (n - i))))  # general P_i-1 term
        xi = np.kron(np.identity(2 ** (i)), np.kron(xi, np.identity(2 ** (n - (i + 1)))))  # general X_i term
        if i == n - 1:
            piplus1 = np.identity(d)  # boundary (P_N-1*X_N)
        else:
            piplus1 = np.kron(np.identity(2 ** (i + 1)),
                              np.kron(piplus1, np.identity(2 ** (n - (i + 2)))))  # general P_i+1 term
        pxp_ar = np.matmul(piminus1, np.matmul(xi, piplus1))  # calculates hamiltonian PER i
        pxp_fin = np.add(pxp_fin, pxp_ar)  # cumulative sum over i
    return pxp_fin


def TiltedIsingHam(n, h_x, h_z):  # !!!!OLD!!! Tilted Ising Hamiltonian OBC n= no of atoms (must be =>2)
    d = 2 ** n
    pxp_fin = np.zeros((d, d))
    for i in range(1, n + 1):  # goes over all atoms for total sum in the end
        zi = Z_i
        ziplus1 = Z_i
        xi = X_i
        for m in range(1, n):  # For Z_i term (Z_1 up to Z_n)
            if m < i:
                zi = np.kron(np.identity(2), zi)
            else:
                zi = np.kron(zi, np.identity(2))
        for t in range(1, n):  # For Z_i+1 term PBC
            if i == n:
                ziplus1 = np.zeros((d, d))  # terminates Z_n*Z_n+1
            elif t < i + 1:
                ziplus1 = np.kron(np.identity(2), ziplus1)
            else:
                ziplus1 = np.kron(ziplus1, np.identity(2))
        for c in range(1, n):  # For X_i term
            if c < i:
                xi = np.kron(np.identity(2), xi)
            else:
                xi = np.kron(xi, np.identity(2))
        pxp_ar = np.add(np.add(np.matmul(zi, ziplus1), (h_z) * zi), (h_x) * xi)  # calculates hamiltonian PER i
        pxp_fin = np.add(pxp_fin, pxp_ar)  # cumulative sum over i
    return pxp_fin.astype('int32')


# Zi*Zi+1 term always diagonal so Zi*Zi+1=Zi+1*Zi

def TIOBCNew(n_TI, h_x, h_z): # Tilted Ising Hamiltonian OBC n= no of atoms (n must be =>2)
    """
    :param n_TI: No of Tilted ising atoms MUST BE =>2
    :param h_x: transverse field
    :param h_z: Z field
    :return:
    """
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
    return TI_fin

def TIOBCNewImpure(n_TI, h_x, h_z): #Tilted Ising with impurity at the Z_1 site!!
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
    TI_fin_new= np.add(TI_fin, 0.11*np.kron(Z_i,np.identity(int(np.divide(d,2)))))
    return TI_fin_new


def Coupling(n_tot, n, Coupmat, h_c):
    d_pxp = 2 ** n
    d_TI = 2 ** np.subtract(n_tot, n)
    d_TOT = 2 ** n_tot
    # d_tot = 2 ** n_tot
    Coupmat = (h_c) * Coupmat  # the coupling nature (2x2 matrix- usually pauli)
    if np.subtract(n_tot,n)==0 or np.subtract(n_tot,n)==n_tot:
        Coupterm = np.zeros((d_TOT,d_TOT))
    else:
        CoupMat_n = np.kron(np.kron(np.identity(2 ** (n - 1)), Coupmat), np.identity(d_TI))
        CoupMat_nplus1 = np.kron(np.kron(np.identity(d_pxp), Coupmat), np.identity(2 ** (n_tot - (n + 1))))
        Coupterm = np.matmul(CoupMat_n, CoupMat_nplus1)
    return Coupterm #returns hilbert space matrix of dimension 2**n_tot


# def PXPBathHamUncoup(n_tot, n, Coupmat, h_x, h_z,h_c):  # PXP+Bath UNCOUPLED!!!!
#     d_pxp = 2 ** n
#     d_TI = 2 ** np.subtract(n_tot, n)
#     # d_tot = 2 ** n_tot
#     PXP = PXPOBCNew(n)
#     TI = TIOBCNew(np.subtract(n_tot, n), h_x, h_z)
#     HamNoCoup = np.add(np.kron(PXP, np.identity(d_TI)), np.kron(np.identity(d_pxp), TI))
#     ####TotHam= np.add(HamNoCoup,Coupling(n_tot,n ,Coupmat))####
#     return HamNoCoup


def PXPBathHam(n_tot, n, Coupmat, h_x, h_z, h_c):
    d_pxp = 2 ** n
    d_TI = 2 ** np.subtract(n_tot, n)
    # d_tot = 2 ** n_tot
    PXP = PXPOBCNew(n)
    TI = TIOBCNew(np.subtract(n_tot, n), h_x, h_z)
    HamNoCoup = np.add(np.kron(PXP, np.identity(d_TI)), np.kron(np.identity(d_pxp), TI))
    TotHam = np.add(HamNoCoup, Coupling(n_tot, n, Coupmat, h_c))
    return TotHam


# comparison of with and without coupling: print(np.allclose(PXPBathHam(n_tot,n,np.zeros((2,2)),1,1),np.add(np.kron(PXPOBCNew(n),np.identity(2**(np.subtract(n_tot, n)))),np.kron(np.identity(2**n),TIOBCNew(np.subtract(n_tot, n), 1, 1)))))


# ========================== OLD PROJECT FUNCTION DECLERATIONS=================================================

def SubspaceMat(n,
                Matrix):  # Outputs matrix of choice in the relevant projected hilbert subspace for PXP model(defined by SubspaceP)
    SubspcRCs = np.nonzero(
        np.any(SubspaceP(n) != 0, axis=0))  # gives number of rows/ coloums (equal) that are included in subspace
    # print("\n states relevant to subspace: \n", SubspcRCs)
    SubspcMat = np.squeeze(Matrix[SubspcRCs, :])  # cutting irrelevant rows
    SubspcMat = np.squeeze(SubspcMat[:, SubspcRCs])  # cutting irrelevant coloumns
    # print("\n block Matrix in constrained subspace: \n", SubspcMat(n,PXPOBCNew(n))
    # print(SubspcMat.shape)
    return SubspcMat


def Subspccount(n, Matrix):  # Outputs The dimensions (Rows #/ Col #) of subspace when given the number of atoms
    SubspcRCs = np.nonzero(
        np.any(SubspaceP(n) != 0, axis=0))  # gives number of rows and coloums! that are included in subspace
    # print("\n states relevant to subspace: \n", SubspcRCs)
    SubspcMat = np.squeeze(Matrix[SubspcRCs, :])
    SubspcMat = np.squeeze(SubspcMat[:, SubspcRCs])
    return np.shape(SubspcMat)[0]


# print(Subspccount(8,PXPOBCNew(8)))

def EvecEval(Mat):  # calculates eigenvalues and eigenstates of HERMITIAN matrix
    eval, evec = la.eigh(Mat)
    return np.real(np.round(eval, 4)), np.round(evec, 4)
#TODO hermitian

# print("\n Hamiltonian Eigenvalues and Eigenvectors: \n", EvecEval(SubspaceMat(m, Matrix)))

def Fig2A(EigenEnVecs,
          Op):  # checking the expectation values with final H eigenvecs and standard basis operators (Subspace dim)
    Eval, Evec = EigenEnVecs
    y = np.matmul(np.transpose(np.conjugate(Evec)), np.matmul(Op, Evec))
    for j in range(0, np.size(Eval)):
        # print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y[j,j],3)))
        plt.plot(Eval[j], np.real(y[j, j]), marker='.', color='C2')
    plt.xlabel('$Energy$')
    plt.ylabel(r'$\langle\hat{Op}\rangle$')
    # plt.savefig('Expectation_Value_OZ.pdf')
    return plt.show()

# Fig2A(EvecEval(SubspaceMat(m, Hamiltonian)),SubspaceMat(m, O_z(m))) #TODO the O_z is good
# Fig2A(EvecEval(SubspaceMat(m, Hamiltonian)),SubspaceMat(m, Z_1(m)))



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


# Fig3B(EvecEval(SubspaceMat(m, Hamiltonian(m))), (SubspaceMat(m, Neelstate(m))))


def EigenSpan(EigenEnVecs,
              Nstate):  # outputs weights (inner product) of Neel state spanned in eigenstate basis (subspace dim)
    Eval, Evec = EigenEnVecs
    Z_2 = Nstate
    y = np.zeros(np.size(Eval)).astype('complex')
    for j in range(0, np.size(Eval)):
        y[j] = np.dot(Z_2, Evec[:, j])
    # print("\n array of <Z_2|EigenVec(j)>:", np.real(y))
    return y


# EigenSpan(EvecEval(SubspaceMat(m, Hamiltonian)), (SubspaceMat(m, Neelstate(m))))

def normconst(EigenSpan):  # sum of the inner products for normalization
    y = EigenSpan
    sum = np.dot(np.conjugate(y), y)
    # print("\n sum \n", np.round(np.real(sum),3))
    return np.round(np.real(sum), 5)

def normconst2(EigenSpan):
    return la.norm(EigenSpan)
# normconst(EigenSpan(EvecEval(SubspaceMat(m, Hamiltonian)), (SubspaceMat(m, Neelstate(m)))))

def TimePropPXP(EigenEnVecs, Subspcdim, Spans,
                T_max):  # SUBSPC Dim time propagation of each eigenstate with it's corresponding eigenenergy ########OLD#########
    Eval, Evec = EigenEnVecs
    w = np.dot(np.divide(1,(np.sqrt(normconst(Spans)))), (Spans))
    t = np.arange(0, T_max, 0.05)
    y = 0
    for t in np.nditer(t):
        Z_2 = np.zeros(Subspcdim)
        Z_2t = np.zeros(Subspcdim)
        for j in range(0, np.size(Eval)):  # alternative way- just multiply evecs as orthogonal ones (easier)
            Z_2 = Z_2 + np.dot(w[j], Evec[:, j])  # Z_2 spanned in eigenstate basis as Cols of a matrix TODO can take it out
            Z_2t = Z_2t + np.dot(np.dot((np.exp(-1j * Eval[j] * t)), w[j]),
                                 Evec[:, j])  # Z_2(t) spanned in eigenstate basis as Cols of a matrix
        y = (np.absolute(np.dot(np.conjugate((Z_2)), (Z_2t)))) ** 2
        plt.plot(t, np.round(y, 4), marker='.', color='C2')
    plt.xlabel('$t$')
    plt.ylabel(r'$|\langle\mathbb{Z}_{2}|\mathbb{Z}_{2}(t)\rangle|^{2}$')
    plt.savefig('fidelity_12atoms.png')
    plt.title('Quantum Fidelity of the Neel State vs. Time')
    plt.show()
#TimePropPXP(EvecEval(SubspaceMat(m, Hamiltonian)),Subspccount(m,Hamiltonian), EigenSpan(EvecEval(SubspaceMat(m, Hamiltonian)), SubspcNeelstate(m)),T_max)

#TODO FIX NORMCONST AND SOLVE NORMALIATION BULLSHIT

def TimeProp(EigenEnVecs, n_tot, Nstate,
             T_max, Color, marker):  # N_TotGENERAL DIM  time propagation of each eigenstate with it's corresponding eigenenergy
    Eval, Evec = EigenEnVecs
    w = np.dot(1 / (np.sqrt(normconst(EigenSpan(EigenEnVecs, Nstate)))), (EigenSpan(EigenEnVecs, Nstate)))
    t = np.arange(1, T_max, 0.05)
    y = 0
    for t in np.nditer(t):
        Z_2 = np.zeros(2 ** n_tot)
        Z_2t = np.zeros(2 ** n_tot)
        for j in range(0, np.size(Eval)):  # alternative way- just multiply evecs as orthogonal ones (easier)
            Z_2 = Z_2 + np.dot(w[j], Evec[:, j])  # Z_2 spanned in eigenstate basis as Cols of a matrix
            Z_2t = Z_2t + np.dot(np.dot((np.exp(-1j * Eval[j] * t)), w[j]),
                                 Evec[:, j])  # Z_2(t) spanned in eigenstate basis as Cols of a matrix
        y = (np.absolute(np.dot(np.conjugate((Z_2)), (Z_2t)))) ** 2
        plt.plot(t, np.round(y, 4), marker=marker, markersize=3, color=Color)
    #plt.show()
#TODO FIX NORMALIZATION????

def RunTimeProp(n_tot, n, Coupl=Z_i, h_x=1, h_z=1, h_c=1, T_max=20):# 1 Time propagation of PXP TI COUPLED
    """
    Runs TimeProp
    """
    H = PXPBathHam(n_tot, n, Coupl, h_x, h_z, h_c)
    # H= PXPBathHamUncoup(n_tot, n, Coupl, h_x, h_z) # Uncoupled version
    EV = EvecEval(H)
    Neel = Neelstate(n_tot)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    markers = np.random.choice(np.array(('s', '^', 'o', 'X')))
    TimeProp(EV, n_tot, Neel, T_max, Color, markers)
#### OLD RUN ######



def RMeanMetric(EV):  # r= 0.39 poisson, r=0.536 W-D
    S = np.diff(EV)  # returns an array of n-1 (NonNegative)
    # print(S)
    r = 0
    c = 0  #counts the r's that don't contribute
    for i in range(1, S.shape[0]):
        r = r + np.divide(min(S[i], S[i - 1]), max(S[i], S[i - 1])) #out=np.zeros((1)), where=max(S[i], S[i - 1]) != 0)
        #print(max(S[i], S[i - 1]))
        # c = c + np.array((0,1))[int(max(S[i], S[i - 1]) == 0)]  # counts the r's that don't contribute
        #print(c)
    r = r / (S.shape[0] - (c+1))  # n-1 minus c+1 more (n-2-c total)
    return r

def RunRmetric(n_TI, h_x, h_z, Hamiltonian): #running the metric for average r per one theta
    H = Hamiltonian(n_TI, h_x, h_z)
    EV = la.eigvalsh(H)
    # print(EV)
    return RMeanMetric(EV)


if __name__ == '__main__':
    print('adam')

#TODO checking what happens when I take coupling to 0 (should work) comparing with the other method of only PXP
#TODO-1- New hamiltonian method and moving all code to that