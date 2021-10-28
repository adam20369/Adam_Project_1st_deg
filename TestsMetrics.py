import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from PXP_TI_Bath import *

def plotFig2A(m,n,p,Hamiltonian,Op): #plotting for 3 atom sizes
    Eval1, Evec1 = EvecEval(SubspaceMat(m,Hamiltonian(m)))
    Eval2, Evec2 = EvecEval(SubspaceMat(n,Hamiltonian(n)))
    Eval3, Evec3 = EvecEval(SubspaceMat(p,Hamiltonian(p)))
    x = np.matmul(np.transpose(np.conjugate(Evec1)), np.matmul(SubspaceMat(m,Op(m)), Evec1))
    y = np.matmul(np.transpose(np.conjugate(Evec2)), np.matmul(SubspaceMat(n,Op(n)), Evec2))
    z = np.matmul(np.transpose(np.conjugate(Evec3)), np.matmul(SubspaceMat(p,Op(p)), Evec3))
    fig, (ax1, ax2, ax3)=  plt.subplots(3)
    for j in range(0, np.size(Eval1)):
        # print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y[j,j],3)))
        ax1.plot(Eval1[j], np.real(x[j, j]), marker='.', color='C2')
        ax1.set_title('8 Atoms')
        ax1.set(xlabel='E', ylabel=r'$\langle n|\hat{O}^{Z}|n\rangle$')
    for i in range(0, np.size(Eval2)):
        # print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y[j,j],3)))
        ax2.plot(Eval2[i], np.real(y[i, i]), marker='.', color='C3')
        ax2.set_title('10 Atoms')
        ax2.set(xlabel='E', ylabel=r'$\langle n|\hat{O}^{Z}|n\rangle$')
    for k in range(0, np.size(Eval3)):
        ax3.plot(Eval3[k], np.real(z[k, k]), marker='.', color='C4')
        ax3.set_title('12 Atoms')
        ax3.set(xlabel='E', ylabel=r'$\langle n|\hat{O}^{Z}|n\rangle$')
    plt.savefig('fig1new')
    plt.show()


# plotFig2A(8,10,12,PXPOBCNew,O_z) #TODO the O_z is good

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


# normconst(EigenSpan(EvecEval(SubspaceMat(m, Hamiltonian)), (SubspaceMat(m, Neelstate(m)))))

def TimePropPXP(EigenEnVecs, Subspcdim, Spans,
                T_max):  # SUBSPC Dim time propagation of each eigenstate with it's corresponding eigenenergy
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
#TODO FIX NORMALIZATION!!!
#TimePropPXP(EvecEval(SubspaceMat(m, Hamiltonian)),Subspccount(m,Hamiltonian), EigenSpan(EvecEval(SubspaceMat(m, Hamiltonian)), SubspcNeelstate(m)),T_max)

def TimeProp(EigenEnVecs, n_tot, Nstate,
             T_max, Color):  # N_TotGENERAL DIM  time propagation of each eigenstate with it's corresponding eigenenergy
    Eval, Evec = EigenEnVecs
    w = np.dot(1 / (np.sqrt(normconst(EigenSpan(EigenEnVecs, Nstate)))), (EigenSpan(EigenEnVecs, Nstate)))
    t = np.arange(0, T_max, 0.05)
    y = 0
    for t in np.nditer(t):
        Z_2 = np.zeros(2 ** n_tot)
        Z_2t = np.zeros(2 ** n_tot)
        for j in range(0, np.size(Eval)):  # alternative way- just multiply evecs as orthogonal ones (easier)
            Z_2 = Z_2 + np.dot(w[j], Evec[:, j])  # Z_2 spanned in eigenstate basis as Cols of a matrix
            Z_2t = Z_2t + np.dot(np.dot((np.exp(-1j * Eval[j] * t)), w[j]),
                                 Evec[:, j])  # Z_2(t) spanned in eigenstate basis as Cols of a matrix
        y = (np.absolute(np.dot(np.conjugate((Z_2)), (Z_2t)))) ** 2
        plt.plot(t, np.round(y, 4), marker='.', color=Color)
        plt.xlabel('$t$')
        plt.ylabel(r'$|\langle\mathbb{Z}_{2}|\mathbb{Z}_{2}(t)\rangle|^{2}$')
        # plt.savefig('new fidelity_12atoms.pdf')
        plt.title('Quantum Fidelity of the Neel State vs. Time')
    #plt.show()
#TODO FIX NORMALIZATION!!!


def RunTimeProp(n_tot, n, Coupl=Z_i, h_x=1, h_z=1, T_max=20):#only even numbers right now
    """
    Runs TimeProp
    """
    H = PXPBathHam(n_tot, n, Coupl, h_x, h_z)
    # H= PXPBathHamUncoup(n_tot, n, Coupl, h_x, h_z) # Uncoupled version
    EV = EvecEval(H)
    Neel = Neelstate(n_tot)
    Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
    TimeProp(EV, n_tot, Neel, T_max,Color)


#TODO FIX NORMALIZATION!!!
#TODO CHeck why it doesn't work for 8 tot and 8 pxp
# TimeProp(EvecEval(PXPBathHam(n_tot,n,Z_i,1,1)), n_tot, Neelstate(n_tot),T_max)
# TODO- does not work without coupling now(When taking 8 and 8)
# TODO- play with bigger number of atoms

def RunRmetric(n, h_x, h_z, Hamiltonian):
    H = Hamiltonian(n, h_x, h_z)
    EV = la.eigvalsh(H)
    # print(EV)
    return RMeanMetric(EV)


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
# theta=np.linspace(1.5,1.64,1000)
# for t in np.nditer(theta):
#     r=RunRmetric(11, np.sin(t), np.cos(t), TIOBCNew)
#     plt.plot(t, r,  marker= '.', color='c4')
# plt.title('11 Atoms')
# plt.show()

if __name__ == '__main__':
    print('adam')
