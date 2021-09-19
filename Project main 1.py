import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

g= np.array([0,1]) #ground state
# print("ground=\n", g)

r= np.array([1,0]) #rydberg state
# print("rydberg=\n", r)

X_i = np.array([[0,1],[1,0]]) # flips ground -> rydberg/ rydberg-> ground |g><r|+|r><g|
# print("X_i = \n", X_i)

Q_i=np.array([[1,0],[0,0]]) # projector on rydberg state (density of excited states) |r><r|
# print("Q_i = \n", Q_i)

P_i=np.array([[0,0],[0,1]]) # projector on ground state (density of ground states) |g><g|
# print("P_i = \n",P_i)

def PXPHamOBC(n): # open boundary conditions Hamiltonian. n= no. of atoms>2, d=2^n dimension of hilbert space #todo edges not good!
    d= 2**n
    pxp_fin=np.zeros((d, d))
    for i in range(1, n - 1):  # goes over all atoms for total sum in the end
            piminus1 = P_i
            xi = X_i
            piplus1 = P_i
            for m in range(0, n - 1): # for kronecker product of P_(i-1)
                if m < i-1:
                    piminus1 = np.kron(np.identity(2), piminus1)
                else:
                    piminus1 = np.kron(piminus1, np.identity(2))

            for t in range(0, n - 1):  # for kronecker product of X_(i)
                if t < i:
                    xi = np.kron(np.identity(2), xi)
                else:
                    xi = np.kron(xi, np.identity(2))

            for c in range(0, n - 1): # for kronecker product of P_(i+1)
                if c < i+1:
                    piplus1 = np.kron(np.identity(2), piplus1)
                else:
                    piplus1 = np.kron(piplus1, np.identity(2))

            #for f in range(0, n - 1):  #todo edges
                    # piplus1 = np.kron(np.identity(2), piplus1)
                    # xi = np.kron(np.identity(2), xi
            pxp_ar = np.matmul(np.matmul(piminus1,xi),piplus1) # calculates PXP for a given atom
            pxp_fin = np.add(pxp_fin, pxp_ar) # cumulative sum of arrays
    return pxp_fin.astype('int32') # sums over all matrices kept in the 3d array
# print("\n PXP= \n", PXPHamOBC(3))



def PXPHamPBC(n): # periodic boundary conditions Hamiltonian. n= no. of atoms>2, d=2^n dimension of hilbert space
    d= 2**n
    pxp_fin=np.zeros((d, d))
    for i in range(0, n):  # goes over all atoms for total sum in the end
            piminus1 = P_i
            xi = X_i
            piplus1 = P_i
            for m in range(0, n - 1): # for kronecker product of P_(i-1)
                if i == 0:
                    piminus1 = np.kron(np.identity(2), piminus1)
                elif m < i-1:
                    piminus1 = np.kron(np.identity(2), piminus1)
                else:
                    piminus1 = np.kron(piminus1, np.identity(2))

            for t in range(0, n - 1):  # for kronecker product of X_(i)
                if t < i:
                    xi = np.kron(np.identity(2), xi)
                else:
                    xi = np.kron(xi, np.identity(2))

            for c in range(0, n - 1): # for kronecker product of P_(i+1)
                if i == n-1:
                    piplus1 = np.kron(piplus1, np.identity(2))
                elif c < i+1:
                    piplus1 = np.kron(np.identity(2), piplus1)
                else:
                    piplus1 = np.kron(piplus1, np.identity(2))

            pxp_ar = np.matmul(np.matmul(piminus1,xi),piplus1) # calculates PXP for a given atom
            pxp_fin = np.add(pxp_fin, pxp_ar) # cumulative sum of arrays
    return pxp_fin.astype('int32') # sums over all matrices kept in the 3d array
# print("\n PXP= \n", PXPHamPBC(5))



def SubspaceP(n): # Projector on the constrained hilbert subspace
    d=2**n
    ProjP = np.identity(d)
    for i in range(0,n):
        qi = Q_i
        qiplus1 = Q_i
        for t in range(0, n - 1):  # for kronecker product of qi
            if t < i:
                qi = np.kron(np.identity(2), qi)
            else:
                qi = np.kron(qi, np.identity(2))

        for m in range(0, n - 1): # for kronecker product of qiplus1
            if i == n-1:
                qiplus1 = np.kron(qiplus1,np.identity(2))
            elif m < i+1:
                qiplus1 = np.kron(np.identity(2), qiplus1)
            else:
                qiplus1 = np.kron(qiplus1, np.identity(2))
        ProjP = np.matmul(ProjP,(np.subtract(np.identity(d),np.matmul(qi,qiplus1))))
    return ProjP.astype('int')
print("\n Constrained subspace:\n", SubspaceP(4))

def Hilvec(n, Mat): # checking between which states a given matrix connects
    d=2**n
    print("Matrix = \n", Mat)
    for j in range (0,d):
        Dimvec = np.zeros(d)
        Dimvec[j] = 1
        PXP_Dimvec= np.dot(Mat, Dimvec)
        if np.any(PXP_Dimvec != 0):
            # print("\n Dimvec= ", Dimvec)
            print("\n Dimvec= ", np.argwhere(Dimvec)+1)
            # print("\n PXP*Dimvec= ", PXP_Dimvec)
            print("\n PXP*Dimvec=", np.argwhere(PXP_Dimvec)+1)
    return
# Hilvec(4, SubspaceP(4))

def Z_1(n):
        Z_1 = np.array([[1, 0], [0, -1]])  # pauli Z matrix
        for i in range(0, n - 1):
            Z_1 = np.kron(Z_1, np.identity(2)).astype('int32')
        return Z_1
# print(Z_1(4))

def O_z(n):
    d = 2 ** n
    O_zsum = np.zeros((d, d))
    for i in range(0, n):
        Z_i = np.array([[1, 0], [0, -1]])  # pauli Z matrix
        for j in range(0, n-1):
            if j < i:
                Z_i = np.kron(np.identity(2), Z_i)
            else:
                Z_i = np.kron(Z_i, np.identity(2))
        O_zsum = np.add(O_zsum, Z_i)  # summation over n pauli Z matrices of dimension d
    O_zop= (1 / n) * O_zsum
    return O_zop
# print(O_z(4))

def ConstrHam(Ham,Constr): #returns PXP ham in constrained subspace
    print("\n PXP= \n", Ham)
    print("\n Constrained subspace:\n", Constr)
    ConstrPXP= np.matmul(np.matmul(Constr, Ham),Constr)
    print("\n Constrained Hilbert subspace PXP Hamiltonian:\n",ConstrPXP)
    return ConstrPXP
# ConstrHam(PXPHamPBC(5),SubspaceP(5))

def effPXP_PBC(n,Ham,Constr): # without 2**(n-2) first rows/coloums
    d=2**n
    EffPXP= ConstrHam(Ham, Constr)[2**(n-2):d, 2**(n-2): d]
    print("\n Effective PXP Hamiltonian \n", EffPXP)
    return EffPXP
#effPXP_PBC(n, PXPHamPBC(n), SubspaceP(n))

def MatCompare(Ham, Constr): #checks if 2 matrices have same entries, returns entry output matrix if not
    v= ConstrHam(Ham,Constr)
    w= Ham
    if np.array_equal(v,w) == False:
        print(np.array_equal(v,w))
        EntComMat= np.equal(v,w)
        print(EntComMat) #falses are the states diluted by constraint
    else:
        print(np.array_equal(v, w))
# MatCompare(PXPHamPBC(5),SubspaceP(5))

def DiagHam(Ham): #calculates eigenvalues and eigenstates of Hamiltonian
    eval, evec = la.eigh(Ham)
    print("\n eigenvalues: ", np.round(eval,4))
    print("\n eigenvectors: \n", np.round(np.transpose(evec),4))
    return np.round(eval,4), np.round(np.transpose(evec),4)
# DiagHam(PXPHamPBC(5))
# DiagHam(ConstrHam(PXPHamPBC(5),SubspaceP(5)))


def SubspcEvecs(n, Ham, Constr): #returns Evecs and Evals of the PXP Hamiltonian IN THE CONSTRAINED HILBERT SUBSPACE!
    d= 2**n
    eval, evec = DiagHam(Ham)
    print(eval, evec)
    x= np.zeros(d)
    y= np.zeros((d,d))
    for j in range(0, d-2**(n-2)):
        x[j]= eval[j]
        y[j,:]= np.dot(Constr, evec[j,:])
    print("\n good ol' x: \n", x)
    print("\n good ol' y: \n", y)
    ConstrEval= x[np.any(y != 0, axis=1)]
    ConstrEvec= y[np.any(y != 0, axis=1), :]
    print("\n Constrained EigenValues:", ConstrEval)
    print("\n Constrained EigenVectors:", ConstrEvec)
    return ConstrEval, ConstrEvec
# n=5
# SubspcEvecs(n, PXPHamPBC(n), SubspaceP(n))



def Expect_op(n, op, EigenHam): #calculates expectation value of operator of choice with respect to eigenvecs of Hamiltonian of choice
    d= 2**n
    Eval, Evec = EigenHam
    y= np.zeros(np.size(Eval))
    for j in range(0, np.size(Eval)):
        Eval[j]=np.round(Eval[j],5)
        y[j]= np.round(np.dot(Evec[j,:], np.dot(op,Evec[j,:])),5)
        print("\n <Operator(", Eval[j], ")>:", y[j])
    #Eval2= Eval[Eval >= 0.1], Eval[Eval <= -0.1] #masking points that are E= 0 # everything is masked from E=0 points
    #y= y[Eval >= 0.1], y[Eval <= -0.1] #masking points that are E= 0
    plt.plot(Eval,  y, '.')
    plt.ylim(ymax=-0.2, ymin=-0.63)
    return plt.show()

i = 8
# Expect_op(i, Z_1(i), SubspcEvecs(i, PXPHamPBC(i), SubspaceP(i))) #expected value of Z_1 w/  H's filtered vectors
# Expect_op(i, O_z(i), SubspcEvecs(i, PXPHamPBC(i), SubspaceP(i))) #expected value of O_z w/  H's filtered vectors

#Expect_op(i, Z_1(i), SubspcEvecs(i, ConstrHam(PXPHamPBC(i), SubspaceP(i)), SubspaceP(i))) # expected value of Z_1 w/ constrained H's filtered vectors
# Expect_op(i, O_z(i), SubspcEvecs(i, ConstrHam(PXPHamPBC(i), SubspaceP(i)), SubspaceP(i))) # expected value of O_z w/ constrained H's filtered vectors

# Expect_op(i, Z_1(i), DiagHam(ConstrHam(PXPHamPBC(i), SubspaceP(i)))) #expected value of Z_1 w/ constrained H's eigenvectors
# Expect_op(i, O_z(i), DiagHam(ConstrHam(PXPHamPBC(i), SubspaceP(i)))) #expected value of O_z w/ constrained H's eigenvectors




def Comrelate(n, op, Ham): # commutation relation of matrices
    d= 2**n
    print("\n", Ham, "\n")
    print("\n", op, "\n")
    y = np.matmul(Ham, op)
    x = np.matmul(op, Ham)
    Com= np.subtract(x,y)
    print(x, "\n\n",y)
    print("Com:\n", Com)
    if np.all(Com == 0):
        print("does commute")
    else:
        print("does not commute")
    return Com
# Comrelate(7, PXPHamPBC(7),O_z(7))


