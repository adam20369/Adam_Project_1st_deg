import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import integrate

#####OPERATOR AND BASIS DEFINITIONS#####

g= np.array([0,1]) #ground state
# print("ground=\n", g)

r= np.array([1,0]) #rydberg state
# print("rydberg=\n", r)

X_i=np.array([[0,1],[1,0]]) # flips ground -> rydberg/ rydberg-> ground |g><r|+|r><g|
# print("X_i = \n", X_i)

Q_i=np.array([[1,0],[0,0]]) #projector on rydberg state (density of excited states) |r><r|
# print("Q_i = \n", Q_i)

P_i=np.array([[0,0],[0,1]]) #projector on ground state (density of ground states) |g><g|
# print("P_i = \n",P_i)

Z_i=np.array([[1,0],[0,-1]]) # pauli Z


def Z_1(n):
    Z_1 = np.array([[1, 0], [0, -1]])  # pauli Z matrix
    for i in range(0, n - 1):
        Z_1 = np.kron(Z_1 , np.identity(2)).astype('int32')
    return Z_1
# print(Z_1(3))

def O_z(n):
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
#print(O_z(3))

def T_op(n): #translation operator for n sites
    d= 2**n
    r=np.array((d/2)).astype('int')
    T= np.zeros((d,d), dtype=int)
    for m in range(0, d):
        for i in range(0, d):
            if i == 2*m:
                T[m,i]= 1
            elif i== 2*m+1:
                T[m+r,i]= 1
    return T
# print (T_op(4))

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
    return pxp_fin.astype('int32')
# print("\n PXP= \n", PXPHamPBC(4))

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

        for m in range(0, n - 1):  # for kronecker product of qiplus1
            if i == n-1:
                qiplus1 = np.kron(qiplus1,np.identity(2))
            elif m < i+1:
                qiplus1 = np.kron(np.identity(2), qiplus1)
            else:
                qiplus1 = np.kron(qiplus1, np.identity(2))
        ProjP = np.matmul(ProjP,(np.subtract(np.identity(d),np.matmul(qi,qiplus1))))
    return ProjP.astype('int')
# print("\n Constrained subspace:\n", SubspaceP(4))

def SubspaceMat(n, Mat): #Outputs a block matrix in Our subspace
    SubspcRCs= np.nonzero(np.any(SubspaceP(n)!=0, axis=0)) # gives number of rows and coloums! that are included in subspace
    #print("\n states relevant to subspace: \n", SubspcRCs)
    SubspcMat= np.squeeze(Mat[SubspcRCs,:])
    SubspcMat=np.squeeze(SubspcMat[:,SubspcRCs])
    #print("\n block Matrix in constrained subspace: \n", SubspcMat)
    #print(SubspcMat.shape)
    return SubspcMat
#p=5
#SubspaceMat(p, PXPHamPBC(p))
#SubspaceMat(p, Z_1(p))
#SubspaceMat(p, O_z(p))
#SubspaceMat(p, T_op(p))

def Subspccount(n): #Outputs The dimensions (Rows #/ Col #) of subspace when given the number of atoms
    SubspcRCs = np.nonzero(np.any(SubspaceP(n) != 0, axis=0))  # gives number of rows and coloums! that are included in subspace
    # print("\n states relevant to subspace: \n", SubspcRCs)
    SubspcMat = np.squeeze(PXPHamPBC(n)[SubspcRCs, :])
    SubspcMat = np.squeeze(SubspcMat[:, SubspcRCs])
    return np.shape(SubspcMat)[0]
#print(Subspccount(8))

def ConstrHam(Ham,Constr): #returns PXP ham in constrained subspace in same dimension as it was
    print("\n PXP= \n", Ham)
    print("\n Constrained subspace:\n", Constr)
    ConstrPXP= np.matmul(np.matmul(Constr, Ham),Constr)
    print("\n Constrained Hilbert subspace PXP Hamiltonian:\n",ConstrPXP)
    return ConstrPXP
# ConstrHam(PXPHamPBC(5),SubspaceP(5))


def MatCompare(Mat1, Mat2): #checks if 2 matrices have same entries, returns entry output matrix if not
    v= Mat1
    w= Mat2
    if np.array_equal(v,w) == False:
        print(np.array_equal(v,w))
        EntComMat= np.equal(v,w)
        print(EntComMat)
    else:
        print(np.array_equal(v, w))
# p=10
# MatCompare(SubspaceMat(p,PXPHamPBC(p)),SubspaceMat(p,ConstrHam(PXPHamPBC(p),SubspaceP(p))))

def DiagHer(Ham): #calculates eigenvalues and eigenstates of hermitian Hamiltonian
    eval, evec = la.eigh(Ham)
    print("\n eigenvalues: ", np.round(eval,4))
    print("\n eigenvectors: \n", np.round(np.transpose(evec),4))
    return eval, evec
#v=5
# DiagHer(SubspaceMat(v,PXPHamPBC(v)))
# DiagHer(ConstrHam(PXPHamPBC(v),SubspaceP(v)))

def DiagonalH(Mat): #calculates eigenvalues and eigenstates of Matrix (hermitian)
    eval, evec = la.eigh(Mat)
    print("\n eigenvalues: ", np.round(eval ,4))
    print("\n eigenvectors: \n", np.round(np.transpose(evec),4))
    return eval, evec
#DiagonalH(O_z(4))


def Unitcheck(mat,l):
    y=np.matmul(np.transpose(mat),mat)
    if np.all(y== np.identity(2**l)) == True:
        print( "\n Unitary \n")
    else:
        print("\n Non-Unitary \n")
    return
#l=4
#Unitcheck(T_op(l),l)

##### CSCO THEOREM- FINDING THE UNIQUE BASIS ######

m=6
def DiagonalTEvals(Mat): #calculates eigenvalues and eigenstates of translation Matrix (Non hermitian)
    eval, evec = la.eig(Mat)
    return np.round(eval,4)
#print("\n Translation matrix Eigenvalues: \n", DiagonalTEvals(SubspaceMat(m, T_op(m))))

def DiagonalTEvec(Mat): #calculates eigenstates of Translation Matrix
    eval, evec = la.eig(Mat)
    return np.round(evec,4)
#print( "\n Translation matrix Eigenvectors \n", DiagonalTEvec(SubspaceMat(m, T_op(m))))

def DiagonalPEvals(Mat):  #calculates eigenvalues and eigenstates of SORTED!! momentum Matrix
    eval, evec = la.eig(Mat)
    return np.round((np.log(eval) * 1j),4)
#print("\n P matrix Eigenvalues: \n", DiagonalPEvals(SubspaceMat(m, T_op(m))))

def SortedPEvals(Mat):  #calculates eigenvalues and eigenstates of SORTED!! momentum Matrix
    eval, evec = la.eig(Mat)
    return np.sort(np.round((np.log(eval) * 1j),4),0)
#print("\n P matrix SORTED Eigenvalues: \n", SortedPEvals(SubspaceMat(m, T_op(m))))

def SortedP_0pos(Mat): #Outputs position of (sorted) T=1 / P=0 eigenvalues (& eigenvector column #)
    B = SortedPEvals(Mat)
    return np.where(B==0+0j)
#print("\n Positions of P=0 Vectors \n", SortedP_0pos(SubspaceMat(m, T_op(m))))

def SortedTEvec(Mat): #calculates SORTED (by P matrix eigenvalues) eigenbasis of Translation Matrix
    eval, evec = la.eig(Mat)
    P_sort = np.argsort(np.round((np.log(eval) * 1j),4),0)
    #print("\n Sorting order ", P_sort)
    return np.round(evec,3)[:,P_sort] #returns as col. vectors
#print( "\n Sorted Translation matrix Eigenvectors \n", SortedTEvec(SubspaceMat(m, T_op(m))))

def BlockSortedH(EigenBasis,HMat): #transforms a Matrix (H) to the basis of SORTED Diagonal P matrix
    eval, evec = DiagonalTEvals(EigenBasis), SortedTEvec(EigenBasis)
    y = np.matmul(np.transpose(np.conjugate(evec)), np.matmul(HMat, evec))
    return np.round(y,2)
#print("\n Hamiltonian in sorted Momentum Basis: \n", BlockSortedH(SubspaceMat(m, T_op(m)), SubspaceMat(m, PXPHamPBC(m))))

def P_0PXPBlock(Ham,Op,BlockArr): #Gets relevant Hamiltonian diagonalized block energies &  Eigenvectors AFTER TRANSFORMATION TO THE RELEVANT BLOCK! (subspace Dim!!)
    BlockH=BlockSortedH(Op, Ham)
    PXPVecs=np.squeeze(BlockH[BlockArr, :])
    #print("\n P=0 block PXP Hamiltonian row vectors (Subspace Dim) \n", PXPVecs)
    PXPBlock=np.squeeze(PXPVecs[:, BlockArr])
    #print("\n Relevant Halmitonian block \n", PXPBlock)
    BEval, BEvec = la.eigh(PXPBlock)
    #print("\n P=0 Block Hamiltonian Energies \n", np.round(BEval,3))
    #print("\n H's P=0 Block Eigenvectors (rows) \n", np.transpose(np.round(BEvec,3)))
    BlockH[(np.amin(BlockArr)):(np.amax(BlockArr)+1), (np.amin(BlockArr)):(np.amax(BlockArr)+1)]= np.transpose(np.round(BEvec,4))
    PXPVecNew=np.squeeze(BlockH[BlockArr, :])
    #print("\n H's P=0 Block full Eigenvec  tor rows (Subspace Dim)\n", np.round(PXPVecNew,3))
    return np.round(BEval,4), np.transpose(np.round(PXPVecNew,3))  # returns Eigenenergies and Eigenvectors [col!!!] (subspace dim) of relevant block
#print(P_0PXPBlock(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), SortedP_0pos(SubspaceMat(m, T_op(m)))))

def OpBasisTrans(EigenBasis,op): #transforms an operator (O_z, Z_1 for e.g) to the relevant block H' basis (= sorted T basis)
    TBasis = EigenBasis
    y = np.matmul(np.conjugate(np.transpose(TBasis)), np.matmul(op, TBasis))
    #print("\n Operator in Eigenbasis: \n", np.round(y,3))
    return np.round(y,3)
#OpBasisTrans(SortedTEvec(SubspaceMat(m, T_op(m))),SubspaceMat(m, Z_1(m)))
#OpBasisTrans(SortedTEvec(SubspaceMat(m, T_op(m))),SubspaceMat(m, O_z(m)))

def Expect_op(PXPBlockEV, TransOp): #Fig 2 A!!!! ## calculates expectation value of operator of choice with respect to eigenvectors & energies of Hamiltonian of choice
    Eval, SubspcEvec = PXPBlockEV
    y=np.matmul(np.transpose(np.conjugate(SubspcEvec)), np.matmul(TransOp,SubspcEvec))
    for j in range(0, np.size(Eval)):
        print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y[j,j],3)))
        plt.plot(Eval[j],  np.real(y[j,j]), '.')
    return plt.show()
#Expect_op(P_0PXPBlock(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), SortedP_0pos(SubspaceMat(m, T_op(m)))), OpBasisTrans(SortedTEvec(SubspaceMat(m, T_op(m))),SubspaceMat(m, Z_1(m)))) #expected value of Z_1 w/ constrained H's eigenvectors
#Expect_op(P_0PXPBlock(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), SortedP_0pos(SubspaceMat(m, T_op(m)))), OpBasisTrans(SortedTEvec(SubspaceMat(m, T_op(m))),SubspaceMat(m, O_z(m)))) #expected value of Z_1 w/ constrained H's eigenvectors

def AltExpect_op(PXPBlockEV, TransOp): #Fig 2 A Alternative way than expect_Op
    Eval, SubspcEvec = PXPBlockEV
    for j in range(0, np.size(Eval)):
        y=np.dot(np.conjugate(SubspcEvec[:,j]),np.dot(TransOp,SubspcEvec[:,j]))
        print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y,3)))
        plt.plot(Eval[j],  np.real(y), '.')
    return plt.show()
#AltExpect_op(P_0PXPBlock(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), SortedP_0pos(SubspaceMat(m, T_op(m)))), OpBasisTrans(SortedTEvec(SubspaceMat(m, T_op(m))),SubspaceMat(m, Z_1(m)))) #expected value of Z_1 w/ constrained H's eigenvectors
#AltExpect_op(P_0PXPBlock(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), SortedP_0pos(SubspaceMat(m, T_op(m)))), OpBasisTrans(SortedTEvec(SubspaceMat(m, T_op(m))),SubspaceMat(m, O_z(m)))) #expected value of Z_1 w/ constrained H's eigenvectors

def EigenEnVecsFin(Ham, T_op, n): #outputs PXP H's P=0 sector full eigenstates - PD (P are P eigenvectors and D are block eigenvectors)
    FullSubspcEV= np.zeros((Subspccount(n),Subspccount(n))).astype('complex')
    BEVal, SubspcEvec= P_0PXPBlock(Ham, T_op, SortedP_0pos(T_op))
    SubspcEvec=np.squeeze(SubspcEvec[SortedP_0pos(T_op),:])
    FullSubspcEV[(np.amin(SortedP_0pos(T_op))):(np.amax(SortedP_0pos(T_op))+1), (np.amin(SortedP_0pos(T_op))):(np.amax(SortedP_0pos(T_op))+1)]= np.transpose(np.round(SubspcEvec,3))
    HBlockEV=np.transpose(FullSubspcEV) #subspc dim DxD matrix with H block eigenvectors (COL!)
    PBlockEV= SortedTEvec(T_op)  #subspc dim DxD matrix with P sorted eigenvectors (COL!)
    #print(HBlockEV, PBlockEV)
    PDmat=np.matmul(PBlockEV, HBlockEV) #gives transformation matrix PD ((PD)^-1*H*(PD)
    #print("\n PDmat \n", PDmat)
    #print("\n Final Row Eigenvectors of H \n", np.squeeze(np.transpose(np.round(PDmat,3)[:,SortedP_0pos(T_op)])))
    return BEVal, np.squeeze(np.round(PDmat,3)[:,SortedP_0pos(T_op)]) #Returns final eigenstates of H (COLS)
#print(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m))

#### END OF CSCO ###

def SortedTBlockEvec(T_op,BlockArr): #returns P=0 block of T's Eigenvectors (Col)
    BlockT=np.squeeze(SortedTEvec(T_op))[:,BlockArr]
    BlockT=np.squeeze(BlockT[BlockArr, :])
    #print("\n T Block Matrix \n", BlockT)
    return BlockT
#SortedTBlockEvec(SubspaceMat(m, T_op(m)), SortedP_0pos(SubspaceMat(m, T_op(m))))

def Fig2A(EigenEnVecs,Op): #checking the expectation values with final H eigenvecs and standard basis operators (Subspace dim)
    Eval, Evec= EigenEnVecs
    y=np.matmul(np.transpose(np.conjugate(Evec)), np.matmul(Op,Evec))
    for j in range(0, np.size(Eval)):
        #print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y[j,j],3)))
        plt.plot(Eval[j],  np.real(y[j,j]), marker='.', color='C2')
    plt.xlabel('$A_n/A_{n-1}$')
    plt.ylabel(r'$\langle\hat{O}^{Z}\rangle$')
    #plt.savefig('Expectation_Value_OZ.pdf')
    return plt.show()
# Fig2A(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m),SubspaceMat(m, O_z(m)))
#Fig2A(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m),SubspaceMat(m, Z_1(m)))

#TODO I GOT UP TO HERE WITH TRANSFERING CODES TO THE NEW PROJECT!!!!!!

def Fig2Aalt(EigenEnVecs,Op): #an alternative way of checking the expectation values
    Eval, Evec = EigenEnVecs
    for j in range(0, np.size(Eval)):
        y=np.dot(np.conjugate(Evec[:,j]),np.dot(Op,Evec[:,j]))
        #print("\n <Operator(", Eval[j], ")>:", np.real(np.round(y,3)))
        plt.plot(Eval[j], np.real(y), marker='.', color='blue')
    plt.xlabel('$A_n/A_{n-1}$')
    plt.ylabel(r'$\langle\hat{O}^{Z}\rangle$')
    #plt.savefig('Expectation_Value_OZ.pdf')
    return plt.show()
#Fig2Aalt(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m), SubspaceMat(m, O_z(m)))
#Fig2Aalt(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m), SubspaceMat(m, Z_1(m)))

def Neelstate(n): #outputs the neel state (Z_2) in standard basis in subspace dim!! ONLY FOR EVEN n!!!!
    d=2**n
    k=np.array(n/2).astype('int32')
    Neelproj=np.kron(Q_i,P_i).astype('int32')
    for i in range(0,k-1):
      Neelproj=np.kron(np.kron(Neelproj,Q_i),P_i)
    SubspcNproj=SubspaceMat(n,Neelproj)
    Nstate= np.squeeze(SubspcNproj[np.nonzero(SubspcNproj),:])[0,:]
    return Nstate
#print(Neelstate(4)) #ONLY EVEN n!!!

####### Conclusion- neel state is always the first state in subspace!!!!######

def Fig3B(EigenEnVecs,Nstate): #ONLY EVEN NUM OF ATOMS
    Eval, Evec= EigenEnVecs
    for j in range(0, np.size(Eval)):
        #print("\n <Operator(", Eval[j], ")>:", np.log10((np.absolute(np.dot(Nstate,Evec[:,j])))**2))
        plt.plot(Eval[j], np.log10((np.absolute(np.dot(Nstate,Evec[:,j])))**2), marker='.', color='C2')
    plt.xlabel('$E$')
    plt.ylabel(r'$log_{10}(|\langle\mathbb{Z}_{2}|\psi\rangle|)^{2}$')
    #plt.title('Overlap of Neel State with Eigenstates vs. Eigenstate Energy')
    plt.savefig('Overlap_12_atoms.pdf')
    return plt.show()
#Fig3B(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m), Neelstate(m))

def EigenSpan(EigenEnVecs,Nstate): #outputs weights (inner product) of Neel state spanned in eigenstate basis (subspace dim)
    Eval, Evec = EigenEnVecs
    Z_2= Nstate
    y= np.zeros(np.size(Eval)).astype('complex')
    for j in range(0, np.size(Eval)):
        y[j]=np.dot(Z_2,Evec[:,j])
    #print("\n array of <Z_2|EigenVec(j)>:", np.real(y))
    return y
#EigenSpan(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m), Neelstate(m))

def normconst(EigenSpan): #Normalization check of the inner products
    y=EigenSpan
    sum= np.dot(np.conjugate(y),y)
    #print("\n sum \n", np.round(np.real(sum),3))
    return np.round(np.real(sum),3)
#normconst(EigenSpan(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m), Neelstate(m)))

def TimeProp(EigenEnVecs,Subspcdim,Spans): #time propagation of each eigenstate with it's corresponding eigenenergy
    Eval, Evec = EigenEnVecs
    w = Spans
    t = np.arange(0,30,0.1)
    Z_2= np.zeros((Subspcdim,np.size(Eval)))
    Z_2t= np.zeros((Subspcdim,np.size(Eval)))
    y= 0
    for t in np.nditer(t):
        for j in range(0, np.size(Eval)): #alternative way- just multiply evecs as orthogonal ones (easier)
            Z_2[:,j]= np.dot(w[j],Evec[:,j]) # Z_2 spanned in eigenstate basis as Cols of a matrix
            Z_2t[:,j]= np.dot(np.dot((np.exp(-1j*Eval[j]*t)),w[j]),Evec[:,j]) #Z_2(t) spanned in eigenstate basis as Cols of a matrix
            y= y +(np.dot(np.conjugate((Z_2[:,j])),(Z_2t[:,j])))
        y= np.absolute(y)
        plt.plot(t, np.round(y**2,3), '.')
    return plt.show()
#TimeProp(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m),Subspccount(m), EigenSpan(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m), Neelstate(m)))

def TimePropnew(EigenEnVecs,Subspcdim,Spans): #time propagation of each eigenstate with it's corresponding eigenenergy
    Eval, Evec = EigenEnVecs
    w = np.dot(1/(np.sqrt(normconst(Spans))),(Spans))
    t = np.arange(0,20,0.05)
    y = 0
    for t in np.nditer(t):
        Z_2 = np.zeros(Subspcdim)
        Z_2t = np.zeros(Subspcdim)
        for j in range(0, np.size(Eval)): #alternative way- just multiply evecs as orthogonal ones (easier)
            Z_2= Z_2+np.dot(w[j],Evec[:,j]) # Z_2 spanned in eigenstate basis as Cols of a matrix
            Z_2t= Z_2t+np.dot(np.dot((np.exp(-1j*Eval[j]*t)),w[j]),Evec[:,j]) #Z_2(t) spanned in eigenstate basis as Cols of a matrix
        y= (np.absolute(np.dot(np.conjugate((Z_2)),(Z_2t))))**2
        plt.plot(t, np.round(y,4), marker='.', color='C2')
        plt.xlabel('$t$')
        plt.ylabel(r'$|\langle\mathbb{Z}_{2}|\mathbb{Z}_{2}(t)\rangle|^{2}$')
        plt.savefig('fidelity_12atoms.pdf')
        #plt.title('Quantum Fidelity of the Neel State vs. Time')
    return plt.show()
TimePropnew(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m),Subspccount(m), EigenSpan(EigenEnVecsFin(SubspaceMat(m, PXPHamPBC(m)), SubspaceMat(m, T_op(m)), m), Neelstate(m)))


#==================================PXP BATH!======================================#

def PXPHamOBC(n):
    d= 2**n
    pxp_fin = np.zeros((d, d))
    for i in range(1, n+1):  # goes over all atoms for total sum in the end
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
            #OKOKOKOKOKOKOKOKOKOK
        for t in range(1, n):  # for kronecker product of X_(i)
            if t < i:
                xi = np.kron(np.identity(2), xi)
            else:
                xi = np.kron(xi, np.identity(2))
            #OKOKOKOKOKOKOKOKOKOK
        for c in range(1, n):  # for kronecker product of P_(i+1)
            if i == n:
                piplus1 = np.identity(d)
            elif c < i + 1:
                piplus1 = np.kron(np.identity(2), piplus1)
            else:
                piplus1 = np.kron(piplus1, np.identity(2))
            #OKOKOKOKOKOKOKOKOK
        pxp_ar = np.matmul(np.matmul(piminus1, xi), piplus1)  # calculates PXP form for a given site
        pxp_fin = np.add(pxp_fin, pxp_ar)  # cumulative sum of PXP's
    return pxp_fin.astype('int32')

# print("\n PXP= \n", PXPHamOBC(4))

def TiltedIsingHam(n,h_x,h_z): # Tilted Ising Hamiltonian OBC n= no of atoms (must be =>2)
    d= 2**n
    pxp_fin = np.zeros((d, d))
    for i in range(1,n+1): # goes over all atoms for total sum in the end
        zi= Z_i
        ziplus1=Z_i
        xi= X_i
        for m in range(1, n): # For Z_i term (Z_1 up to Z_n)
            if m < i:
                zi = np.kron(np.identity(2), zi)
            else:
                zi = np.kron(zi, np.identity(2))
        for t in range(1, n): # For Z_i+1 term PBC
            if i == n:
                ziplus1= np.zeros((d,d)) # terminates Z_n*Z_n+1
            elif t < i + 1:
                ziplus1= np.kron(np.identity(2), ziplus1)
            else:
                ziplus1= np.kron(ziplus1, np.identity(2))
        for c in range(1, n): # For X_i term
            if c < i:
                xi= np.kron(np.identity(2), xi)
            else:
                xi= np.kron(xi, np.identity(2))
        pxp_ar =np.add(np.add(np.matmul(zi, ziplus1),(h_z)*zi),(h_x)*xi) #calculates hamiltonian PER i
        pxp_fin =np.add(pxp_fin,pxp_ar) # cumulative sum over i
    return pxp_fin.astype('int32')

# Zi*Zi+1 term always diagonal so Zi*Zi+1=Zi+1*Zi


def PXPIsing(ntot, n, hx, hz): #complete hamiltonian of coupled systems OBC, ntot= number of total atoms, n= number of PXP system
    d= 2**ntot #total dimension of hilbert space
    pxp_fin= np.zeros((d,d))
    for i in range(1,ntot + 1): #all initial declarations are already taking in account the separations of the systems)
        pimin1 = np.kron(P_i, np.identity(2**(ntot-n))) # of PXP model
        xiPXP = np.kron(X_i,np.identity(2**(ntot-n))) # of PXP model
        pipl1 = np.kron(P_i, np.identity(2**(ntot-n))) # of PXP model
        zi = np.kron(np.identity(2**(n-1)),Z_i) # of Tilted Ising model (n-1 due to coupling)
        zipl1 = np.kron(np.identity(2**(n-1)),Z_i) # of Tilted Ising model (n-1 due to coupling)
        xiTI = np.kron(np.identity(2**(n-1)),X_i) # of titled Ising model (n-1 due to coupling)
        for m in range(1,n): # for kronecker product of Pi-1
            if i > n-1: # stops the PXP chain after n atoms #todo change to n (think if we want to close OBC of PXP before TI or not
                pimin1= np.zeros((d,d))
            elif i == 1:  # for XP for first site (OBC)
                pimin1 = np.identity(d)
            elif m < i - 1:
                pimin1 = np.kron(np.identity(2), pimin1)
            else:
                pimin1 = np.kron(pimin1, np.identity(2))
        for t in range(1,n): # for kronecker product of Xi
            if i > n-1: # stops the PXP chain after n atoms #todo change to n (think if we want to close OBC of PXP before TI or not
                xiPXP= np.zeros((d,d))
            elif t < i:
                xiPXP = np.kron(np.identity(2), xiPXP)
            else:
                xiPXP = np.kron(xiPXP, np.identity(2))
        for c in range(1, n):  # for kronecker product of Pi+1
            if i > n-1: #stops the PXP chain after n atoms #todo we will never have to change this one, only add identity term like for i==0 for pi-1
                pipl1 = np.zeros((d,d))
            elif c < i+1:
                pipl1 = np.kron(np.identity(2), pipl1)
            else:
                pipl1 = np.kron(pipl1, np.identity(2))
# =======END OF PXP PART============
        for j in range(n, ntot): # for kronecker product of Zi term
            if i < n:
                zi = np.zeros((d,d))
            elif j < i:
                zi= np.kron(np.identity(2), zi)
            else:
                zi= np.kron(zi, np.identity(2))
        for b in range(n, ntot): # from kronecker product of Zi+1 term
            if i < n:
                zipl1 = np.zeros((d,d))
            elif b < i+1:
                zipl1 = np.kron(np.identity(2), zipl1)
            else:
                zipl1 = np.kron(zipl1, np.identity(2))
        for g in range(n, ntot): # for kronecker product of Xi term
            if i < n:
                xiTI = np.zeros((d,d))
            elif g < i:
                xiTI = np.kron(np.identity(2), xiTI)
            else:
                xiTI = np.kron(xiTI, np.identity(2))
        if i < ntot:
            pxp_ar = np.matmul(np.matmul(pimin1, xiPXP), pipl1) +np.matmul(zi,zipl1) + (hz) * (zi) + (hx) * (xiTI)
        else:
            pxp_ar= np.matmul(np.matmul(pimin1,xiPXP),pipl1)+(hz)*(zi)+(hx)*(xiTI)
        pxp_fin=np.add(pxp_ar,pxp_fin) #summation of each i's hamiltonian
    return pxp_fin.astype('int32')


def PXPIsing2(ntot, n, hx, hz): ### PXPIsing2 was meant to compare slightly different methods (doesn't count as a serious check..)
    d= 2**ntot #total dimension of hilbert space
    pxp_fin= np.zeros((d,d))
    for i in range(1,ntot + 1): #all initial declarations are already taking in account the separations of the systems)
        pimin1 = np.kron(P_i, np.identity(2**(ntot-n))) # of PXP model
        xiPXP = np.kron(X_i,np.identity(2**(ntot-n))) # of PXP model
        pipl1 = np.kron(P_i, np.identity(2**(ntot-n))) # of PXP model
        zi = np.kron(np.identity(2**(n-1)),Z_i) # of Tilted Ising model (n-1 due to coupling)
        zipl1 = np.kron(np.identity(2**(n-1)),Z_i) # of Tilted Ising model (n-1 due to coupling)
        xiTI = np.kron(np.identity(2**(n-1)),X_i) # of titled Ising model (n-1 due to coupling)
        for m in range(1,n): # for kronecker product of Pi-1
            if i > n-1: # stops the PXP chain after n atoms #todo change to n (think if we want to close OBC of PXP before TI or not
                pimin1= np.zeros((d,d))
            elif i == 1:  # for XP for first site (OBC)
                pimin1 = np.identity(d)
            elif m < i - 1:
                pimin1 = np.kron(np.identity(2), pimin1)
            else:
                pimin1 = np.kron(pimin1, np.identity(2))
        for t in range(1,n): # for kronecker product of Xi
            if i > n-1: # stops the PXP chain after n atoms #todo change to n (think if we want to close OBC of PXP before TI or not
                xiPXP= np.zeros((d,d))
            elif t < i:
                xiPXP = np.kron(np.identity(2), xiPXP)
            else:
                xiPXP = np.kron(xiPXP, np.identity(2))
        for c in range(1, n):  # for kronecker product of Pi+1
            if i > n-1: #stops the PXP chain after n atoms #todo we will never have to change this one, only add identity term like for i==0 for pi-1
                pipl1 = np.zeros((d,d))
            elif c < i+1:
                pipl1 = np.kron(np.identity(2), pipl1)
            else:
                pipl1 = np.kron(pipl1, np.identity(2))
# =======END OF PXP PART============
        for j in range(n, ntot): # for kronecker product of Zi term
            if i < n:
                zi = np.zeros((d,d))
            elif j < i:
                zi= np.kron(np.identity(2), zi)
            else:
                zi= np.kron(zi, np.identity(2))
        for b in range(n, ntot): # from kronecker product of Zi+1 term
            if i < n:
                zipl1 = np.zeros((d,d))
            elif i == ntot: #blows up last sum (it's uneccessary)
                zipl1= np.zeros((d,d))
            elif b < i+1:
                zipl1 = np.kron(np.identity(2), zipl1)
            else:
                zipl1 = np.kron(zipl1, np.identity(2))
        for g in range(n, ntot): # for kronecker product of Xi term
            if i < n:
                xiTI = np.zeros((d,d))
            elif g < i:
                xiTI = np.kron(np.identity(2), xiTI)
            else:
                xiTI = np.kron(xiTI, np.identity(2))
        pxp_ar = np.matmul(np.matmul(pimin1, xiPXP), pipl1) +np.matmul(zi,zipl1) + (hz) * (zi) + (hx) * (xiTI)
        pxp_fin=np.add(pxp_ar,pxp_fin) #summation of each i's hamiltonian
    return pxp_fin.astype('int32')

#we took PXP to be OBC only on the left hand side,



def PXPIsing3(ntot, n, hx, hz): #full OBC version for PXP system
    d= 2**ntot #total dimension of hilbert space
    pxp_fin= np.zeros((d,d))
    for i in range(1,ntot + 1): #all initial declarations are already taking in account the separations of the systems)
        pimin1 = np.kron(P_i, np.identity(2**(ntot-n))) # of PXP model
        xiPXP = np.kron(X_i,np.identity(2**(ntot-n))) # of PXP model
        pipl1 = np.kron(P_i, np.identity(2**(ntot-n))) # of PXP model
        zi = np.kron(np.identity(2**(n-1)),Z_i) # of Tilted Ising model (n-1 due to coupling)
        zipl1 = np.kron(np.identity(2**(n-1)),Z_i) # of Tilted Ising model (n-1 due to coupling)
        xiTI = np.kron(np.identity(2**(n-1)),X_i) # of titled Ising model (n-1 due to coupling)
        for m in range(1,n): # for kronecker product of Pi-1
            if i > n: # stops the PXP chain after n+1 atoms
                pimin1= np.zeros((d,d))
            elif i == 1:  # for XP for first site (OBC)
                pimin1 = np.identity(d)
            elif m < i - 1:
                pimin1 = np.kron(np.identity(2), pimin1)
            else:
                pimin1 = np.kron(pimin1, np.identity(2))
        for t in range(1,n): # for kronecker product of Xi
            if i > n: # stops the PXP chain after n+1 atoms
                xiPXP= np.zeros((d,d))
            elif t < i:
                xiPXP = np.kron(np.identity(2), xiPXP)
            else:
                xiPXP = np.kron(xiPXP, np.identity(2))
        for c in range(1, n):  # for kronecker product of Pi+1
            if i > n: #stops the PXP chain after n atoms
                pipl1 = np.zeros((d,d))
            elif i == n:
                pipl1 = np.identity(d)
            elif c < i+1:
                pipl1 = np.kron(np.identity(2), pipl1)
            else:
                pipl1 = np.kron(pipl1, np.identity(2))
# =======END OF PXP PART============
        for j in range(n, ntot): # for kronecker product of Zi term
            if i < n:
                zi = np.zeros((d,d))
            elif j < i:
                zi= np.kron(np.identity(2), zi)
            else:
                zi= np.kron(zi, np.identity(2))
        for b in range(n, ntot): # from kronecker product of Zi+1 term
            if i < n:
                zipl1 = np.zeros((d,d))
            elif i == ntot: #blows up last sum (it's uneccessary)
                zipl1= np.zeros((d,d))
            elif b < i+1:
                zipl1 = np.kron(np.identity(2), zipl1)
            else:
                zipl1 = np.kron(zipl1, np.identity(2))
        for g in range(n, ntot): # for kronecker product of Xi term
            if i < n:
                xiTI = np.zeros((d,d))
            elif g < i:
                xiTI = np.kron(np.identity(2), xiTI)
            else:
                xiTI = np.kron(xiTI, np.identity(2))
        pxp_ar = np.matmul(np.matmul(pimin1, xiPXP), pipl1) +np.matmul(zi,zipl1) + (hz) * (zi) + (hx) * (xiTI)
        pxp_fin=np.add(pxp_ar,pxp_fin) #summation of each i's hamiltonian
    return pxp_fin.astype('int32')


def PXPIsing4(ntot, n, hx, hz): #Full OBC PXP, faster method
    d= 2**ntot #total dimension of hilbert space
    pxp_fin= np.zeros((d,d))
    for i in range(1,ntot + 1): #all initial declarations are already taking in account the separations of the systems)
        pimin1 = np.kron(P_i, np.identity(2**(ntot-n))) # of PXP model
        xiPXP = np.kron(X_i,np.identity(2**(ntot-n))) # of PXP model
        pipl1 = np.kron(P_i, np.identity(2**(ntot-n))) # of PXP model
        zi = np.kron(np.identity(2**(n-1)),Z_i) # of Tilted Ising model (n-1 due to coupling)
        zipl1 = np.kron(np.identity(2**(n-1)),Z_i) # of Tilted Ising model (n-1 due to coupling)
        xiTI = np.kron(np.identity(2**(n-1)),X_i) # of titled Ising model (n-1 due to coupling)
        for m in range(1,n): # for kronecker product of Pi-1
            if i > n: # stops the PXP chain after n+1 atoms
                pimin1= np.zeros((d,d))
            elif i == 1:  # for XP for first site (OBC)
                pimin1 = np.identity(d)
            elif m < i - 1:
                pimin1 = np.kron(np.identity(2), pimin1)
            else:
                pimin1 = np.kron(pimin1, np.identity(2))
        for t in range(1,n): # for kronecker product of Xi
            if i > n: # stops the PXP chain after n+1 atoms
                xiPXP= np.zeros((d,d))
            elif t < i:
                xiPXP = np.kron(np.identity(2), xiPXP)
            else:
                xiPXP = np.kron(xiPXP, np.identity(2))
        for c in range(1, n):  # for kronecker product of Pi+1
            if i > n: # stops the PXP chain after n+1 atoms
                pipl1 = np.zeros((d,d))
            elif c < i+1:
                pipl1 = np.kron(np.identity(2), pipl1)
            else:
                pipl1 = np.kron(pipl1, np.identity(2))
# =======END OF PXP PART============
        for j in range(n, ntot): # for kronecker product of Zi term
            if i < n:
                zi = np.zeros((d,d))
            elif j < i:
                zi= np.kron(np.identity(2), zi)
            else:
                zi= np.kron(zi, np.identity(2))
        for b in range(n, ntot): # from kronecker product of Zi+1 term
            if i < n:
                zipl1 = np.zeros((d,d))
            elif b < i+1:
                zipl1 = np.kron(np.identity(2), zipl1)
            else:
                zipl1 = np.kron(zipl1, np.identity(2))
        for g in range(n, ntot): # for kronecker product of Xi term
            if i < n:
                xiTI = np.zeros((d,d))
            elif g < i:
                xiTI = np.kron(np.identity(2), xiTI)
            else:
                xiTI = np.kron(xiTI, np.identity(2))
        if i==ntot:
            pxp_ar= np.matmul(np.matmul(pimin1,xiPXP),pipl1)+(hz)*(zi)+(hx)*(xiTI)
        elif i==1:
            pxp_ar= np.matmul(xiPXP,pipl1)
        elif i < n:
            pxp_ar= np.matmul(np.matmul(pimin1,xiPXP),pipl1)
        elif i > n-1:
            pxp_ar = np.matmul(np.matmul(pimin1, xiPXP), pipl1) +np.matmul(zi,zipl1) + (hz) * (zi) + (hx) * (xiTI)
        pxp_fin=np.add(pxp_ar,pxp_fin) #summation of each i's hamiltonian
    return pxp_fin.astype('int32')

