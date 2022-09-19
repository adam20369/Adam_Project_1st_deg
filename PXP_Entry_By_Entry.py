from PXP_TI_Bath import *
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations

def basis(n): # builds a matrix for the basis of all excitations (all the possible excitations for an n atoms chain)- NOT SUBSPACE
    vec= np.zeros(n)
    vec[0]=1
    permute= np.vstack((np.zeros(n), list(multiset_permutations(vec.astype('int')))))
    for i in range(0,n-1):
        vec[i+1]=1
        permute= np.vstack((permute,list(multiset_permutations(vec.astype('int')))))
    # print((np.fliplr(permute)))
    return np.fliplr(permute)

PXP= np.array(())

def PXP(n):
    Base = basis(n)
    Ham= np.zeros((Base.shape[0], Base.shape[0]))
    for j in range(0, Base.shape[0]):
        for i in range(0,n):
            if Base[j, i]==1 and Base[j, i - 1]==0 and Base[j, i + 1]==0:
                vecj= Base[j].copy()
                vecj[i]=0
                Ham[np.argwhere((Base == vecj).all(axis=1))[0],j]= 1
            if Base[j, i]==0 and Base[j, i - 1]==0 and Base[j, i + 1]==0:
                vecj=Base[j].copy()
                vecj[i]=1
            vecj
#      Ham[masheu,masheu]=1
#      if Base[j, i] and Base[j + 1, i]: