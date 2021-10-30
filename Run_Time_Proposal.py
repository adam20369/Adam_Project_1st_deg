import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from PXP_TI_Bath import *


def RunTimeProp4(n_tot, n_Array, Coupl=Z_i, h_x=1, h_z=1, T_max=20): #time propagation of 4 different PXP atom sizes (total N conserved)
    markers = itertools.cycle(('s', '^', 'X', 'o', '*', 'h', 'D'))
    for n in n_Array:
        H = PXPBathHam(n_tot, n, Coupl, h_x, h_z)
        EV = EvecEval(H)
        Neel = Neelstate(n_tot)
        Color = np.array((np.random.rand(), np.random.rand(), np.random.rand()))
        marker= next(markers)
        TimeProp(EV, n_tot, Neel, T_max, Color, marker)
    plt.legend(['{} atoms'.format(n_Array[0]),'{} atoms'.format(n_Array[1]),'{} atoms'.format(n_Array[2]),'{} atoms'.format(n_Array[3])])
    plt.xlabel('$t$')
    plt.ylabel(r'$|\langle\mathbb{Z}_{2}|\mathbb{Z}_{2}(t)\rangle|^{2}$')
    plt.savefig('new fidelity_12atoms.pdf')
    plt.title('Quantum Fidelity of the Neel State vs. Time')