import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from matplotlib.lines import Line2D
from PXP_TI_Bath import *

def RunTimeProp4(n_tot, n_Array, Coupl=Z_i, h_x=1, h_z=1, T_max=20): #time propagation of 4 different PXP atom sizes (total N conserved)
    markers = np.array(('s', '^', 'o', 'd'))
    colors = np.array(('b','r','y','k'))
    n_start= n_Array[0]
    for n in n_Array:
        H = PXPBathHam(n_tot, n, Coupl, h_x, h_z)
        EV = EvecEval(H)
        Neel = Neelstate(n_tot)
        Color = colors[n-n_start]
        marker= markers[n-n_start]
        TimeProp(EV, n_tot, Neel, T_max, Color, marker)
    custom_lines = [Line2D([0], [0], color=colors[0], marker=markers[0]),
                    Line2D([0], [0], color=colors[1], marker=markers[1]),
                    Line2D([0], [0], color=colors[2], marker=markers[2]),
                    Line2D([0], [0], color=colors[3], marker=markers[3])] #LEGEND DEFINITIONS
    plt.legend(custom_lines, ['{} atoms'.format(n_Array[0]), '{} atoms'.format(n_Array[1]), '{} atoms'.format(n_Array[2]),'{} atoms'.format(n_Array[3])])
    plt.xlabel('$t$')
    plt.ylabel(r'$|\langle\mathbb{Z}_{2}|\mathbb{Z}_{2}(t)\rangle|^{2}$')
    plt.savefig('new fidelity_12atoms.pdf')
    plt.title('Quantum Fidelity of {}-atom Neel State'.format(n_tot))
# RunTimeProp4(10,np.arange(5,9,1), Coupl=Z_i, h_x=np.sin(0.485*np.pi), h_z=(0.485*np.pi), T_max=20)
# plt.show()