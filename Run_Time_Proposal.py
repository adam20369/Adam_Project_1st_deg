import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import itertools
import timeit
from scipy import integrate
from matplotlib.lines import Line2D
from PXP_TI_Bath import *
from Coupling_To_Bath import *

def RunTimeProp4(n_tot, n_Array, Coupl=Z_i, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_c=1, T_max=20, T_int=0.05):
    """
    time propagation of 4 different PXP atom sizes (total N conserved)
    :param n_tot:
    :param n_Array:
    :param Coupl:
    :param h_x:
    :param h_z:
    :param h_c:
    :param T_max:
    :return:
    """
    markers = np.array(('s', '^', 'o', 'd'))
    colors = np.array(('b','r','y','k'))
    n_start = n_Array[0]
    for n in n_Array:
        H = PXPBathHam(n_tot, n, Coupl, h_x, h_z, h_c)
        EV = EvecEval(H)
        NeelHaarstate = NeelHaar(n_tot, n_Array)
        Color = colors[n-n_start]
        marker = markers[n-n_start]
        TimeProp(EV, n_tot, NeelHaarstate, T_max, T_int, Color, marker)
    custom_lines = [Line2D([0], [0], color=colors[0], marker=markers[0]),
                    Line2D([0], [0], color=colors[1], marker=markers[1]),
                    Line2D([0], [0], color=colors[2], marker=markers[2]),
                    Line2D([0], [0], color=colors[3], marker=markers[3])] #LEGEND DEFINITIONS
    plt.legend(custom_lines, ['{} atoms'.format(n_Array[0]), '{} atoms'.format(n_Array[1]), '{} atoms'.format(n_Array[2]),'{} atoms'.format(n_Array[3])])
    plt.xlabel('$t$')
    plt.ylabel(r'$|\langle\mathbb{Z}_{2}|\mathbb{Z}_{2}(t)\rangle|^{2}$')
    #plt.savefig('new fidelity_12atoms.pdf')
    plt.title('Quantum Fidelity of {}-atom Neel State with coupling strength {}'.format(n_tot,h_c))
# RunTimeProp4(10,np.arange(5,9,1), Coupl=Z_i, h_x=np.sin(0.485*np.pi), h_z=(0.485*np.pi), T_max=20)
# plt.show()

#taking without coupling and with coupling and seeing that neel state of ONLY the PXP model
# and seeing that it stays the same about the part of the TI model, we take an infinite temperature state (average energy state, (1/Z)*tr(H)

def RunTimeProp4new(n_totArray, n_pxp, Coupl=Z_i, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), h_c=1, T_max=20):
    """
      time propagation of 4 different TOTAL atom sizes, PXP number conserved
    :param n_totArray:
    :param n_pxp:
    :param Coupl:
    :param h_x:
    :param h_z:
    :param h_c:
    :param T_max:
    :return:
    """
    markers = np.array(('s', '^', 'o', 'd'))
    colors = np.array(('b','r','y','k'))
    n_start = n_totArray[0]
    for n_tot in n_totArray:
        H = PXPBathHam(n_tot, n_pxp, Coupl, h_x, h_z, h_c) #TODO check what happens if coupling=0
        EV = EvecEval(H)
        NeelHaarstate = NeelHaar(n_tot, n_pxp)
        Color = colors[n_tot-n_start]
        marker = markers[n_tot-n_start]
        TimeProp(EV, n_tot, NeelHaarstate, T_max, Color, marker)
    custom_lines = [Line2D([0], [0], color=colors[0], marker=markers[0]),
                    Line2D([0], [0], color=colors[1], marker=markers[1]),
                    Line2D([0], [0], color=colors[2], marker=markers[2]),
                    Line2D([0], [0], color=colors[3], marker=markers[3])] #LEGEND DEFINITIONS
    plt.legend(custom_lines, ['{} atoms'.format(n_totArray[0]), '{} atoms'.format(n_totArray[1]), '{} atoms'.format(n_totArray[2]),'{} atoms'.format(n_totArray[3])])
    plt.xlabel('$t$')
    plt.ylabel(r'$|\langle\mathbb{Z}_{2}|\mathbb{Z}_{2}(t)\rangle|^{2}$')
    #plt.savefig('new fidelity_12atoms.pdf')
    plt.title('Quantum Fidelity of {}-PXP Neel State with coupling strength {}'.format(n_pxp,h_c))

#RunTimeProp4new(np.arange(9,13,1),6 , Coupl=Z_i, h_x=np.sin(0.485*np.pi), h_z=np.cos(0.485*np.pi), T_max=20)
# plt.show()
