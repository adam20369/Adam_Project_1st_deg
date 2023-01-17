import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from PXP_Entry_By_Entry import *
from PXP_E_B_E_Sparse import *
import O_z_Oscillations as Ozosc
import numpy.linalg as la
from time import time
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import comb
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler

def Plot_vs_h_c_diff_n_TI_condifence():
    '''
    Plot for different TI numbers, a normalized damping graph vs Coupling strength h_c
    :return: a plot of plots
    '''
    n_PXP=9
    T_start = 0
    T_max = 400
    T_step = 1000
    Start_cutoff= 8 #implement
    End_cutoff= 1000 #implement
    h_c_max=1.3
    h_c_step=0.1
    n_TI_start=n_PXP
    n_TI_max=14
    n_TI=np.linspace(n_TI_start,n_TI_max,n_TI_max - n_TI_start +1,endpoint=True)
    h_c=np.linspace(0, h_c_max, int(np.round(((h_c_max)/h_c_step),1)) +1,endpoint=True)
    data_ave= np.empty((len(h_c)))
    data_errors= np.empty((2,(len(h_c))))
    cmap = cm.get_cmap('plasma',int(n_TI_max-5)) #length of colormap should be about x2 of the number of plots
    for j in np.nditer(n_TI):
        for i in np.nditer(h_c):
            data_ave[int(i*10)]=np.load('PXP_{}_Gammas/Gamma_ave_{}_{}_{}.npy'.format(n_PXP,n_PXP,int(j),np.round(i,2)))
            #print(data_ave)
            data_errors[:,int(i*10)]=np.load('PXP_{}_Gammas/Gamma_errors_confidence_{}_{}_{}.npy'.format(n_PXP,n_PXP,int(j),np.round(i,2)))
            #print(data_errors)
        data_errors_fin_0 = data_ave - data_errors[0, :]  # DATA ERRORS NEED TO BE +/- from Null
        data_errors_fin_1 = data_errors[1, :] - data_ave  # DATA ERRORS NEED TO BE +/- from Null
        data_errors_fin = np.empty((2, len(data_errors_fin_0)))
        data_errors_fin[0, :] = data_errors_fin_0/data_ave[0] #Scaled errors as to gamma/gamma_0
        data_errors_fin[1, :] = data_errors_fin_1/data_ave[0] #Scaled errors as to gamma/gamma_0
        scaled_data_ave= data_ave/data_ave[0] #Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin[:, :], color=cmap.colors[int(j-n_TI[0])], marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3, label= '{} TI atoms'.format(int(j)))
        #plt.fill_between(h_c[:], scaled_data_ave[:] - data_errors_fin[0, :], scaled_data_ave[:] + data_errors_fin[1, :])
    plt.title(r'Normalized damping for {} PXP atoms vs coupling strength'.format(n_PXP))
    plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
    plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
    plt.legend()
    plt.savefig('Figures/Norm_damping_{}_PXP_vs_h_c_{}_confi_err.png'.format(n_PXP,h_c_max))
    return plt.show()
Plot_vs_h_c_diff_n_TI_condifence()

def Plot_vs_h_c_diff_n_TI_std():
    '''
    Plot for different TI numbers, a normalized damping graph vs Coupling strength h_c
    :return: a plot of plots
    '''
    n_PXP=9
    T_start = 0
    T_max = 400
    T_step = 1000
    h_c_max=1.3
    h_c_step=0.1
    n_TI_start= n_PXP
    n_TI_max=14
    n_TI=np.linspace(n_TI_start,n_TI_max,n_TI_max - n_TI_start +1,endpoint=True)
    h_c=np.linspace(0, h_c_max, int(np.round(((h_c_max)/h_c_step),1)) +1,endpoint=True)
    data_ave= np.empty((len(h_c)))
    data_errors= np.empty((len(h_c)))
    cmap = cm.get_cmap('plasma',int(n_TI_max-5)) #length of colormap should be about x2 of the number of plots
    for j in np.nditer(n_TI):
        for i in np.nditer(h_c):
            data_ave[int(i*10)]=np.load('PXP_{}_Gammas/Gamma_ave_{}_{}_{}.npy'.format(n_PXP,n_PXP,int(j),np.round(i,2)))
            #print(data_ave)
            data_errors[int(i*10)]=np.load('PXP_{}_Gammas/Gamma_errors_std_{}_{}_{}.npy'.format(n_PXP,n_PXP,int(j),np.round(i,2)))
            #print(data_errors)
        data_errors_fin = data_errors/data_ave[0] #Scaled errors as to gamma/gamma_0
        #print('data_errors',data_errors)
        scaled_data_ave= data_ave/data_ave[0] #Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin, color=cmap.colors[int(j-n_TI[0])], marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3, label= '{} TI atoms'.format(int(j)))
        plt.fill_between(h_c[:], data_errors_fin, data_errors_fin)
    plt.title(r'Normalized damping for {} PXP atoms vs coupling strength'.format(n_PXP))
    plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
    plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
    plt.legend()
    plt.savefig('Figures/Norm_damping_{}_PXP_vs_h_c_{}_std_err.png'.format(n_PXP,h_c_max))
    return plt.show()
Plot_vs_h_c_diff_n_TI_std()



# def Plot_vs_h_c_confidence():
#     '''
#      Plot normalized damping graph vs Coupling strength h_c for a fixed PXP and TI number
#     :return:
#     '''
#     n_PXP=10
#     n_TI=10
#     T_start = 0
#     T_max = 200
#     T_step = 2000
#     h_c_max=1
#     h_c=np.linspace(0, h_c_max, 11,endpoint=True)
#     data_ave= np.empty((len(h_c)))
#     data_errors= np.empty((2,(len(h_c))))
#     for i in np.nditer(h_c):
#             data_ave[int(i*10)]=np.load('PXP_{}_Gammas/Gamma_ave_{}_{}_{}.npy'.format(n_PXP,n_PXP,n_TI,np.round(i,2)))
#             print(data_ave)
#             data_errors[:,int(i*10)]=np.load('PXP_{}_Gammas/Gamma_errors_{}_{}_{}.npy'.format(n_PXP,n_PXP,n_TI,np.round(i,2)))
#             print(data_errors)
#     data_errors_fin_0 = data_ave - data_errors[0, :]  # DATA ERRORS NEED TO BE +/- from Null
#     data_errors_fin_1 = data_errors[1, :] - data_ave  # DATA ERRORS NEED TO BE +/- from Null
#     data_errors_fin = np.empty((2, len(data_errors_fin_0)))
#     data_errors_fin[0, :] = data_errors_fin_0/data_ave[0] #Scaled errors as to gamma/gamma_0
#     data_errors_fin[1, :] = data_errors_fin_1/data_ave[0] #Scaled errors as to gamma/gamma_0
#     scaled_data_ave= data_ave/data_ave[0] #Scaled errors as to gamma/gamma_0
#     plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin[:, :], color='b', marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3)
#     plt.title(r'Normalized damping for {} PXP atoms, {} TI atoms vs $h_c$ strength'.format(n_PXP,n_TI))
#     plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
#     plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
#     #plt.fill_between(h_c[:],scaled_data_ave[:]-data_errors_fin[0, :],scaled_data_ave[:]+data_errors_fin[1, :])
#     return plt.show()
# #Plot_vs_h_c_confidence()

# def print_gamma_combine():
#     n_PXP=10
#     T_start = 0
#     T_max = 400
#     T_step = 1000
#     h_c_max=1
#     h_c_step=0.1
#     n_TI_max=13
#     n_TI_start=10
#     n_TI=np.linspace(n_TI_start,n_TI_max,n_TI_max - n_TI_start +1,endpoint=True)
#     h_c=np.linspace(0, h_c_max, int((h_c_max)/h_c_step) +1,endpoint=True)
#     for j in np.nditer(n_TI):
#         for i in np.nditer(h_c):
