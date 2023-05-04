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

n_PXP_feed = 9
n_TI_max_feed = 14
T_start_feed = 0
T_max_feed = 400
T_step_feed = 1000
h_c_max_feed = 3
h_c_step_feed = 0.1
Start_cutoff_feed = 105
End_cutoff_feed = 235

#data_check_Ave=np.load('PXP_10_True_X_i_Gammas_cutoff_105_235/Gamma_ave_True_X_i_10_10_2.0_cutoff_105_235.npy')
#print(data_check_Ave)

def Plot_vs_h_c_diff_n_TI_condifence():
    '''
    Plot for different TI numbers, a normalized damping graph vs Coupling strength h_c
    :return: a plot of plots
    '''
    n_PXP= n_PXP_feed
    T_start = T_start_feed
    T_max = T_max_feed
    T_step = T_step_feed
    Start_cutoff = Start_cutoff_feed
    End_cutoff = End_cutoff_feed
    h_c_max = h_c_max_feed
    h_c_step = h_c_step_feed
    n_TI_start = n_PXP
    n_TI_max = n_TI_max_feed
    n_TI= np.linspace(n_TI_start,n_TI_max,n_TI_max - n_TI_start +1,endpoint=True)
    h_c= np.linspace(0, h_c_max, int(np.round(((h_c_max)/h_c_step),1)) +1,endpoint=True)
    data_ave= np.empty((len(h_c)))
    data_errors= np.empty((2,(len(h_c))))
    cmap = cm.get_cmap('plasma',int(n_TI_max-5)) #length of colormap should be about x2 of the number of plots
    for j in np.nditer(n_TI):
        for i in np.nditer(h_c):
            data_ave[int(np.round(i,1)*10)]=np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_ave_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP,Start_cutoff,End_cutoff,n_PXP,int(j),np.round(i,1),Start_cutoff,End_cutoff))
            #print(data_ave)
            data_errors[:,int(np.round(i,1)*10)]=np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_errors_confidence_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP,Start_cutoff,End_cutoff,n_PXP,int(j),np.round(i,1),Start_cutoff,End_cutoff))
            #print(data_errors)
        data_errors_fin_0 = data_ave - data_errors[0, :]  # DATA ERRORS NEED TO BE +/- from Null
        data_errors_fin_1 = data_errors[1, :] - data_ave  # DATA ERRORS NEED TO BE +/- from Null
        data_errors_fin = np.empty((2, len(data_errors_fin_0)))
        data_errors_fin[0, :] = data_errors_fin_0/data_ave[0] #Scaled errors as to gamma/gamma_0
        data_errors_fin[1, :] = data_errors_fin_1/data_ave[0] #Scaled errors as to gamma/gamma_0
        scaled_data_ave= data_ave/data_ave[0] #Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin[:, :], color=cmap.colors[int(j-n_TI[0])], marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3, label= '{} TI atoms'.format(int(j)))
        #plt.fill_between(h_c[:], scaled_data_ave[:] - data_errors_fin[0, :], scaled_data_ave[:] + data_errors_fin[1, :])
    plt.title(r'$Z_i$ coupling normalized damping for {} PXP atoms vs $h_c$, cutoff {}-{}'.format(n_PXP,Start_cutoff,End_cutoff))
    plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
    plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
    plt.legend()
    plt.savefig('Figures/True_X_i_Osc/Z_i_damping_{}_PXP_vs_h_c_{}_confi_err_cutoff_{}_{}.png'.format(n_PXP,h_c_max,Start_cutoff,End_cutoff))
    return plt.show()
#Plot_vs_h_c_diff_n_TI_condifence()

def Plot_vs_h_c_diff_n_TI_std():
    '''
    Plot for different TI numbers, a normalized damping graph vs Coupling strength h_c
    :return: a plot of plots
    '''
    n_PXP= n_PXP_feed
    T_start = T_start_feed
    T_max = T_max_feed
    T_step = T_step_feed
    Start_cutoff = Start_cutoff_feed #implement DONT THINK IT IS HERE!
    End_cutoff = End_cutoff_feed #implement DONT THINK IT IS HERE!
    h_c_max = h_c_max_feed
    h_c_step = h_c_step_feed
    n_TI_start = n_PXP
    n_TI_max = n_TI_max_feed
    n_TI=np.linspace(n_TI_start,n_TI_max,n_TI_max - n_TI_start +1,endpoint=True)
    h_c=np.linspace(0, h_c_max, int(np.round(((h_c_max)/h_c_step),1)) +1,endpoint=True)
    data_ave= np.empty((len(h_c)))
    data_errors= np.empty((len(h_c)))
    cmap = cm.get_cmap('plasma',int(n_TI_max-5)) #length of colormap should be about x2 of the number of plots
    for j in np.nditer(n_TI):
        for i in np.nditer(h_c):
            data_ave[int(np.round(i,1)*10)]=np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_ave_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP,Start_cutoff,End_cutoff,n_PXP,int(j),np.round(i,1),Start_cutoff,End_cutoff))
            #print(data_ave)
            data_errors[int(np.round(i,1)*10)]=np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_errors_std_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP,Start_cutoff,End_cutoff,n_PXP,int(j),np.round(i,1),Start_cutoff,End_cutoff))
            #print(data_errors)
        data_errors_fin = data_errors/data_ave[0] #Scaled errors as to gamma/gamma_0
        #print('data_errors',data_errors)
        scaled_data_ave= data_ave/data_ave[0] #Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin, color=cmap.colors[int(j-n_TI[0])], marker='s',markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3, label= '{} TI atoms'.format(int(j)))
        plt.fill_between(h_c[:], data_errors_fin, data_errors_fin)
        print(data_ave[11])
    plt.title(r'$Z_i$ coupling normalized damping for {} PXP atoms vs $h_c$, cutoff {}-{}'.format(n_PXP,Start_cutoff,End_cutoff))
    plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
    plt.xlabel(r'Coupling Strength $h_c$',fontsize=12)
    plt.legend()
    plt.savefig('Figures/True_X_i_Osc/Z_i_damping_{}_PXP_vs_h_c_{}_std_err_cutoff_{}_{}.png'.format(n_PXP,h_c_max,Start_cutoff,End_cutoff))
    return plt.show()
#Plot_vs_h_c_diff_n_TI_std()

def Plot_vs_h_c_diff_n_TI_Rel_Err_new():
    '''
    Plot for different TI numbers, a normalized damping graph vs Coupling strength h_c with error from pcov
    :return: a plot of plots with error from pcov
    '''
    n_PXP = n_PXP_feed
    T_start = T_start_feed
    T_max = T_max_feed
    T_step = T_step_feed
    Start_cutoff = Start_cutoff_feed  # implement DONT THINK IT IS HERE!
    End_cutoff = End_cutoff_feed  # implement DONT THINK IT IS HERE!
    h_c_max = h_c_max_feed
    h_c_step = h_c_step_feed
    n_TI_start = n_PXP
    n_TI_max = n_TI_max_feed
    n_TI = np.linspace(n_TI_start, n_TI_max, n_TI_max - n_TI_start + 1, endpoint=True)
    h_c = np.linspace(0, h_c_max, int(np.round(((h_c_max) / h_c_step), 1)) + 1, endpoint=True)
    data_ave = np.empty((len(h_c)))
    data_errors = np.empty((len(h_c)))
    cmap = cm.get_cmap('plasma', int(n_TI_max - 5))  # length of colormap should be about x2 of the number of plots
    for j in np.nditer(n_TI):
        # for i in np.nditer(h_c): # X_i CASE
        #     data_ave[int(np.round(i, 1) * 10)] = np.load('PXP_{}_True_X_i_Gammas_cutoff_{}_{}/Gamma_ave_True_X_i_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP, Start_cutoff, End_cutoff,n_PXP, int(j), np.round(i, 1),Start_cutoff, End_cutoff))
        #     # print(data_ave)
        #     data_errors[int(np.round(i, 1) * 10)] = np.load('PXP_{}_True_X_i_Gammas_cutoff_{}_{}/Gamma_error_ave_True_X_i_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP, Start_cutoff,End_cutoff, n_PXP,int(j), np.round(i, 1),Start_cutoff,End_cutoff))
        #     # print(data_errors)
        for i in np.nditer(h_c): # Z_i CASE
            data_ave[int(np.round(i, 1) * 10)] = np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_ave_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP, Start_cutoff, End_cutoff,n_PXP, int(j), np.round(i, 1),Start_cutoff, End_cutoff))
            # print(data_ave)
            data_errors[int(np.round(i, 1) * 10)] = np.load('PXP_{}_Gammas_cutoff_{}_{}/Gamma_error_ave_{}_{}_{}_cutoff_{}_{}.npy'.format(n_PXP, Start_cutoff,End_cutoff, n_PXP,int(j), np.round(i, 1),Start_cutoff,End_cutoff))
            # print(data_errors)
        data_errors_fin = data_errors / (data_ave[0])  # Scaled errors as to gamma/gamma_0 - STD/2
        # print('data_errors',data_errors)
        scaled_data_ave = data_ave / data_ave[0]  # Scaled errors as to gamma/gamma_0
        plt.errorbar(h_c[:], scaled_data_ave[:], yerr=data_errors_fin, color=cmap.colors[int(j - n_TI[0])], marker='s',
                     markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3,
                     label='{} TI atoms'.format(int(j)))
        plt.fill_between(h_c[:], data_errors_fin, data_errors_fin)
        print(data_ave[11])
    plt.title(r'$Z_i$ coupling normalized damping for {} PXP atoms vs $h_c$, cutoff {}-{}'.format(n_PXP, Start_cutoff,End_cutoff))
    #plt.title(r'$X_i$ coupling normalized damping for {} PXP atoms vs $h_c$, cutoff {}-{}'.format(n_PXP, Start_cutoff,End_cutoff))
    plt.ylabel(r' Normalized Damping $\frac{\gamma}{\gamma_0}$', fontsize=10)
    plt.xlabel(r'Coupling Strength $h_c$', fontsize=12)
    plt.legend()
    #plt.savefig('Figures/True_X_i_Osc/Z_i_damping_{}_PXP_vs_h_c_{}_Rel_err_new_{}_{}.png'.format(n_PXP, h_c_max,Start_cutoff,End_cutoff)) #Z_i case
    #plt.savefig('Figures/True_X_i_Osc/X_i_damping_{}_PXP_vs_h_c_{}_Rel_err_new_{}_{}.png'.format(n_PXP, h_c_max,Start_cutoff,End_cutoff)) #X_i case
    return plt.show()
Plot_vs_h_c_diff_n_TI_Rel_Err_new()
