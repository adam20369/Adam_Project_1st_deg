import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from Coupling_To_Bath import *
from PXP_E_B_E_Sparse import *
from Entanglement_Entropy import *
import numpy.linalg as la
import scipy.linalg as scla
from time import time
from scipy import integrate
from sympy.utilities.iterables import multiset_permutations
import scipy.sparse as sp
from scipy.special import comb



def Entanglement_entropy_calc_PXP_TI(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    calculating entanglement entropy of each eigenstate from singular values -sigma^2ln(sigma) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat, eval = Evec_SVD_PXP_TI(n_PXP, n_TI, np.round(h_c,4))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    plt.scatter(eval, np.round(Entanglement_entropy_vec,32))
    plt.title('Entanglement vs Energy for {} PXP & {} TI atoms, $h_c$ {} '.format(n_PXP, n_TI, np.round(h_c,4)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return plt.show()

def EE_Avg_Std_plot(n_PXP, n_TI,h_c_start, h_c_max, interval_width=0.5):
    '''
    Plots average and Standard deviation of entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of average entanglement and Standard deviation vs h_c
    '''
    h_c = np.arange(h_c_start, h_c_max + 0.1, 0.1)
    avg = np.zeros((np.size(h_c)))
    std = avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i, 1) * 10 - h_c_start * 10)], std[int(np.round(i, 1) * 10 - h_c_start * 10)] = Entanglement_entropy_avg_std(n_PXP, n_TI, i, interval_width)
    plt.plot(h_c[:], avg[:], linestyle='-',color='b')
    #plt.errorbar(h_c[:], avg[:], yerr=std[:],marker='s', markersize=2, linestyle='-', barsabove=True, capsize=3, capthick=3)
    plt.fill_between(h_c[:], avg[:] - std[:], avg[:]+std[:], alpha=0.4)
    plt.title('Average Entanglement vs $h_c$ for {} PXP & {} TI atoms'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Average Entanglement Entropy')
    plt.show()


def EE_PXP_TI_Cluster_Plot(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    plots entanglement entropy of each eigenstate from singular values -sigma^2ln(sigma) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Entanglement_h_c_{}.npy'.format(h_c)))
    eval = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}'.format(n_PXP,n_TI),'Eval_h_c_{}.npy'.format(h_c)))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    plt.scatter(eval, np.round(Entanglement_entropy_vec,32))
    plt.title('Entanglement vs Energy for {} PXP & {} TI atoms, $h_c$ {} '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return plt.show()


def EE_Avg_Std_Cluster_plot(n_PXP, n_TI, h_c_start, h_c_max, interval_width):
    '''
    Plots average and Standard deviation of entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of average entanglement and Standard deviation vs h_c
    '''
    h_c = np.arange(h_c_start, h_c_max + 0.1, 0.1)
    avg = np.zeros((np.size(h_c)))
    std = avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i, 1) * 10 - h_c_start * 10)] = np.load(
            os.getcwd() + os.path.join('/EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP, n_TI, interval_width),
                                       'Average_h_c_{}_width_{}.npy'.format(np.round(i, 1), interval_width)))
        std[int(np.round(i, 1) * 10 - h_c_start * 10)] = np.load(
            os.getcwd() + os.path.join('/EE_PXP_{}_TI_{}/AVG_STD_width_{}'.format(n_PXP, n_TI, interval_width),
                                       'STD_h_c_{}_width_{}.npy'.format(np.round(i, 1), interval_width)))
    plt.plot(h_c[:], avg[:], linestyle='-', color='b')
    plt.fill_between(h_c[:], avg[:] - std[:], avg[:] + std[:], alpha=0.4)
    plt.title('Average Entanglement vs $h_c$ for {} PXP & {} TI atoms'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Average Entanglement Entropy')
    plt.savefig('Figures/Entanglement_Entropy/Average_EE_filled_{}_PXP_{}_TI_{}_max_h_c_{}_width.png'.format(n_PXP, n_TI,h_c_max,interval_width))
    plt.show()



def Entanglement_entropy_calc_True_X_i_PXP_TI(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    ''' XX coupling!
    calculating entanglement entropy of each eigenstate from singular values -sigma^2ln(sigma) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat, eval = Evec_SVD_True_X_i_PXP_TI(n_PXP, n_TI, np.round(h_c,5))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    plt.scatter(eval, np.round(Entanglement_entropy_vec,32))
    plt.title('Entanglement vs Energy for {} PXP & {} TI atoms, $h_c$ {} XX coupl '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return plt.show()


def Entanglement_entropy_True_X_i_avg_std(n_PXP, n_TI, h_c, interval_width): #8 PXP 10 TI max calc
    ''' XX coupling!
    Calculating entanglement entropy average and standard deviation for some energy interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: average, standard deviation (two scalars)
    '''
    SVD_vec_mat, eval = Evec_SVD_True_X_i_PXP_TI(n_PXP, n_TI, np.round(h_c,5))
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    eval_interval_arg_min_left= np.min(np.argwhere(eval>(-h_c-interval_width)))
    eval_interval_arg_max_left=np.min(np.argwhere(eval>(-h_c+interval_width)))
    eval_interval_arg_min_right= np.min(np.argwhere(eval>(h_c-interval_width)))
    eval_interval_arg_max_right= np.min(np.argwhere(eval>(h_c+interval_width)))
    # print('coupling=',h_c,'minimum left interval of eval', eval[eval_interval_arg_min_left])
    # print('coupling=',h_c,'maximum left interval of eval', eval[eval_interval_arg_max_left])
    # print('coupling=',h_c,'minimum right interval of eval', eval[eval_interval_arg_min_right])
    # print('coupling=',h_c,'maximum right interval of eval', eval[eval_interval_arg_max_right])
    average_left = np.mean(Entanglement_entropy_vec[eval_interval_arg_min_left:eval_interval_arg_max_left])
    average_right = np.mean(Entanglement_entropy_vec[eval_interval_arg_min_right:eval_interval_arg_max_right])
    average= (average_left+average_right)/2
    std_left = np.std(Entanglement_entropy_vec[eval_interval_arg_min_left:eval_interval_arg_max_left])
    std_right = np.std(Entanglement_entropy_vec[eval_interval_arg_min_right:eval_interval_arg_max_right])
    std= (std_left+std_right)/2
    return np.round(average,5), np.round(std,5)


def EE_PXP_TI_True_X_i_Cluster_Plot(n_PXP, n_TI, h_c): #TOP NUMBER IS 8x10 - uses 25 giga (8x11 uses 100 Giga)
    '''
    plots entanglement entropy of each eigenstate XX Coupling from singular values -sigma^2ln(sigma) for splitting at PXP - TI boundary
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :return: Plot of Entanglement entropy vs energy for each eigenstate & eigenenergy
    '''
    SVD_vec_mat = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}_True_X_i'.format(n_PXP,n_TI),'Entanglement_h_c_{}_True_X_i.npy'.format(h_c)))
    eval = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}_True_X_i'.format(n_PXP,n_TI),'Eval_h_c_{}_True_X_i.npy'.format(h_c)))
    print(eval)
    Entanglement_entropy_vec = -np.sum(2*(SVD_vec_mat**2)*np.nan_to_num(np.log(SVD_vec_mat)),axis=1)
    print(Entanglement_entropy_vec)

    plt.scatter(eval, np.round(Entanglement_entropy_vec,32))
    plt.title('Entanglement vs Energy for {} PXP & {} TI atoms, XX coupling $h_c$ {} '.format(n_PXP, n_TI, np.round(h_c,5)))
    plt.xlabel('Energy')
    plt.ylabel('Entanglement Entropy')
    #plt.savefig("Figures/Entanglement_Entropy/Entropy_for_PXP_{}_TI_{}_Coup_{}.png".format(n_PXP, n_TI, np.round(h_c,5)))
    return plt.show()

def EE_True_X_i_Avg_Std_Cluster_plot(n_PXP, n_TI,h_c_start, h_c_max,interval_width):
    '''
    Plots average and Standard deviation of entanglement for given interval
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param h_c: coupling strength
    :param interval_width: energy interval width +-delta
    :param interval_center = h_c!!!!! (moves with the coupling exactly - weird)
    :return: plot of average entanglement and Standard deviation vs h_c
    '''
    h_c = np.arange(h_c_start, h_c_max + 0.1, 0.1)
    avg = np.zeros((np.size(h_c)))
    std = avg.copy()
    for i in np.nditer(h_c):
        avg[int(np.round(i, 1) * 10 - h_c_start * 10)]  = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}_True_X_i/AVG_STD_width_{}_True_X_i'.format(n_PXP,n_TI,interval_width),'Average_h_c_{}_width_{}_True_X_i.npy'.format(np.round(i,1),interval_width)))
        std[int(np.round(i, 1) * 10 - h_c_start * 10)] = np.load(os.getcwd()+os.path.join('/EE_PXP_{}_TI_{}_True_X_i/AVG_STD_width_{}_True_X_i'.format(n_PXP,n_TI,interval_width),'STD_h_c_{}_width_{}_True_X_i.npy'.format(np.round(i,1),interval_width)))
    plt.plot(h_c[:], avg[:], linestyle='-',color='b')
    plt.fill_between(h_c[:], avg[:] - std[:], avg[:]+std[:], alpha=0.4)
    plt.title('Average Entanglement vs $h_c$ for {} PXP & {} TI atoms XX coupl'.format(n_PXP, n_TI))
    plt.xlabel('Coupling Strength $h_c$')
    plt.ylabel('Average Entanglement Entropy')
    plt.savefig('Figures/Entanglement_Entropy/Average_EE_True_X_i_filled_{}_PXP_{}_TI_{}_max_h_c_{}_width.png'.format(n_PXP,n_TI,h_c_max,interval_width))
    plt.show()
