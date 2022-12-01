import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from Cluster_Sparse_Osc_Para import *
from PXP_E_B_E_Sparse import *
import numpy.linalg as la

def Cluster_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2):
    '''
    Returns <Neel|O_z(t)|Neel> values and corresponding time values, working with EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI Atoms
    :param J: TI ising term strength
    :param Initialstate:  Initial Vector state we would like to propagate
    :param h_x: longtitudinal term strength (TI)
    :param h_z: transverse term strength
    :param h_c: coupling term strength
    :param T_start: Start Time of propagation
    :param T_max: Max Time of propagation
    :param T_step: time step (division)
    :param h_imp: impurity (TI) strength
    :param m: impurity site
    :return: 2 vectors -  <NeelxHaar|O_z(t)|NeelxHaar> values and corresponding time values??????
    '''
    O_z_PXP = O_z_PXP_Entry_Sparse(n_PXP, PXP_Subspace_Algo)
    O_z_Full = sp.kron(O_z_PXP,sp.eye(2**n_TI))
    Propagated_ket = spla.expm_multiply(-1j*PXP_TI_coupled_Sparse(n_PXP, n_TI, J, h_x, h_z, h_c, h_imp, m),Initialstate ,
                                        start= T_start , stop=T_max ,num=T_step,endpoint = True)
    Propagated_ket_fin= np.transpose(Propagated_ket)
    Propagated_bra_fin = np.conjugate(Propagated_ket)
    Sandwich = np.diag(Propagated_bra_fin @ O_z_Full @ Propagated_ket_fin)
    return Sandwich.round(4).astype('float')

def Run_Cluster_Averaged_Sparse_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step):
    '''
    Runs time AVERAGE propagation plotter in EBE sparse method
    :param n_PXP: No. of PXP atoms
    :param n_TI: No. of TI atoms
    :param Initialstate: NeelHaar state usually
    :param J: Ising term strength
    :param h_x: longtitudinal field strength
    :param h_z: Traverse field strength
    :param h_c: coupling strength
    :param T_start: start time
    :param T_max: end time
    :param T_step: time division
    :return: Plot of Time propagation
    '''
    Initialstate = Neel_EBE_Haar(n_PXP,n_TI)
    J = 1
    h_x = np.sin(0.485 * np.pi)
    h_z = np.cos(0.485 * np.pi)
    Sandwich = Cluster_Sparse_Time_prop(n_PXP, n_TI, Initialstate, J, h_x, h_z, h_c, T_start, T_max, T_step, h_imp=0, m=2)
    np.save('Sparse_time_propagation_sample_{}.npy'.format(seed), Sandwich)
    return

#Run_Cluster_Averaged_Sparse_Time_prop(n_PXP, n_TI, h_c ,T_start, T_max, T_step)

def Sparse_time_combine(seed_max):
    '''
    averages over Sparse time realizations
    :param seed_max: number of realizations
    :return: saves average
    '''
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    data = np.empty((seed_max,len(Time)))
    for j in range(1,seed_max):
        data[j,:]= np.load('Sparse_time_propagation_sample_{}.npy'.format(j)) #creates
    np.save('Sparse_time_propagation_combine.npy', data)
#Sparse_time_combine(seed_max)

def Sparse_time_ave():
    '''
    averages over Sparse time realizations
    :return: saves average
    '''
    data= np.load('Sparse_time_propagation_combine.npy')
    data_ave = np.mean(data,axis=0)
    np.save('Sparse_time_propagation_ave.npy', data_ave)
#Sparse_time_ave(seed_max)


def Bootstrap(Sample_no):
    '''
    Bootstrapping of time propagation samples
    :return: 95% confidence interval upper and lower bounds for each of time steps' average of random samples
    '''
    Time = np.linspace(T_start, T_max, T_step, endpoint=True)
    lower_upper = np.empty((2,len(Time)))
    data = np.load('Sparse_time_propagation_combine.npy')
    for i in range(0,len(Time)):
        sample = np.random.choice(data[:,i],(seed_max, Sample_no), replace=True) # creates [(seed_max No.) x n] matrix of randomly sampled arrays (with return) from the original
        sample_ave = np.mean(sample, axis=0)  # vector of averages sampled from one row of propagation data (random)
        lower_mean = np.quantile(sample_ave, 0.025)
        upper_mean = np.quantile(sample_ave, 0.975)
        lower_upper[0,i] = lower_mean
        lower_upper[1,i] = upper_mean
    np.save('Sparse_time_propagation_errors.npy',lower_upper)
#Bootstrap(seed_max)

Sparse_time_ave()
Bootstrap(Sample_no)
