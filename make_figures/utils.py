import numpy as np
from tqdm import tqdm
import ot
import matplotlib.pyplot as plt
from glob import glob
import os


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_files(path, filename = 'lfp.npy'):
    
    all_paths = glob(path)
    
    result = dict()
    for path in all_paths:
        for root, dirs, files in os.walk(path):
            if len(glob(os.path.join(root, filename)))>0:

                file_path = glob(os.path.join(root,filename))[0]
                sim_name = file_path.split('/')[-2]
                print(sim_name)
                try:
                    file = np.load(file_path, allow_pickle=True)[()]
                except:
                    print('File could not be loaded')
                result[sim_name] = file
    return result
                
def find_all_fir_rates_files_sim(path,filename):
    
    result = dict()
    for root, dirs, files in os.walk(path):
        if len(glob(os.path.join(root,filename)))>0:
            
            fir_rate_file_path = glob(os.path.join(root,filename))[0]
            sim_name = fir_rate_file_path.split('/')[-2]
            fir_rate_file = np.load(fir_rate_file_path, allow_pickle=True)[()]
            result[sim_name] = fir_rate_file
    return result


def compute_dist_matrix_exp(file_list_exp, file_list_sim = None, mode = 'pairwise_exp', return_G0 = False):
    """
    This function takes in lists of either sinks or sources in the CSD from different animals, and computes the 
    Wasserstein between them
    
    Args:
        file_list (list): list of sinks of sources extracted from the CSD in different animals. Each element 
        in the list contains all sinks or all sources from a single animal

    Returns:
        distance_matrix (arr): An array containing the pairwise wasserstein distances between the sinks/sources 
                               of different animals
        M (arr): The cost matrix
        G0_all (arr): Matrix containing the movement of all sink/source elements
        x_s (arr): vector used for constructing the cost matrix
        x_t (arr): vector used for constructing the cost matrix
    """
    
    if mode == 'pairwise_exp':
        n_files = len(file_list_exp)
        distance_matrix = np.zeros((n_files, n_files))
        if return_G0:
            G0_all = np.zeros((n_files, n_files, np.shape(file_list_exp)[1]*np.shape(file_list_exp)[2], \
                              np.shape(file_list_exp)[1]*np.shape(file_list_exp)[2]))
        #G0_all = list()
        for i in tqdm(range(n_files)):
            #G0_temp = list()
            for j in range(i, n_files):
                animal_a = file_list_exp[i]
                animal_b = file_list_exp[j]
                animal_a_reshape = np.reshape(animal_a, (-1))
                animal_b_reshape = np.reshape(animal_b, (-1))

                x = np.linspace(0, 1, animal_a.shape[0], endpoint=True)
                y = np.linspace(0, 1, animal_b.shape[1], endpoint=True)
                xx, yy = np.meshgrid(x,y)

                x_s = np.array([np.reshape(xx, (-1)), np.reshape(yy, (-1))]).T
                x_t = x_s

                # Constructing a cost matrix. It denotes how much it should cost to move in space vs in time
                M = ot.dist(x_s, x_t)
                M /= M.max()

                # Calculating how much you have to move the sinks/sources of animal a to match the sinks/sources
                # of animal b
                G0 = ot.emd(animal_a_reshape, animal_b_reshape, M)

                distance_matrix[i,j] = (M * G0).sum()
                distance_matrix[j,i] = (M * G0).sum()
                
                if return_G0:
                    G0_all[i,j] = G0
                    G0_all[j,i] = G0
                
                #G0_temp.append(G0)
            #G0_all.append(G0_temp)

        if return_G0:
            return distance_matrix, M, G0_all, x_s, x_t
        else:
            return distance_matrix
    elif mode == 'pairwise_exp_trials':
        n_files = len(file_list_exp)
        distance_matrix = np.zeros((n_files, n_files))
        #G0_all = np.zeros((n_files, n_files, np.shape(file_list_exp)[1]*np.shape(file_list_exp)[2], \
        #                  np.shape(file_list_exp)[1]*np.shape(file_list_exp)[2]))
        for i in tqdm(range(n_files)):
            #G0_temp = list()
            for j in range(i, n_files):
                animal_a = file_list_exp[i]
                animal_b = file_list_exp[j]
                animal_a_reshape = np.reshape(animal_a, (-1))
                animal_b_reshape = np.reshape(animal_b, (-1))

                x = np.linspace(0, 1, animal_a.shape[0], endpoint=True)
                y = np.linspace(0, 1, animal_b.shape[1], endpoint=True)
                xx, yy = np.meshgrid(x,y)

                x_s = np.array([np.reshape(xx, (-1)), np.reshape(yy, (-1))]).T
                x_t = x_s

                # Constructing a cost matrix. It denotes how much it should cost to move in space vs in time
                M = ot.dist(x_s, x_t)
                M /= M.max()

                # Calculating how much you have to move the sinks/sources of animal a to match the sinks/sources
                # of animal b
                G0 = ot.emd(animal_a_reshape, animal_b_reshape, M)

                distance_matrix[i,j] = (M * G0).sum()
                distance_matrix[j,i] = (M * G0).sum()

                #G0_all[i,j] = G0
                #G0_all[j,i] = G0
                
                #G0_temp.append(G0)
            #G0_all.append(G0_temp)

        return distance_matrix#, M, G0_all, x_s, x_t
    else:
        n_files_exp = len(file_list_exp)
        n_files_sim = len(file_list_sim)
        distance_matrix = np.zeros((n_files_exp, n_files_sim))
        G0_all = list()
        for i in tqdm(range(n_files_exp)):
            G0_temp = list()
            for j in range(n_files_sim):
                animal = file_list_exp[i]
                sim = file_list_sim[j]
                animal_reshape = np.reshape(animal, (-1))
                sim_reshape = np.reshape(sim, (-1))

                x = np.linspace(0, 1, animal.shape[0], endpoint=True)
                y = np.linspace(0, 1, animal.shape[1], endpoint=True)
                xx, yy = np.meshgrid(x,y)

                x_s = np.array([np.reshape(xx, (-1)), np.reshape(yy, (-1))]).T
                x_t = x_s

                # Constructing a cost matrix. It denotes how much it should cost to move in space vs in time
                M = ot.dist(x_s, x_t)
                M /= M.max()

                # Calculating how much you have to move the sinks/sources of animal a to match the sinks/sources
                # of animal b
                G0 = ot.emd(animal_reshape, sim_reshape, M)

                distance_matrix[i,j] = (M * G0).sum()
                #distance_matrix[j,i] = (M * G0).sum()

                G0_temp.append(G0)
            G0_all.append(G0_temp)
        return distance_matrix, M, G0_all, x_s, x_t
    

def plot_wasserstein_result(a, b, G, xs, xt, name_a = 'Animal A', name_b = 'Animal B', mode = 'source', M = 0, thr=1.e-3, dist = -1):
    
    mx = G.max()
    if mode=='sink':
        c1 = 'Blues'
        c2 = 'Greens'
    if mode=='source':
        c1 = 'Reds'
        c2 = 'Purples'
    #dist = np.round((M*G).sum(), decimals=4)
    
    fig, ax = plt.subplots(figsize=(10, 4), ncols=3, nrows=1, sharey=True, sharex=True)
    
    colormap_range = np.max(np.abs(a))
    # Note: origin = 'lowerleft here'
    im = ax[0].imshow(a, extent=(0,1,0,1), cmap=c1, vmin=0, vmax=colormap_range)
    ax[0].grid()
    #ax[0].set_xlabel('Time from flash onset (ms)')
    ax[0].set_ylabel('Depth ($\mu$m)')
    ax[0].set_title(mode.capitalize()+'s animal '+name_a)
    #ax[0].set_xticks(np.arange(0, a.shape[1], 20))
    #ax[0].set_xticklabels(np.arange(0, a.shape[1]+20, 20))
    #ax[0].set_yticks(np.arange(0, a.shape[0], 6))
    #ax[0].set_yticklabels(-np.arange(0,1000,200))
    
    ax[0].set_xticks(np.arange(0, 1+0.2, 0.2))
    ax[0].set_xticklabels(np.arange(0, a.shape[1]+60, 20))
    ax[0].set_yticks(np.arange(0, 1+0.25, 0.25))
    ax[0].set_yticklabels(np.array([-800, -600, -400, -200, 0]))
    
    colormap_range = np.max(np.abs(b))
    # Note: origin = 'lowerleft here'
    im = ax[1].imshow(b, extent=(0,1,0,1), cmap=c2, vmin=0, vmax=colormap_range)
    ax[1].grid()
    ax[1].set_xlabel('Time from flash onset (ms)')
    ax[1].set_title(mode.capitalize()+'s animal '+name_b)

    colormap_range = np.max(np.abs(a))
    # Note: origin = 'lowerleft here'
    im = ax[2].imshow(a, extent=(0,1,0,1), cmap=c1, vmin=0, vmax=colormap_range, alpha=0.7)
    colormap_range = np.max(np.abs(b))
    im = ax[2].imshow(b, extent=(0,1,0,1), cmap=c2, vmin=0, vmax=colormap_range, alpha=0.5)

    toggle = True
    for i in tqdm(range(xs.shape[0])):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                #if toggle:
                #    label = 'Movement of \n'+mode
                #    toggle = False
                #else:
                #    label = None
                ax[2].plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], alpha=G[i, j]*0.1 / mx, color='black')#, label = label)

    ax[2].grid()
    #ax[2].set_xlabel('Time from flash onset (ms)')
    ax[2].set_title('Norm. WD between \n'+mode+'s = '+str(np.round(dist, 3)))

    return fig

def subtract_lfp_baseline_all_sims(lfp_orig, tstim_onset = 1000, contributions = False, contributions_summed = False):
        
    #TODO: Fix the stupid data organization that forces all these if-statements
    
    lfp_out = dict()

    if contributions:
        for sim_name in lfp_orig.keys():
            print(sim_name)
            lfp_dict_pops = dict()
            for pop_name in lfp_orig[sim_name].keys():
                lfp_dict = dict()
                
                lfp_temp = lfp_orig[sim_name][pop_name]['trial_avg'].T
                
                lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
            
                lfp_dict['trial_avg'] = lfp_temp.T
                
                lfp_trials_temp = []
                for itrial in range(lfp_orig[sim_name][pop_name]['all_trials'].shape[1]):
                    
                    lfp_temp = lfp_orig[sim_name][pop_name]['all_trials'][:,itrial].T
                    
                    lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
                    
                    lfp_trials_temp.append(lfp_temp.T)
                
                lfp_dict['all_trials'] = np.array(lfp_trials_temp)
                
                lfp_dict_pops[pop_name] = lfp_dict
                                    
            lfp_out[sim_name] = lfp_dict_pops
            
    elif contributions_summed:
        for sim_name in lfp_orig.keys():
            print(sim_name)
            lfp_dict_pops = dict()
            for pop_name in lfp_orig[sim_name]['all_trials'].keys():
                lfp_dict = dict()
                
                lfp_temp = lfp_orig[sim_name]['trial_avg'][pop_name].T
                
                lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
            
                lfp_dict['trial_avg'] = lfp_temp.T
                
                lfp_trials_temp = []
                for itrial in range(lfp_orig[sim_name]['all_trials'][pop_name].shape[1]):
                    
                    lfp_temp = lfp_orig[sim_name]['all_trials'][pop_name][:,itrial].T
                    
                    lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)
                    
                    lfp_trials_temp.append(lfp_temp.T)
                
                lfp_dict['all_trials'] = np.array(lfp_trials_temp)
                
                lfp_dict_pops[pop_name] = lfp_dict
                                    
            lfp_out[sim_name] = lfp_dict_pops
    else:
        for sim_name in lfp_orig.keys():
            lfp_dict = dict()
            print(sim_name)
            lfp_temp = lfp_orig[sim_name]['trial_avg'].T

            lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)

            lfp_dict['trial_avg'] = lfp_temp.T

            lfp_trials_temp = []
            for itrial in range(lfp_orig[sim_name]['all_trials'].shape[1]):
                lfp_temp = lfp_orig[sim_name]['all_trials'][:,itrial].T

                lfp_temp -= np.mean(lfp_temp[:tstim_onset], axis = 0)

                lfp_trials_temp.append(lfp_temp.T)
                
            lfp_dict['all_trials'] = np.array(lfp_trials_temp)
                
            lfp_out[sim_name] = lfp_dict
            
    return lfp_out