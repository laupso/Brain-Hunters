# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:12:01 2020

@author: opsomerl
"""

import numpy as np
import basics_steinmetz as bs
import pca_steinmetz as pcas
from matplotlib import pyplot as plt



#%% Data retrieval
alldat = bs.import_dataset()



#%% Run N iterations where NT trials are randomly selected
np.random.seed(None)

pre_stim = np.arange(25,45) # Pre-stimulus zone
npc = 5 # Number of principle components
NS = 20 # Number of iterations
NT = 92 # Number of trials to be selected within each session
NB = 250

all_pc_correct   = np.ndarray([npc, NS, NB])
all_pc_incorrect = np.ndarray([npc, NS, NB])
for i in range(NS):
    
    # Select trials
    spks, correct = pcas.collect_spks(alldat, NT, True)
    
    # Compute PCA with those trials
    spks_pre_stim = spks[:,:,pre_stim]
    
    model = pcas.fit_pca_model(spks_pre_stim, npc)
    
    pc = pcas.compute_pca(spks, model, npc)
    
    for j in range(npc):
        
        pc_j = pc[j]
        all_pc_correct[j,i,:] = pc_j[correct,:].mean(axis = 0)
        all_pc_incorrect[j,i,:] = pc_j[~correct,:].mean(axis = 0)
        
        
# Average pc across iterations
pc_correct_av  = all_pc_correct.mean(axis = 1)
pc_correct_sem = all_pc_correct.std(axis = 1) / np.sqrt(NS)
    
pc_incorrect_av  = all_pc_incorrect.mean(axis = 1)
pc_incorrect_sem = all_pc_incorrect.std(axis = 1) / np.sqrt(NS)    
    
    

#%% Plot PC
       
plt.figure(figsize= (15, 4))

time = bs.get_time(alldat[0])

for j in range(npc):
    ax = plt.subplot(1,npc,j+1)
    
#    plt.plot(time, pc_correct_av[j])
#    plt.plot(time, pc_incorrect_av[j])

    plt.fill_between(time, pc_correct_av[j] - pc_correct_sem[j], pc_correct_av[j] + pc_correct_sem[j])
    plt.fill_between(time, pc_incorrect_av[j] - pc_incorrect_sem[j], pc_incorrect_av[j] + pc_incorrect_sem[j])
        
    if j==0:
        plt.legend(['correct','incorrect'], fontsize=8)
        ax.set(ylabel = 'mean firing rate (Hz)')
     
    plt.title('PC %d'%j)   
    

    
##%% Add "shuffled" trials
#NS = 100
#
#pc_shuff_1_array = np.ndarray([NS,npc,250])
#pc_shuff_2_array = np.ndarray([NS,npc,250])  
#
#
#for i in range(NS):
#
#    set1 = np.random.choice(NT,np.floor(NT/2))
#    set2 = np.random.choice(NT,np.floor(NT/2))
#    
#    for j in range(npc):
#        
#        pc_j = pc[j]
#        
#        pc_shuff_1_array[i,j,:] = pc_j[set1,:].mean(axis=0)
#        pc_shuff_2_array[i,j,:] = pc_j[set2,:].mean(axis=0)
#        
#        
#pc_shuff_1_av = pc_shuff_1_array.mean(axis=0)
#pc_shuff_2_av = pc_shuff_2_array.mean(axis=0)    
#
#pc_shuff_1_sd = pc_shuff_1_array.std(axis=0)
#pc_shuff_2_sd = pc_shuff_2_array.std(axis=0)    
#
##for j in range(npc):
#    
#    ax = plt.subplot(2,npc,npc+j+1)
#    pc_j = pc[j]
#       
#    plt.fill_between(time, pc_shuff_1_av[j,:] - pc_shuff_1_sd[j,:], pc_shuff_1_av[j,:] + pc_shuff_1_sd[j,:])
#    plt.fill_between(time, pc_shuff_2_av[j,:] - pc_shuff_2_sd[j,:], pc_shuff_2_av[j,:] + pc_shuff_2_sd[j,:])
#    # plt.plot(time, pc_shuff_1_av[j,:])  
#    # plt.plot(time, pc_shuff_2_av[j,:])
#     
#    if j==0:
#        plt.legend(['correct','incorrect'], fontsize=8)
#        ax.set(ylabel = 'mean firing rate (Hz)')
#        
#    ax.set(xlabel = 'Time [ms]')
#    
