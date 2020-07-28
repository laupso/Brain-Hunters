# -*- coding: utf-8 -*-
"""
Apply PCA on pre-stimulus zone for all neurons. In order to balance the number of correct/incorrect 
trials across sessions, trials are drawn randomly from each session

Created on Fri Jul 24 15:31:58 2020

@author: opsomerl
"""


import numpy as np
import basics_steinmetz as bs
import pca_steinmetz as pcas
from matplotlib import pyplot as plt



#%% Data retrieval
alldat = bs.import_dataset()



#%% Merge sessions
np.random.seed(None)

# Number of trials
NT = 92

spks, correct = pcas.collect_spks(alldat, NT, True)


  
#%%PCA based on pre-stimulus activity
pre_stim = np.arange(25,45)

npc = 5
    

model = pcas.fit_pca_model(spks[:,:,pre_stim], npc)

pc = pcas.compute_pca(spks, model, npc)
    

#%% Plot PC
       
plt.figure(figsize= (17, 7))

time = bs.get_time(alldat[0])*1000

for j in range(npc):
    ax = plt.subplot(2,npc,j+1)
    pc_j = pc[j]
     
    plt.plot(time, pc_j[correct, :].mean(axis=0))  
    plt.plot(time, pc_j[~correct, :].mean(axis=0))
     
    if j==0:
        plt.legend(['correct','incorrect'], fontsize=8)
        ax.set(ylabel = 'mean firing rate (Hz)')
     
    plt.title('PC %d'%j)   
    
    
#%% Add "shuffled" trials
NS = 2

pc_shuff_1_array = np.ndarray([NS,npc,250])
pc_shuff_2_array = np.ndarray([NS,npc,250])  


for i in range(NS):

    set1 = np.random.choice(NT,int(np.floor(NT/2)))
    set2 = np.random.choice(NT,int(np.floor(NT/2)))
    
    for j in range(npc):
        
        pc_j = pc[j]
        
        pc_shuff_1_array[i,j,:] = pc_j[set1,:].mean(axis=0)
        pc_shuff_2_array[i,j,:] = pc_j[set2,:].mean(axis=0)
        
        
pc_shuff_1_av = pc_shuff_1_array.mean(axis=0)
pc_shuff_2_av = pc_shuff_2_array.mean(axis=0)    

pc_shuff_1_sd = pc_shuff_1_array.std(axis=0)
pc_shuff_2_sd = pc_shuff_2_array.std(axis=0)    

for j in range(npc):
    
    ax = plt.subplot(2,npc,npc+j+1)
    pc_j = pc[j]
       
    plt.fill_between(time, pc_shuff_1_av[j,:] - pc_shuff_1_sd[j,:], pc_shuff_1_av[j,:] + pc_shuff_1_sd[j,:])
    plt.fill_between(time, pc_shuff_2_av[j,:] - pc_shuff_2_sd[j,:], pc_shuff_2_av[j,:] + pc_shuff_2_sd[j,:])

     
    if j==0:
        plt.legend(['correct','incorrect'], fontsize=8)
        ax.set(ylabel = 'mean firing rate (Hz)')
        
    ax.set(xlabel = 'Time [ms]')
    
