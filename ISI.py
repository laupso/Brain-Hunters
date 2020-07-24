# -*- coding: utf-8 -*-
"""
Test script to explore the Steinmetz data set

Created on Tue Jul 21 09:49:24 2020

@author: BRAIN HUNTERS
"""

import numpy as np
import basics_steinmetz as bs
import plots_steinmetz as plts
from matplotlib import pyplot as plt
from matplotlib import rcParams 

#%%
def calc_isi(trl_list):
    """
    calulates the interspike intervals for a given set of trials
    
    Args:
        trl_list: list of (neurons, trials, time)
    
    Returns:
        isi_list: list of intersike intervals
    
    """
    spike_list  = []
    isi_list    = []
    
    # look for each neuron
    for neuron in range(trl_list.shape[0]):
        
        # in each trial
        for trial in range(trl_list.shape[1]):
            
            # to find the ISIs 
            for spike in range(trl_list.shape[2]):
                
                if trl_list[neuron,trial,spike] > 0 :
                    
                    # first get all the spikes and save them in a list
                    spike_list = np.hstack([spike_list, time_l[spike]])
                    
            # then calculate ISI        
            for spikes in range(len(spike_list) - 1):
                
                isi_list = np.hstack([isi_list,(spike_list[spikes] - spike_list[spikes + 1])])  
                
            spike_list = []
    return isi_list
#%% Data retrieval
alldat = bs.import_dataset()

#%% Import matplotlib and set defaults
plts.set_fig_default()

#%% Calculate the mean ISI for correct vs incorrect trials

task_areas = ['VPM','PO','MD','SNr','GPo','POL','LS','ZI','DG','CA3','CAI','SCm','MRN','CP','ACB','BLA','MG','PAG'] 
pre_stim   = np.arange(25,45)

mean_isi_corr  = []
mean_isi_incorr  = []

var_isi_corr = []
var_isi_incorr = []

for area in range(len(alldat)):

    # Select just one of the recordings here. 11 is nice because it has some neurons in vis ctx. 
    dat      = alldat[area]
    
    # Store data into separate variables
    dt       = dat['bin_size'] # binning at 10 ms
    time     = bs.get_time(dat)
    time_l   = time.tolist()
    stimulus = dat['contrast_left'] - dat['contrast_right']
    stimulus = np.sign(stimulus) 
    spks     = 1/dt * dat['spks']
    response = dat['response']
    
    # Logical arrays for selecting trials and brain regions
    task_area       = np.isin(dat['brain_area'], task_areas)
    correct_trials  = response == stimulus
    correct_go      = correct_trials & (stimulus != 0)
    correct_no_go   = correct_trials & (stimulus == 0)
    incorrect_go    = ~(correct_trials) & (stimulus != 0)
    incorrect_no_go = ~(correct_trials) & (stimulus == 0)

    # only look at brain areas that are contained in the task areas
    if any(np.isin(dat['brain_area'], task_areas)): 
        
        # get correct and incorrect trials in the pre_stim interval
        rel_trls         = spks[task_area,:,:]
        rel_trls         = rel_trls[:,:,pre_stim]
        
        trls_corr        = rel_trls[:,correct_go,:]
        trls_incorr      = rel_trls[:,incorrect_go,:]
        
        isi_corr = calc_isi(trls_corr)
        isi_incorr = calc_isi(trls_incorr)
        
        mean_isi_corr.append(np.mean(isi_corr))
        mean_isi_incorr.append(np.mean(isi_incorr))
        
        var_isi_corr.append(np.var(isi_corr))
        var_isi_incorr.append(np.var(isi_incorr))

        isi_corr = []
        trls_incorr = []
 
#%%
mean_isi = np.array([np.mean(mean_isi_corr), np.mean(mean_isi_incorr)]) 
std_isi = np.array([np.std(mean_isi_corr), np.std(mean_isi_incorr)])

var_isi = np.array([np.mean(var_isi_corr), np.mean(var_isi_incorr)]) 
var_std_isi = np.array([np.std(var_isi_corr), np.std(var_isi_incorr)])
       
#%% Plot the mean
ind = np.arange(0, 2)
width = 0.05      
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xticks(ind)
ax.set_xticklabels(['Correct Trials', 'Incorrect Trials'])

plt.gca().invert_yaxis()
plt.bar(ind, mean_isi, yerr=std_isi, capsize=10)

plt.tight_layout()
plt.show()
#%% Plot the vari‚ance
ind = np.arange(0, 2)
width = 0.05      
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xticks(ind)
ax.set_xticklabels(['Correct Trials', 'Incorrect Trials'])

plt.bar(ind, var_isi, yerr=var_std_isi, capsize=10)

plt.tight_layout()
plt.show()