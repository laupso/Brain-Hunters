# -*- coding: utf-8 -*-
"""
Plot the variance of pre-stimulus firing rate within task-related regions for correct vs. incorrect 
trials.

Created on Wed Jul 22 16:56:53 2020

@author: BRAIN HUNTERS
"""
import numpy as np
import basics_steinmetz as bs
import plots_steinmetz as plts
from matplotlib import pyplot as plt
from matplotlib import rcParams 


#%% Data retrieval
alldat = bs.import_dataset()



#%% Settings
task_type  = 'go' # 'go' or 'no_go'
task_areas = ['VPM','PO','MD','SNr','GPo','POL','LS','ZI','DG','CA3','CAI','SCm','MRN','CP','ACB','BLA','MG','PAG'] 
sessions   = np.arange(0,39) 
pre_stim   = np.arange(25,45)


#%% Compute variance across trials for correct vs incorrect

var_spks_corr   = np.array([])
var_spks_incorr = np.array([])


for s in sessions:   

    # Select session data
    dat = alldat[s]
    
    # Get data
    dt       = dat['bin_size']
    time     = bs.get_time(dat)[pre_stim]
    stimulus = dat['contrast_left'] - dat['contrast_right']
    stimulus = np.sign(stimulus) 
    spks     = 1/dt * dat['spks'][:,:,pre_stim]
    response = dat['response']
    go_cue   = dat['gocue']
    areas    = dat['brain_area']
        
    # Logical arrays for selecting trials and brain regions
    correct_trials  = response == stimulus
    if task_type == 'go':
        correct_trials = correct_trials & (stimulus != 0)
        incorrect_trials = ~(correct_trials) & (stimulus != 0)
    else:
        correct_trials = correct_trials & (stimulus == 0)
        incorrect_trials = ~(correct_trials) & (stimulus == 0)
        
      
    # Select neurons within specified brain areas
    isin_task_area = np.isin(areas,task_areas)
    if not np.any(isin_task_area):
        continue
        
    spks = spks[isin_task_area,:,:]
    
    spks_c = spks[:,correct_trials,:].mean(axis=2)
    spks_i = spks[:,incorrect_trials,:].mean(axis=2)
    
    var_spks_corr   = np.hstack([var_spks_corr, np.var(spks_c,axis = 1)])
    var_spks_incorr = np.hstack([var_spks_incorr, np.var(spks_i,axis = 1)])
    

var_spks = np.vstack([var_spks_corr, var_spks_incorr]) # 1st row = correct trials, 2nd row is incorrect trials
    

#%% Figure
plts.set_fig_default()
rcParams['figure.figsize'] = [12,5] 

plt.figure()
ax = plt.subplot(121)
plt.hist(var_spks.T, range = [0,200])
plt.legend(['correct','incorrect'])
ax.set(xlabel = 'FR variance (Hz^2)')

ax = plt.subplot(122)
plt.boxplot(var_spks.T,labels=['correct','incorrect'])
ax.set(ylabel = 'FR variance (Hz^2)')