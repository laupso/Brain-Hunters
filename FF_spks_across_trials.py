# -*- coding: utf-8 -*-
"""
Plot the Fanno Factor of pre-stimulus firing rate within task-related regions for correct vs. incorrect 
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


#%% Compute Fano factor for correct vs incorrect
mean_fr_c = np.array([])
mean_fr_w = np.array([])
var_fr_c  = np.array([])
var_fr_w  = np.array([])
FF_fr_c   = np.array([])
FF_fr_w   = np.array([])

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
        
    # Logical arrays for selecting trials 
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
    
    fr_c = spks[:,correct_trials,:].mean(axis=2)
    fr_w = spks[:,incorrect_trials,:].mean(axis=2) 
    
    mean_fr_c = np.hstack([mean_fr_c, fr_c.mean(axis=1)])
    mean_fr_w = np.hstack([mean_fr_w, fr_w.mean(axis=1)])
    
    var_fr_c = np.hstack([var_fr_c, fr_c.var(axis=1)])
    var_fr_w = np.hstack([var_fr_w, fr_w.var(axis=1)])
    
    
FF_fr_c = var_fr_c / mean_fr_c
FF_fr_w = var_fr_w / mean_fr_w

FF_fr_c[np.isnan(FF_fr_c)] = 0
FF_fr_w[np.isnan(FF_fr_w)] = 0

mean_fr = np.vstack([mean_fr_c, mean_fr_w])
var_fr  = np.vstack([var_fr_c, var_fr_w])
FF_fr   = np.vstack([FF_fr_c, FF_fr_w])


FR_summary = {'Mean':mean_fr.T, 'Var':var_fr.T, 'FF': FF_fr.T}



#%% Figure
plts.set_fig_default()
rcParams['figure.figsize'] = [12,5] 

Y = 'FF'

plt.figure()
ax = plt.subplot(121)
plt.hist(FR_summary[Y],range=[0,60])
plt.legend(['correct','incorrect'])
ax.set(xlabel = 'Fano Factor')

ax = plt.subplot(122)
plt.boxplot(FR_summary[Y],labels=['correct','incorrect'])
ax.set(ylabel = 'Fano Factor')




