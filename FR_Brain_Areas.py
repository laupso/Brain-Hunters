#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot normalized firing rates in prestimulus duration in correct vs. incorrect trials separated in different regions

Created on Wed Jul 22 20:48:48 2020

@author: Brain Hunters
"""

#%% import libraries
import numpy as np
import basics_steinmetz as bs
import plots_steinmetz as plts
from matplotlib import pyplot as plt
from matplotlib import rcParams 
from scipy import stats


#%% Data retrieval
alldat = bs.import_dataset()

#%% Import matplotlib and set defaults
plts.set_fig_default()

#%% Calculate firing rate in each region

# Define prestim duration + preallocation
start_pre=25 #25th time bine
stop_pre=45
prestim_time=np.arange(start_pre,stop_pre) 

total_spikes_list=[]
tot_cor_go_av=[]
tot_incor_go_av=[]
cwi=[]
task_areas=['VPM','PO','MD','SNr','GPo','POL','LS','ZI','DG','CA3','CAI','SCm','MRN','CP','ACB','BLA','MG','PAG'] 

for area in range(len(task_areas)):
        
    for session in np.setxor1d(range(len(alldat)),[1,5,20]):
        dat=alldat[session]
        dt=dat['bin_size'] # binning at 10 ms
        spks=1/dt*dat['spks'] #number of spikes in each time bin
        stimulus=dat['contrast_left']-dat['contrast_right']
        stimulus=np.sign(stimulus) 
        response=dat['response']
        correct_trials=response == stimulus
        correct_go=correct_trials & (stimulus != 0)
        incorrect_go=~(correct_trials) & (stimulus != 0)
    
        within_area=np.isin(dat['brain_area'],task_areas[area])
        spks_in_region=spks[within_area,:,:]
        spks_in_region=spks_in_region[:,:,prestim_time]
        spks_in_region_cor_av=spks_in_region[:,correct_go,:].mean(axis=(1,2)) #average over all trials and all time bins of each neuron
        spks_in_region_incor_av=spks_in_region[:,incorrect_go,:].mean(axis=(1,2))
        total_spikes_list.append(spks_in_region) #returns all the sessions of interest
        tot_cor_go_av=np.concatenate((tot_cor_go_av,spks_in_region_cor_av),axis=None)
        tot_incor_go_av=np.concatenate((tot_incor_go_av,spks_in_region_incor_av),axis=None)
        
    #delete zero-activity neurons
    sum_go_av=tot_cor_go_av+tot_incor_go_av
    rem_idx=np.array(np.where(sum_go_av==0))
    sum_go_act=np.delete(sum_go_av,rem_idx)
    tot_cor_go_act=np.delete(tot_cor_go_av,rem_idx)
    tot_incor_go_act=np.delete(tot_incor_go_av,rem_idx)
    
    #(Normalized) firing rate for each neuron
    normFR_cor_go=tot_cor_go_act#/sum_go_act*100 #=>add it for normalization
    normFR_incor_go=tot_incor_go_act#/sum_go_act*100 #=>add it for normalization
    #Stack them together
    stak_cor_incor_normFR=np.vstack([normFR_cor_go,normFR_incor_go])
    # stak_cor_incor_FR=np.vstack([tot_cor_go_av,tot_incor_go_av]) #including zero-activity neurons
    #Average neural activity and std
    normFR_cor_go_av=normFR_cor_go.mean()
    normFR_incor_go_av=normFR_incor_go.mean()
    normFR_cor_go_sem=stats.sem(normFR_cor_go)
    normFR_incor_go_sem=stats.sem(normFR_incor_go)                
            
    #Histogram of the average of all neurons        
    plts.set_fig_default()
    rcParams['figure.figsize'] = [14,5] 
    
    plt.figure()
    ax=plt.subplot(121)

    plt.hist(stak_cor_incor_normFR.T)
    plt.legend(['correct','incorrect'])
    plt.xlabel('FR')
    plt.ylabel('# of neurons')

    #Plot firing rates
    ax=plt.subplot(122)
    plt.bar([2,3],[normFR_cor_go_av,normFR_incor_go_av],yerr=[normFR_cor_go_sem,normFR_incor_go_sem])
    
    #plot properties
    plt.ylabel('Average Firing Rate')
    plt.title('Activity in correct vs. incorrect trials in prestimulus period')
    plt.xticks([2,3], ('correct trials', 'incorrect trials'))
    plt.yticks(np.arange(0,5,.5))
    plt.ylim(0,5)
    plt.title('brain area: %s ' %task_areas[area])

        
plt.show()

#should average over all session for each brain area