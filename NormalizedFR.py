#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot normalized firing rates in prestimulus duration in correct vs. incorrect trials

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

#%% Calculate normalized firing rate

# Define prestim duration + preallocation
start_pre=25 #25th time bine
stop_pre=45
prestim_time=np.arange(start_pre,stop_pre) 

total_spikes_list=[]
tot_cor_go_av=[]
tot_incor_go_av=[]
cwi=[]
task_areas=['VPM','PO','MD','SNr','GPo','POL','LS','ZI','DG','CA3','CAI','SCm','MRN','CP','ACB','BLA','MG','PAG'] 

for session in range(len(alldat)):
    dat=alldat[session]
    dt=dat['bin_size'] # binning at 10 ms
    spks=1/dt*dat['spks'] #number of spikes in each time bin
    within_task=np.isin(dat['brain_area'],task_areas)
    stimulus=dat['contrast_left']-dat['contrast_right']
    stimulus=np.sign(stimulus) 
    response=dat['response']
    correct_trials=response == stimulus
    correct_go=correct_trials & (stimulus != 0)
    incorrect_go=~(correct_trials) & (stimulus != 0)
    if sum(within_task)==0:
        print('session n.',session, 'is not of interest')
        continue
    else:
        spks_in_region=spks[within_task,:,:]
        spks_in_region=spks_in_region[:,:,prestim_time]
        spks_in_region_cor_av=spks_in_region[:,correct_go,:].mean(axis=(0,1))
        spks_in_region_incor_av=spks_in_region[:,incorrect_go,:].mean(axis=(0,1))
        total_spikes_list.append(spks_in_region) #returns all the sessions of interest
        tot_cor_go_av=np.concatenate((tot_cor_go_av,spks_in_region_cor_av),axis=None)
        tot_incor_go_av=np.concatenate((tot_incor_go_av,spks_in_region_incor_av),axis=None)
        
#calculate Correct/Wron Index
cwi=(tot_cor_go_av-tot_incor_go_av)/(tot_cor_go_av+tot_incor_go_av)*100
cwi=np.reshape(cwi,(-1,20))
cwi_av=cwi.mean(axis=0) #average over CWIs of each time bin in all sessions
cwi_std=cwi.std(axis=0)
cwi_sem=stats.sem(cwi,axis=0)

#%% Check the significancy of CWI against popmean=zero

ttest,pvalue=stats.ttest_1samp(cwi,0) #should 2sample be used?
pvalue_binary=pvalue<.05  #true if significant
pvalue_binary=1*pvalue_binary # 1 if significant


 #%% Plot CWI in correct vs incorrect trials   
rcParams['figure.figsize'] = [5,5]  

plt.figure()
plt.errorbar(prestim_time,cwi_av,yerr=cwi_sem,label='Average CWI+/-SEM')
plt.plot(prestim_time,np.zeros(cwi_av.shape),label='Zero-Mean line')
plt.scatter(prestim_time,pvalue_binary,c='green',label='Significant points')

plt.legend()
plt.show()