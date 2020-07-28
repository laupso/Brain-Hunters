#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:36:42 2020

@author: Brain Hunters
"""


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
# from scipy import stats


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
tot_ff_cor_go=[]
tot_ff_incor_go=[]

# task_areas=['VPM','PO','MD','SNr','GPo','POL','LS','ZI','DG','CA3','CAI','SCm','MRN','CP','ACB','BLA','MG','PAG'] 
task_areas_good=['PO','MD']

for area in range(len(task_areas_good)):
        
    for session in range(len(alldat)):
        dat=alldat[session]
        within_area=np.isin(dat['brain_area'],task_areas_good[area])
        
        if sum(within_area)==0:
            print('session n.',session, 'is not of interest')
            continue
    
        else:
            dt=dat['bin_size'] #binning at 10 ms
            spks=1/dt*dat['spks'] #number of spikes in each time bin
            stimulus=dat['contrast_left']-dat['contrast_right']
            stimulus=np.sign(stimulus) 
            response=dat['response']
            correct_trials=response == stimulus
            correct_go=correct_trials & (stimulus != 0)
            incorrect_go=~(correct_trials) & (stimulus != 0)
        
            spks_in_region=spks[within_area,:,:]
            spks_in_region=spks_in_region[:,:,prestim_time]
            spks_in_region_cor_av_tim=spks_in_region[:,correct_go,:].mean(axis=(2)) #average over time bins 
            spks_in_region_incor_av_tim=spks_in_region[:,incorrect_go,:].mean(axis=(2))
            spks_in_region_cor_av=spks_in_region_cor_av_tim.mean(axis=(1)) #average over time bins and trials
            spks_in_region_incor_av=spks_in_region_incor_av_tim.mean(axis=(1))
            spks_in_region_cor_var=spks_in_region_cor_av_tim.var(axis=(1)) 
            spks_in_region_incor_var=spks_in_region_incor_av_tim.var(axis=(1))
            #the total variables
            total_spikes_list.append(spks_in_region) #returns all the sessions of interest
            tot_ff_cor_go=np.hstack([tot_ff_cor_go,(spks_in_region_cor_var)])#/spks_in_region_cor_av)])
            tot_ff_incor_go=np.hstack([tot_ff_incor_go,(spks_in_region_incor_var)])#/spks_in_region_cor_av)])
            tot_cor_go_av=np.concatenate((tot_cor_go_av,spks_in_region_cor_av),axis=None) #average FR over all trials and time bins
            tot_incor_go_av=np.concatenate((tot_incor_go_av,spks_in_region_incor_av),axis=None)
            
        
    #delete zero-activity neurons
    sum_go_av=tot_cor_go_av+tot_incor_go_av
    rem_idx=np.array(np.where(tot_cor_go_av<=3) or np.where(tot_incor_go_av<=3))
    sum_go_act=np.delete(sum_go_av,rem_idx)
    tot_cor_go_act=np.delete(tot_cor_go_av,rem_idx)
    tot_incor_go_act=np.delete(tot_incor_go_av,rem_idx)
    tot_ff_cor_go_act=np.delete(tot_ff_cor_go,rem_idx)
    tot_ff_incor_go_act=np.delete(tot_ff_incor_go,rem_idx)
    
#%% Scatter plot FF by FR
rcParams['figure.figsize'] = [10,10] 
fig, ax = plt.subplots()

plt.scatter(tot_ff_cor_go_act,tot_cor_go_act,c='blue',label='correct')
plt.scatter(tot_ff_incor_go_act,tot_incor_go_act,c='red',label='incorrect')

ax.set(xlabel='var',ylabel='FR',title='Scatter Plot-FR vs. var',xlim=[0,100],ylim=[0,10])


plt.legend()
plt.show()
            
