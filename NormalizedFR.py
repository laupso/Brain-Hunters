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
        
#delete zero-activity neurons--
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
    
#calculate Correct/Wron Index
cwi=(tot_cor_go_av-tot_incor_go_av)/(tot_cor_go_av+tot_incor_go_av)*100
cwi=np.reshape(cwi,(-1,20))
cwi_av=cwi.mean(axis=0) #average over CWIs of each time bin in all sessions
cwi_std=cwi.std(axis=0)
cwi_sem=stats.sem(cwi,axis=0)

#%% Check the significancy of CWI against popmean=zero

ttest0,pvalue0=stats.ttest_1samp(cwi,0) 
pvalue_binary=pvalue0<.05  #true if significant
pvalue_binary=1*pvalue_binary # 1 if significant

ttest,pvalue=stats.ttest_rel(normFR_cor_go,normFR_incor_go)  

 #%% Plot CWI in correct vs incorrect trials   
rcParams['figure.figsize'] = [5,5]  

plt.figure()
plt.errorbar(prestim_time,cwi_av,yerr=cwi_sem,label='Average CWI+/-SEM')
plt.plot(prestim_time,np.zeros(cwi_av.shape),label='Zero-Mean line')
plt.scatter(prestim_time,pvalue_binary,c='green',label='Significant points')

plt.legend()
plt.show()

#%%add bar graph

plts.set_fig_default()
rcParams['figure.figsize'] = [14,5] 

plt.figure()
ax=plt.subplot(121)

plt.hist(stak_cor_incor_normFR.T)
# plt.title('brain area: %s ' %task_areas[area])
plt.legend(['correct','incorrect'])
plt.xlabel('FR')
plt.ylabel('# of neurons')

#Plot firing rates
ax=plt.subplot(122)
plt.bar([2,3],[normFR_cor_go_av,normFR_incor_go_av],yerr=[normFR_cor_go_sem,normFR_incor_go_sem])

#plot properties
plt.ylabel('Average Firing Rate')
plt.xticks([2,3], ('correct trials', 'incorrect trials'))
plt.yticks(np.arange(0,5,.5))
plt.ylim(0,5)
plt.title('p-value= %f ' %pvalue)

    
plt.show()
