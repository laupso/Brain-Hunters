# -*- coding: utf-8 -*-
"""
Try and play with PCA

Created on Thu Jul 23 19:02:19 2020

@author: opsomerl
"""

import numpy as np
import basics_steinmetz as bs
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 


#%% Data retrieval
alldat = bs.import_dataset()

dat = alldat[11]
    

#%% Balance the number of trials
brain_area = dat['brain_area'][:4]

region_list = ["vis_ctx", "thal", "hipp", "other_ctx", "midbrain", "basal_ganglia", "cortical_subplate", "other"]
brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                ]


regions_id = len(region_list) * np.ones(len(brain_area)) 

for j in range(len(brain_groups)):
  regions_id[np.isin(brain_area, brain_groups[j])] = j
  
regions = np.array(region_list)[regions_id.astype(int)]


    
#%% Settings
task_type  = 'go' # 'go' or 'no_go'


#%% Set some logical vectors to categorize trials
stimulus = dat['contrast_left'] - dat['contrast_right']
stimulus = np.sign(stimulus) 
response = dat['response'] # right - nogo - left (-1, 0, 1)

go        = response != 0
nogo      = response == 0
right     = stimulus < 0
left      = stimulus > 0
misses    = (stimulus != 0) & (nogo)
correct   = response == stimulus
incorrect = ~correct

  
  
#%% PCA based on pre-stimulus activity
pre_stim = np.arange(25,45)
    
NN = dat['spks'].shape[0]
NT = dat['spks'].shape[-1]

droll = np.reshape(dat['spks'][:,:,pre_stim], (NN,-1)) 
droll = droll - np.mean(droll, axis=1)[:, np.newaxis]
model = PCA(n_components = 5).fit(droll.T)
W = model.components_
pc_10ms = W @ np.reshape(dat['spks'], (NN,-1))
pc_10ms = np.reshape(pc_10ms, (5, -1, NT))



#%% Figure
        
        
plt.figure(figsize= (20, 6))
for j in range(len(pc_10ms)):
  ax = plt.subplot(2,len(pc_10ms)+1,j+1)
  pc1 = pc_10ms[j]

  plt.plot(pc1[correct   & go & left, :].mean(axis=0))  
  plt.plot(pc1[incorrect & go & left, :].mean(axis=0))
 
   
  if j==0:
    plt.legend(['correct left','incorrect left'], fontsize=8)
  ax.set(xlabel = 'binned time', ylabel = 'mean firing rate (Hz)')
  plt.title('PC %d'%j)

  ax = plt.subplot(2,len(pc_10ms)+1,len(pc_10ms)+1 + j+1)
  
  plt.plot(pc1[correct   & go & right, :].mean(axis=0))  
  plt.plot(pc1[incorrect & go & right, :].mean(axis=0))

  if j==0:
    plt.legend(['correct right','incorrect right'], fontsize=8)
  ax.set(xlabel = 'binned time', ylabel = 'mean firing rate (Hz)')
  plt.title('PC %d'%j)
  
  