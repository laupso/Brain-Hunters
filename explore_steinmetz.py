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


#%% Data retrieval
alldat = bs.import_dataset()



#%% Import matplotlib and set defaults
plts.set_fig_default()



#%% Basic plots of single-trial neuron recordings
rcParams['figure.figsize'] = [5,5]  

# Select just one of the recordings here. 11 is nice because it has some neurons in vis ctx. 
dat = alldat[11]
print(dat.keys())

dt = dat['bin_size'] # binning at 10 ms

time = bs.get_time(dat)

trials  = np.array([1,2])



spks = dat['spks']


choice_regions = ['MRN','SCm','SNr','ZI','CP','MOs','MOp','PL']
within_choice  = np.isin(dat['brain_area'] , choice_regions)
trials = dat['response'] == 0


spks = spks[within_choice,:,:]
spks = spks[:,trials,:]

spks_av = spks.mean(axis=(0,1))


#choice_spks = dat['spks'][within_choice]
#choice_resp = dat['response'][within_choice]
#stimulus = dat['contrast_left'] - dat['contrast_right']
#stimulus = np.sign(stimulus) # right higher - equal - left higher (-1 - 0 - 1)
#
#
#correct_trials = response == stimulus
#correct_go = correct_trials & (stimulus != 0)
#correct_no_go = correct_trials & (stimulus == 0)
#incorrect_go = ~(correct_trials) & (stimulus != 0)
#incorrect_no_go = ~(correct_trials) & (stimulus == 0)
#
#
#
plt.plot(time,spks_av.T)



#%% Basic plots of population average


ax = plt.subplot(1,5,1)
response = dat['response'] # right - nogo - left (-1, 0, 1)
vis_right = dat['contrast_right'] # 0 - low - high
vis_left = dat['contrast_left'] # 0 - low - high
plt.plot(time, 1/dt * dat['spks'][:,response>=0].mean(axis=(0,1))) # left responses
plt.plot(time, 1/dt * dat['spks'][:,response<0].mean(axis=(0,1))) # right responses
plt.plot(time, 1/dt * dat['spks'][:,vis_right>0].mean(axis=(0,1))) # stimulus on the right
plt.plot(time, 1/dt * dat['spks'][:,vis_right==0].mean(axis=(0,1))) # no stimulus on the right

plt.legend(['left resp', 'right resp', 'right stim', 'no right stim'], fontsize=12)
ax.set(xlabel  = 'time (sec)', ylabel = 'firing rate (Hz)');


#%% Plot correct go versus incorrect go
ax = plt.subplot(1,5,1)


plt.plot(time, 1/dt * dat['spks'][:,correct_go].mean(axis=(0,1))) # correct go response
plt.plot(time, 1/dt * dat['spks'][:,incorrect_go].mean(axis=(0,1))) # right responses
# plt.plot(dt * np.arange(NT), 1/dt * dat['spks'][:,vis_right>0].mean(axis=(0,1))) # stimulus on the right
# plt.plot(dt * np.arange(NT), 1/dt * dat['spks'][:,vis_right==0].mean(axis=(0,1))) # no stimulus on the right
plt.plot(np.array([0,0]),np.array([3,5]))
plt.plot(np.array([go_cue[2],go_cue[2]]),np.array([3,5]))
print(go_cue[2])

plt.legend(['correct go', 'incorrect go', 'Stimulus onset','go cue'], fontsize=12)
ax.set(xlabel  = 'time (sec)', ylabel = 'firing rate (Hz)');

