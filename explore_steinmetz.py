# -*- coding: utf-8 -*-
"""
Test script to explore the Steinmetz data set

Created on Tue Jul 21 09:49:24 2020

@author: BRAIN HUNTERS
"""

import basics_steinmetz as bs
import plots_steinmetz as plts
from matplotlib import pyplot as plt

#%% Data retrieval
alldat = bs.import_dataset()



#%% Import matplotlib and set defaults
plts.set_fig_default()



#%% Basic plots of single-trial neuron recordings

# Select just one of the recordings here. 11 is nice because it has some neurons in vis ctx. 
dat = alldat[11]
print(dat.keys())

dt = dat['bin_size'] # binning at 10 ms

time = bs.get_time(dat)

trials  = 1
neurons = 1
spks  = dat['spks'][neurons,trials].mean(axis=(0,1))

plt.plot(time,spks.T)



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




