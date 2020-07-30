#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:01:07 2020

@author: Brain Hunters
"""

#%% import libraries
import numpy as np
import basics_steinmetz as bs
import plots_steinmetz as plts
from matplotlib import pyplot as plt
# from matplotlib import rcParams 
# from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import style
from sklearn.model_selection import train_test_split
style.use('fivethirtyeight')
from sklearn.neighbors import KNeighborsClassifier

#%% Data retrieval
alldat = bs.import_dataset()

#%% Import matplotlib and set defaults
plts.set_fig_default()

#%% Calculate firing rate in each region

# Define prestim duration + preallocation
start_pre=25 #25th time bine
stop_pre=45
prestim_time=np.arange(start_pre,stop_pre) 

# task_areas=['VPM','PO','MD','SNr','GPo','POL','LS','ZI','DG','CA3','CAI','SCm','MRN','CP','ACB','BLA','MG','PAG'] 
task_areas_good=['PO','MD']

total_spikes_list=[]
X=np.zeros((548,len(task_areas_good))) #needs modification for of 548

for area in range(len(task_areas_good)):
    tot_cor_go_av=[]
    tot_incor_go_av=[]
    
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
            spks_in_region_cor_av=spks_in_region[:,correct_go,:].mean(axis=(1,2)) #average over time bins and trials
            spks_in_region_incor_av=spks_in_region[:,incorrect_go,:].mean(axis=(1,2))
            
            #the total variables
            total_spikes_list.append(spks_in_region) #returns all the sessions of interest
            tot_cor_go_av=np.concatenate((tot_cor_go_av,spks_in_region_cor_av),axis=None) #average FR over all trials and time bins
            tot_incor_go_av=np.concatenate((tot_incor_go_av,spks_in_region_incor_av),axis=None)
            
        
    #delete zero-activity neurons
    sum_go_av=tot_cor_go_av+tot_incor_go_av
    rem_idx=np.array(np.where(tot_cor_go_av<=1) or np.where(tot_incor_go_av<=1))
    tot_cor_go_act=np.delete(tot_cor_go_av,rem_idx)
    tot_incor_go_act=np.delete(tot_incor_go_av,rem_idx)
    #FR_correct=tot_cor_go_act--FR_incorrect=tot_incor_go_act
    tot_cor_go_act=np.random.choice(tot_cor_go_act,274) #randomly subsample with the number of elements the smallest brain region has
    tot_incor_go_act=np.random.choice(tot_incor_go_act,274) #needs modification for 274
    
    X[:,area]=np.hstack([tot_cor_go_act,tot_incor_go_act]) #make an array containing average of correct and incorrect in one column and each column represents each region
    # X.append(np.hstack([tot_cor_go_act,tot_incor_go_act])) 
    
#%% 
X,target=bs.collect_spks(alldat, NT=92, balance_correct_incorrect=True, brain_regions = 'all')

#%% 0. Load in the data and split the descriptive and the target feature
X_train, X_test, y_train, y_test = train_test_split(X.mean(axis=2).T,1*target,test_size=0.2,random_state=0) 
target=pd.Series(target)
X_train=pd.DataFrame(X_train)

#%% 1. Standardize the data
for col in X_train.columns:
    X_train[col] = StandardScaler().fit_transform(X_train[col].values.reshape(-1,1))

#%% 2. Compute the mean vector mu and the mean vector per class mu_k
mu=np.mean(X_train,axis=0).values.reshape(-1,1) # Mean vector mu --> Since the data has been standardized, the data means are zero 

mu_k = []

for i,orchid in enumerate(np.unique(target)):
    mu_k.append(np.mean(X_train.where(target==orchid),axis=0))
mu_k = np.array(mu_k).T


#%% 3. Compute the Scatter within and Scatter between matrices
data_SW = []
Nc = []
for i,orchid in enumerate(np.unique(target)):
    a = np.array(X_train.where(target==orchid).dropna().values-mu_k[:,i].reshape(1,-1))
    data_SW.append(np.dot(a.T,a))
    Nc.append(np.sum(target==orchid))
    
SW = np.sum(data_SW,axis=0)
SB = np.dot(Nc*np.array(mu_k-mu),np.array(mu_k-mu).T)
   
#%% 4. Compute the Eigenvalues and Eigenvectors of SW^-1 SB
eigval, eigvec = np.linalg.eig(np.dot(np.linalg.inv(SW),SB))

#%% 5. Select the two largest eigenvalues 
eigen_pairs = [[np.abs(eigval[i]),eigvec[:,i]] for i in range(len(eigval))]
eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0],reverse=True)
w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real)) # Select two largest

#%% 6. Transform the data with Y=X*w
Y = X_train.dot(w)

#%% Plot the data
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.set_xlim(-3,3)
ax0.set_ylim(-3,3)
target_label=['incorrect','correct']

for l,c,m in zip(np.unique(y_train),['r','g'],['s','x']):
    ax0.scatter(Y[0][y_train==l],
                Y[1][y_train==l],
               c=c, marker=m, label=target_label[np.int(l)],edgecolors='black')
ax0.legend(loc='upper right')


# Plot the voroni spaces
means = []

for m,target in zip(['s','x'],np.unique(y_train)):
    means.append(np.mean(Y[y_train==target],axis=0))
    ax0.scatter(np.mean(Y[y_train==target],axis=0)[0],np.mean(Y[y_train==target],axis=0)[1],marker=m,c='black',s=100)


mesh_x, mesh_y = np.meshgrid(np.linspace(-3,3),np.linspace(-4,3)) 
mesh = []


for i in range(len(mesh_x)):
    for j in range(len(mesh_x[0])):
        date = [mesh_x[i][j],mesh_y[i][j]]
        mesh.append((mesh_x[i][j],mesh_y[i][j]))


NN = KNeighborsClassifier(n_neighbors=1)
NN.fit(means,['r','g'])        
predictions = NN.predict(np.array(mesh))

ax0.scatter(np.array(mesh)[:,0],np.array(mesh)[:,1],color=predictions,alpha=0.3)


plt.show()