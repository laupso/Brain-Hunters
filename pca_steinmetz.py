# -*- coding: utf-8 -*-
"""
Useful functions to apply PCA on Steinmetz dataset
Created on Mon Jul 27 15:21:16 2020

@author: opsomerl
"""


def collect_spks(alldat, NT=92, balance_correct_incorrect=True, brain_areas='all'):
    
    import numpy as np
    
    spks = np.zeros([30000,NT,250])
    
    i0 = 0
    
    for dat in alldat:
        
        resp = dat['response']
        stim = dat['contrast_left'] - dat['contrast_right']
        stim = np.sign(stim) 
        correct_   = resp == stim
        incorrect_ = ~correct_
        
        min_NT = np.min([np.sum(correct_),np.sum(incorrect_)])
        
        if (balance_correct_incorrect) and (2*min_NT < NT):
            # Don't take this session if number of trials is too small
            continue
       
        active_neurons  = dat['spks'].sum(axis=2).mean(axis=1) > 1
        if not brain_areas == 'all':
            isin_brain_area = np.isin(dat['brain_area'], brain_areas)
        else:
            isin_brain_area = np.array(np.ones_like(dat['brain_area'])).astype(bool)
        
        spks_ = dat['spks'][active_neurons & isin_brain_area,:,:]
        NN = spks_.shape[0]
        
        if balance_correct_incorrect:          
            NTb = int(np.floor(NT/2))
            idx1 = np.random.choice(np.array(np.where(correct_)).ravel(), NTb, replace = False)
            idx2 = np.random.choice(np.array(np.where(incorrect_)).ravel(), NTb, replace = False)
            idx  = np.concatenate([idx1,idx2])   
        else:          
            idx = np.random.choice(spks_.shape[0], NT, replace = False)
            
        
        spks[i0:i0+NN,:,:] = spks_[:,idx,:]
                
        i0 = i0 + NN
        
    
    spks = spks[:i0,:,:]
    
    if balance_correct_incorrect:
        correct = np.zeros(NT)
        correct[:NTb] = 1
    else:
        correct = None
    
    
    return spks, correct.astype(bool)


def fit_pca_model(spks,n_comp):
    
    import numpy as np
    from sklearn.decomposition import PCA 
        
    NN = spks.shape[0]  # Number of neurons 

    droll = np.reshape(spks, (NN,-1)) 
    droll = droll - np.mean(droll, axis=1)[:, np.newaxis]
    model = PCA(n_components = n_comp).fit(droll.T)
    
    return model
    
    
def compute_pca(spks, model, n_comp):
    
    import numpy as np
    
    NN = spks.shape[0]  # Number of neurons 
    NB = spks.shape[-1] # Number of time bins
    
    W = model.components_
    pc = W @ np.reshape(spks, (NN,-1))
    pc = np.reshape(pc, (n_comp, -1, NB))
    
    return pc
    
    