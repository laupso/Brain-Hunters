# -*- coding: utf-8 -*-
"""
Useful functions to apply PCA on Steinmetz dataset
Created on Mon Jul 27 15:21:16 2020

@author: opsomerl
"""


def fit_pca_model(spks,n_comp=None):
    
    import numpy as np
    from sklearn.decomposition import PCA 
        
    NN = spks.shape[0]  # Number of neurons 

    droll = np.reshape(spks, (NN,-1)) 
    droll = droll - np.mean(droll, axis=1)[:, np.newaxis]
    model = PCA(n_components = n_comp).fit(droll.T)
    
    return model

    
    
def compute_pca(spks, model):
    
    import numpy as np
    
    NN = spks.shape[0]  # Number of neurons 
    NB = spks.shape[-1] # Number of time bins
    
    W = model.components_
    n_comp = model.n_components_
    pc = W @ np.reshape(spks, (NN,-1))
    pc = np.reshape(pc, (n_comp, -1, NB))
    
    return pc
    
    
def plot_explained_variance(model, threshold = 0.9):
    
    from matplotlib import pyplot as plt
    import numpy as np
    
    cum_exp_var = np.cumsum(model.explained_variance_ratio_)
    
    ncomp_thrsh = np.where(cum_exp_var > threshold)[0]
    
    plt.figure(figsize = (7,6))
    ax = plt.subplot(111)
    plt.plot(np.cumsum(model.explained_variance_ratio_))
    plt.plot(model.explained_variance_ratio_)
    
    plt.legend(("Cumulative","Explained Variance"))
    
    if len(ncomp_thrsh) > 0:
        plt.hlines(threshold, 0, ncomp_thrsh[0], color = 'k', linestyle = '--')
        plt.vlines(ncomp_thrsh[0], 0, threshold, color = 'k', linestyle = '--')
    
    ax.set(xlabel = 'Number of components', ylabel = 'Explained Variance Ratio')
    plt.grid(True)
    
    
    
    