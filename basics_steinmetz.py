# -*- coding: utf-8 -*-
"""
Set of basic functions that can be used to work with the Steinmetz dataset.

FUNCTIONS:
    import_dataset(data_path)
        imports the complete dataset
        
    collect_spks(...)
        
    get_time(dat)
        computes time vector for a particular session

        

Created on Tue Jul 21 12:57:32 2020

@author: BRAIN HUNTERS
"""


def import_dataset(data_path='.'):
    """ Import the complete Steimetz dataset. If the data files are already 
    available in data_path, the data is loaded from there. 
    Otherwise, the data is downloaded from osf.io and stored in data_path.
    
    Args:
        data_path: string, optional
            path where the data is located or where it should be downloaded 
            into.
    
    Returns:
        alldat: ndarray
        an array containing the data from the 39 sessions
    """

    import os, requests
    
    fname = []
    for j in range(3):
      fname.append('steinmetz_part%d.npz'%j)
      
      
    url = ["https://osf.io/agvxh/download"]
    url.append("https://osf.io/uv3mw/download")
    url.append("https://osf.io/ehmw2/download")
    
    for j in range(len(url)):
      if not os.path.isfile(fname[j]):
        try:
          r = requests.get(url[j])
        except requests.ConnectionError:
          print("!!! Failed to download data !!!")
        else:
          if r.status_code != requests.codes.ok:
            print("!!! Failed to download data !!!")
          else:
            with open(fname[j], "wb") as fid:
              fid.write(r.content)
    
    # Data loading
    import numpy as np

    
    alldat = np.array([])
    for j in range(len(fname)):
      alldat = np.hstack((alldat, np.load(fname[j], allow_pickle=True)['dat']))
      
    return alldat


def collect_spks(alldat, NT=92, balance_correct_incorrect=True, brain_regions = 'all'):
    """ Collects spiking activity of neurons across all sessions.
    
    Args:
        alldat: ndarray 
            complete Steinmetz dataset
            
        NT: scalar integer
            number of trials to select (randomly) within each session
            
        balance_correct_incorrect: bool
            if True, equal numbers of correct and incorrect trials are selected. Note that if there 
            is not enough trials in a particular session, the session is skipped.
            
        brain_regions: list
            list of brain regions to consider. Set to 'all' to consider all brain regions
            
    Returns:
        ndarray
            spks: (N_neurons x N_trials x N_bins)
        logical array
            correct: bool, true for correct trials
    """
    
    import numpy as np
    import basics_steinmetz as bs
    
    # Initialize ndarray
    spks = np.zeros([30000,NT,250])
    
    i0 = 0
    
    # Loop through all sessions
    for dat in alldat:
        
        # Select neurons that are active and located within specified brain regions
        active_neurons  = dat['spks'].sum(axis=2).mean(axis=1) > 1
        breg = bs.get_brain_region(dat['brain_area'])
        if not brain_regions == 'all':
            isin_brain_regions = np.isin(breg, brain_regions)
        else:
            isin_brain_regions = np.array(np.ones_like(dat['brain_area'])).astype(bool)
        
        spks_ = dat['spks'][active_neurons & isin_brain_regions,:,:]
        NN = spks_.shape[0]
        
        if NN == 0:
            continue
               
        # Divide trials into correct and incorrect trials
        resp = dat['response']
        stim = dat['contrast_left'] - dat['contrast_right']
        stim = np.sign(stim) 
        correct_   = resp == stim
        incorrect_ = ~correct_

        min_NT = np.min([np.sum(correct_),np.sum(incorrect_)])
        if (balance_correct_incorrect) and (2*min_NT < NT):
            # Don't take this session if number of trials is too small
            continue
       
        # Select trials randomly
        if balance_correct_incorrect:          
            NTb = int(np.floor(NT/2))
            idx1 = np.random.choice(np.array(np.where(correct_)).ravel(), NTb, replace = False)
            idx2 = np.random.choice(np.array(np.where(incorrect_)).ravel(), NTb, replace = False)
            idx  = np.concatenate([idx1,idx2])   
        else:          
            idx = np.random.choice(spks_.shape[0], NT, replace = False)
         
            
        # Store into ndarray
        spks[i0:i0+NN,:,:] = spks_[:,idx,:]
                
        i0 = i0 + NN
         
    spks = spks[:i0,:,:]
    
    # Create logical array specifying correct trials
    if balance_correct_incorrect:
        correct = np.zeros(NT)
        correct[:NTb] = 1
    else:
        correct = None
    
    
    return spks, correct.astype(bool)



def get_time(sessdat):
    """ Returns the time vector for the dataset of a specific session
    
    Args:
        sessdat: ndarray
            session dataset
            
    Returns:
        vector of time values
    
    """
    import numpy as np
    
    dt = sessdat['bin_size'] # binning 
    NT = sessdat['spks'].shape[-1]
    time = dt * np.arange(NT) - 0.5 # substract 500ms so that 0 corresponds to stimulus onset
    
    return time

    

def get_brain_region(brain_area):
    """ Returns the brain region where brain_area is located.
        Possible brain regions: ["vis ctx", "thal", "hipp", "other ctx", "midbrain", 
        "basal ganglia", "cortical subplate", "other"]
    
    Args:
        brain_area: ndarray of brain areas
            names of brain area
    
    Returns:
        region: ndarray of brain areas
    """
    
    import numpy as np
    
    region_list = ["vis_ctx", "thal", "hipp", "other_ctx", "midbrain", "basal_ganglia", "cortical_subplate", "other"]
    
    brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                    ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                    ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                    ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                    ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                    ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                    ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                    ]
    
    
    regions_id = (len(region_list)-1) * np.ones(len(brain_area)) 
    
    for j in range(len(brain_groups)):
      regions_id[np.isin(brain_area, brain_groups[j])] = j
      
    regions = np.array(region_list)[regions_id.astype(int)]
        
    return regions
        
        
        
   
    
    
    