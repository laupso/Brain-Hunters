# -*- coding: utf-8 -*-
"""
Set of basic functions that can be used to work with the Steinmetz dataset.

FUNCTIONS:
    import_dataset(data_path)
        imports the complete dataset
        
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
    
    
    regions_id = len(region_list) * np.ones(len(brain_area)) 
    
    for j in range(len(brain_groups)):
      regions_id[np.isin(brain_area, brain_groups[j])] = j
      
    regions = np.array(region_list)[regions_id.astype(int)]
        
    return regions
        
        
        
   
    
    
    