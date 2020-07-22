# -*- coding: utf-8 -*-
"""
Set of basic functions that can be used to work with the Steinmetz dataset.

FUNCTIONS:
    import_dataset(data_path)
        imports the complete dataset
        
    get_time(sessdat)
        compute time vector for a particular session
        


Created on Tue Jul 21 12:57:32 2020

@author: BRAIN HUNTERS

##CommentfromClaire
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
