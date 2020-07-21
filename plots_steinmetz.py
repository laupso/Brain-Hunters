# -*- coding: utf-8 -*-
"""
Set of functions for plotting beautiful Steimetz data

FUNCTIONS:
    set_fig_default()
        set default figure settings
    

Created on Tue Jul 21 16:16:36 2020

@author: BRAIN HUNTERS
"""

def set_fig_default():
    """
    Import matplotlib and set defaults
    
    Args:
        none
    
    Returns:
        none
    
    """
    
    from matplotlib import rcParams 
    rcParams['figure.figsize'] = [20, 4]
    rcParams['font.size'] = 15
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['figure.autolayout'] = True
    #try
