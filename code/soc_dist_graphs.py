# social distancing graphs
"""
Created on Fri Apr 17 16:35:32 2020

@author: mai2125
"""

import pandas as pd
import numpy as np
import soc_dist_presets as ps
import matplotlib.pyplot as mplt




def plot_state_mobility(state):
    '''
    Given a state (str) plots the net social distancing adherence score (SoDA)
    over the course of the quarantine
    
    Input: full state name (str)
    
    Output: Plot of the SoDA vs days
    '''
    states = ps.states_df
    ax = states[states["states"] == state].plot(x="stay_at_home_days",
               y="Mobility Score")
    ax.set_title(state)
    ax.set_ylabel("Mobility Score")
    ax.axhline(color ='k')
    ax.axvline(color='r', ls="--")
    
    return

def plot_net_mobility():
    '''
    Plots the social distancing adherence score for each state in one bar graph
    '''
    ax = ps.mobi_df.plot(x="states", kind="bar", figsize=(20,10))
    ax.set_ylabel("Mobility Score")
    ax.set_title("US Mobility Scores per State")
    
    return
