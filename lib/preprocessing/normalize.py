#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:51:15 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
import numpy as np

class Normalize:
    
    def __init__(self):
        return
    
    def mean_max_normalize(self, X):
        X -= np.mean(X)
        X /= np.max(np.abs(X))
        return X
    
    def mean_stdev_normalize(self, X):
        X -= np.mean(X)
        X /= np.std(X)
        return X    