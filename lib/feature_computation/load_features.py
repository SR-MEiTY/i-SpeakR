#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 13:36:51 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
import numpy as np

class LoadFeatures:
    INFO = None
    FEATURE_NAME = ''
    
    def __init__(self, info, feature_name):
        '''
        Initialize the feature loading class object.

        Parameters
        ----------
        info : dict
            Details regarding the computed features.
        feature_name : str
            Name of the feature to be loaded.

        Returns
        -------
        feature_vector_: dict
            Dictionary containing the speaker wise feature arrays.

        '''
        self.INFO = info
        self.FEATURE_NAME = feature_name
        
        
    
    def load(self):
        '''
        Load the feature vectors.

        Returns
        -------
        feature_vectors_ : dict
            Dictionary containing the speaker-wise feature arrays.

        '''
        feature_vectors_ = {}
        for split_id_ in self.INFO.keys():
            if not self.INFO[split_id_]['feature_name']==self.FEATURE_NAME:
                print('Wrong feature path')
                continue
            
            speaker_id_ = self.INFO[split_id_]['speaker_id']
            if speaker_id_ not in feature_vectors_.keys():
                feature_vectors_[speaker_id_] = {}
            feature_path_ = self.INFO[split_id_]['file_path']
            fv = np.load(feature_path_, allow_pickle=True)
            # The feature vectors must be stored as individual rows in the 2D array
            if np.shape(fv)[1]>np.shape(fv)[0]:
                fv = fv.T
            if speaker_id_ not in feature_vectors_.keys():
                feature_vectors_[speaker_id_] = np.empty([], dtype=np.float32)
                        
            if np.size(feature_vectors_[speaker_id_])<=1:
                feature_vectors_[speaker_id_] = np.array(fv, ndmin=2)
            else:
                feature_vectors_[speaker_id_] = np.append(feature_vectors_[speaker_id_], np.array(fv, ndmin=2), axis=0)
            
        return feature_vectors_            
        