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
        
        
    
    def load(self, dim=None):
        '''
        Load the feature vectors.
        
        Parameters
        ----------
        dim: int, optional
            Number of dimensions of the features.
        
        speaker_wise: bool, optional
            Whether to return speaker-wise data or a combined array
        Returns
        -------
        feature_vectors_ : dict
            Dictionary containing the speaker-wise feature arrays.

        '''
        X_combined_ = np.empty([], dtype=np.float32)
        feature_vectors_ = {}
        split_count_ = 0
        for split_id_ in self.INFO.keys():
            if not self.INFO[split_id_]['feature_name']==self.FEATURE_NAME:
                print('Wrong feature path')
                continue
            
            speaker_id_ = self.INFO[split_id_]['speaker_id']
            if speaker_id_ not in feature_vectors_.keys():
                feature_vectors_[speaker_id_] = {}
            feature_path_ = self.INFO[split_id_]['file_path']
            fv_ = np.load(feature_path_, allow_pickle=True)
            # The feature vectors must be stored as individual rows in the 2D array
            if dim:
                if np.shape(fv_)[0]==dim:
                    fv_ = fv_.T
            elif np.shape(fv_)[1]>np.shape(fv_)[0]:
                fv_ = fv_.T
            feature_vectors_[speaker_id_][split_id_] = np.array(fv_, ndmin=2)
            if np.size(X_combined_)<=1:
                X_combined_ = np.array(fv_, dtype=np.float32)
            else:
                X_combined_ = np.append(X_combined_, np.array(fv_, dtype=np.float32), axis=0)

            split_count_ += 1
            print(f'Loading features ({split_count_}/{len(self.INFO.keys())})', end='\r', flush=True)
        print('')
        
        return feature_vectors_, X_combined_
        