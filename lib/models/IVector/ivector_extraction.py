#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:31:50 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
from sklearn.mixture import GaussianMixture
import numpy as np
from lib.models.GMM_UBM.speaker_adaptation import SpeakerAdaptation
import pickle
import os
# from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lib.metrics.performance_metrics import PerformanceMetrics
import psutil
import sys


class IVector:
    NCOMP = 0
    MODEL_DIR = ''
    BACKGROUND_MODEL = None
    FEATURE_SCALING = 0
    SCALER = None
    N_BATCHES = 0
    
    
    def __init__(self, model_dir, opDir, num_mixtures=128, feat_scaling=0):
        self.MODEL_DIR = model_dir
        self.OPDIR = opDir
        self.NCOMP = num_mixtures
        self.FEATURE_SCALING = feat_scaling
        self.N_BATCHES = 1
    
    
    def train_ubm(self, X, cov_type='diag', max_iterations=100, num_init=1, verbosity=1):
        '''
        Training the GMM Universal Background Model.

        Parameters
        ----------
        X : dict
            Dictionary containing the speaker-wise fature arrays.
        cov_type : str, optional
            Type of covariance to be used to train the GMM. The default is 'diag'.
        max_iterations : int, optional
            Maximum number of iterations to train the GMM. The default is 100.
        num_init : int, optional
            Number of random initializations of the GMM model. The default is 3.
        verbosity : int, optional
            Flag to indicate whether to print GMM training outputs. The default is 1.

        Returns
        -------
        None.

        '''
        X_combined_ = np.empty([], dtype=np.float32)
        for speaker_id_ in X.keys():
            for split_id_ in X[speaker_id_].keys():
                if np.size(X_combined_)<=1:
                    X_combined_ = np.array(X[speaker_id_][split_id_], dtype=np.float32)
                else:
                    X_combined_ = np.append(X_combined_, np.array(X[speaker_id_][split_id_], dtype=np.float32), axis=0)
        print(f'Development data shape={np.shape(X_combined_)} data_type={X_combined_.dtype}')

        ''' Feature Scaling '''
        if self.FEATURE_SCALING==1:
            self.SCALER = StandardScaler(with_mean=True, with_std=False).fit(X_combined_)
            X_combined_ = self.SCALER.transform(X_combined_)
            X_combined_ = X_combined_.astype(np.float32)
        elif self.FEATURE_SCALING==2:
            self.SCALER = StandardScaler(with_mean=True, with_std=True).fit(X_combined_)
            X_combined_ = self.SCALER.transform(X_combined_)
            X_combined_ = X_combined_.astype(np.float32)
        
        ram_mem_avail_ = psutil.virtual_memory().available >> 20 # in MB; >> 30 in GB
        print(f'Available RAM: {ram_mem_avail_} MB')
        
        if isinstance(X_combined_, np.float32):
            ram_mem_req_ = 2*(np.size(X_combined_)*4 + np.shape(X_combined_)[0]*self.NCOMP*4) >> 20
        elif isinstance(X_combined_, np.float64):
            ram_mem_req_ = 2*(np.size(X_combined_)*8 + np.shape(X_combined_)[0]*self.NCOMP*8) >> 20
        else:
            ram_mem_req_ = 2*(np.size(X_combined_)*8 + np.shape(X_combined_)[0]*self.NCOMP*8) >> 20
        print(f'RAM required: {ram_mem_req_} MB')

        if ram_mem_req_>ram_mem_avail_:
            self.N_BATCHES = int(np.ceil(ram_mem_req_/(0.3*ram_mem_avail_)))
            '''
            Batch-wise training GMM-UBM
            '''
            print(f'Training GMM-UBM in a batch-wise manner. # Batches={self.N_BATCHES}')
            
            batch_size_ = int(np.shape(X_combined_)[0]/self.N_BATCHES)
            random_sample_idx_ = list(range(np.shape(X_combined_)[0]))
            np.random.shuffle(random_sample_idx_)
            batch_start_ = 0
            batch_end_ = 0
            for batch_i_ in range(self.N_BATCHES):
                batch_start_ = batch_end_
                batch_end_ = np.min([batch_start_+batch_size_, np.shape(X_combined_)[0]])
                X_combined_batch_ = np.array(X_combined_[random_sample_idx_[batch_start_:batch_end_], :], dtype=np.float32)
                if batch_i_==0:
                    training_success_ = False
                    reg_covar_ = 1e-6
                    while not training_success_:
                        try:
                            self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbosity, reg_covar=reg_covar_)
                            self.BACKGROUND_MODEL.fit(X_combined_batch_)
                            training_success_ = True
                        except:
                            reg_covar_ = np.max([reg_covar_*10, 1e-1])
                            print(f'Singleton component error. Reducing reg_covar to {np.round(reg_covar_,6)}')

                    print(f'Batch: {batch_i_+1} model trained')
                else:
                    adapt_ = None
                    del adapt_
                    adapt_ = SpeakerAdaptation().adapt_ubm(X_combined_batch_.T, self.BACKGROUND_MODEL, use_adapt_w_cov=False)
                    self.BACKGROUND_MODEL.means_ = adapt_['means']
                    self.BACKGROUND_MODEL.weights_ = adapt_['weights']
                    self.BACKGROUND_MODEL.covariances_ = adapt_['covariances']
                    self.BACKGROUND_MODEL.precisions_ = adapt_['precisions']
                    self.BACKGROUND_MODEL.precisions_cholesky_ = adapt_['precisions_cholesky']
                    print(f'Batch: {batch_i_+1} model updated')
        
        else:
            training_success_ = False
            reg_covar_ = 1e-6
            while not training_success_:
                try:
                    self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbosity, reg_covar=reg_covar_)
                    self.BACKGROUND_MODEL.fit(X_combined_)
                    training_success_ = True
                except:
                    reg_covar_ = np.max([reg_covar_*10, 1e-1])
                    print(f'Singleton component error. Reducing reg_covar to {np.round(reg_covar_,6)}')
        
        ubm_fName = self.MODEL_DIR + '/ubm.pkl'
        with open(ubm_fName, 'wb') as f:
            pickle.dump({'model':self.BACKGROUND_MODEL, 'scaler':self.SCALER}, f, pickle.HIGHEST_PROTOCOL)
        
        return
    
    
    def Baum_Welch_Statistics(self, X):
        '''
        Adaptation of the UBM model for each enrolling speaker.

        Parameters
        ----------
        X : dict
            Dictionary containing the speaker-wise data.

        Returns
        -------
        None.

        '''
        if not self.BACKGROUND_MODEL:
            ubm_fName_ = self.MODEL_DIR + '/ubm.pkl'
            if not os.path.exists(ubm_fName_):
                print('Background model does not exist')
                return
            try:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)['model']
                with open(ubm_fName_, 'rb') as f_:
                    self.SCALER = pickle.load(f_)['scaler']
            except:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)
        

        ''' Feature Scaling '''
        if self.FEATURE_SCALING>0:
            X_combined_ = np.empty([], dtype=np.float32)
            for speaker_id_ in X.keys():
                for split_id_ in X[speaker_id_].keys():
                    if np.size(X_combined_)<=1:
                        X_combined_ = np.array(X[speaker_id_][split_id_], dtype=np.float32)
                    else:
                        X_combined_ = np.append(X_combined_, np.array(X[speaker_id_][split_id_], dtype=np.float32), axis=0)
    
            if self.FEATURE_SCALING==1:
                self.SCALER = StandardScaler(with_mean=True, with_std=False).fit(X_combined_)
                X_combined_ = X_combined_.astype(np.float32)
            elif self.FEATURE_SCALING==2:
                self.SCALER = StandardScaler(with_mean=True, with_std=True).fit(X_combined_)
                X_combined_ = X_combined_.astype(np.float32)
        
        Nc_ = {}
        Fc_ = {}
        for speaker_id_ in X.keys():
            for split_id_ in X[speaker_id_].keys():
                fv_ = None
                del fv_
                
                # Load features
                fv_ = X[speaker_id_][split_id_]
                
                # Compute a posteriori log-likelihood
                log_lkdh_ = self.BACKGROUND_MODEL.predict_proba(fv_)
                
                # Compute a posteriori normalized probability
                amax_ = np.max(log_lkdh_, axis=0)
                norm_log_lkhd_ = np.subtract(log_lkdh_, np.repeat(np.array(amax_, ndmin=2), np.shape(log_lkdh_)[0], axis=0))
                log_lkhd_sum_ = amax_ + np.log(np.sum(np.exp(norm_log_lkhd_), axis=0))
                gamma_ = np.exp(np.subtract(log_lkdh_, np.repeat(np.array(log_lkhd_sum_, ndmin=2), np.shape(log_lkdh_)[0], axis=0)))
                
                # Compute Baum-Welch statistics
                n_ = np.sum(gamma_, axis=0) # zeroeth-order
                f_ = np.multiply(fv_.T, gamma_) # first-order
                
                Nc_[split_id_] = np.expand_dims(np.aray(n_, ndmin=2), axis=0)
                Fc_[split_id_] = np.expand_dims(f_, axis=1)
        
        