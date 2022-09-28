#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:31:50 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
from sklearn.mixture import GaussianMixture
import numpy as np
from lib.models.GMM_UBM.speaker_adaptation import SpeakerAdaptation
from lib.models.IVector import sidekit_util
from lib.models.IVector.statserver import StatServer
import sidekit
import pickle
import os
from sklearn.preprocessing import StandardScaler
from lib.metrics.performance_metrics import PerformanceMetrics
import psutil
import sys


class IVector:
    NCOMP = 0
    UBM_DIR = ''
    MODEL_DIR = ''
    BACKGROUND_MODEL = None
    FEATURE_SCALING = 0
    SCALER = None
    N_BATCHES = 0
    
    
    def __init__(self, ubm_dir, model_dir, opDir, num_mixtures=128, feat_scaling=0):
        self.UBM_DIR = ubm_dir
        self.MODEL_DIR = model_dir
        self.OPDIR = opDir
        self.NCOMP = num_mixtures
        self.FEATURE_SCALING = feat_scaling
        self.N_BATCHES = 1
    
    
    def train_ubm(self, X, ram_mem_req, cov_type='diag', max_iterations=100, num_init=1, verbosity=1):
        '''
        Training the GMM Universal Background Model.

        Parameters
        ----------
        X_combined_ : array
            All feature arrays.
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

        ram_mem_avail_ = psutil.virtual_memory().available >> 20 # in MB; >> 30 in GB
        print(f'Available RAM: {ram_mem_avail_} MB')
        print(f'RAM required: {ram_mem_req} MB')

        if (ram_mem_req>ram_mem_avail_) or (ram_mem_req>self.MEM_LIMIT):
            self.N_BATCHES = int(ram_mem_req/self.MEM_LIMIT) # int(np.ceil(ram_mem_req_/(0.1*ram_mem_avail_)))
            '''
            Batch-wise training GMM-UBM
            '''
            print(f'Training GMM-UBM in a batch-wise manner. # Batches={self.N_BATCHES}')
            
            speaker_ids_ = [spId_ for spId_ in X.keys()]
            np.random.shuffle(speaker_ids_)
            
            for batch_i_ in range(self.N_BATCHES):
                if speaker_ids_==[]:
                    break
                X_batch_ = np.empty([])
                data_size_ = 0
                while data_size_<self.MEM_LIMIT:
                    spId_ = speaker_ids_.pop()
                    for split_id_ in X[spId_].keys():
                        if np.size(X_batch_)<=1:
                            X_batch_ = X[spId_][split_id_]
                        else:
                            X_batch_ = np.append(X_batch_, X[spId_][split_id_], axis=0)
                    data_size_ = (np.size(X_batch_)*8 + np.shape(X_batch_)[0]*self.NCOMP*8) >> 20
                print(f'Batch={batch_i_} X_batch={np.shape(X_batch_)} data_size={data_size_}')
                
                if batch_i_==0:
                    training_success_ = False
                    reg_covar_ = 1e-6
                    self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbosity, reg_covar=reg_covar_)
                    self.BACKGROUND_MODEL.fit(X_batch_)

                    print(f'Batch: {batch_i_+1} model trained')
                else:
                    adapt_ = None
                    del adapt_
                    adapt_ = SpeakerAdaptation().adapt_ubm(X_batch_.T, self.BACKGROUND_MODEL, use_adapt_w_cov=False)
                    self.BACKGROUND_MODEL.means_ = adapt_['means']
                    self.BACKGROUND_MODEL.weights_ = adapt_['weights']
                    self.BACKGROUND_MODEL.covariances_ = adapt_['covariances']
                    self.BACKGROUND_MODEL.precisions_ = adapt_['precisions']
                    self.BACKGROUND_MODEL.precisions_cholesky_ = adapt_['precisions_cholesky']
                    print(f'Batch: {batch_i_+1} model updated')
        
        else:
            X_combined_ = np.empty([], dtype=np.float32)
            speaker_count_ = 0
            for speaker_id_ in X.keys():
                split_count_ = 0
                for split_id_ in X[speaker_id_].keys():
                    if np.size(X_combined_)<=1:
                        X_combined_ = np.array(X[speaker_id_][split_id_], dtype=np.float32)
                    else:
                        X_combined_ = np.append(X_combined_, np.array(X[speaker_id_][split_id_], dtype=np.float32), axis=0)
                    split_count_ += 1
                    # print(f'Splits per speaker: ({split_count_}/{len(X[speaker_id_].keys())})', end='\r', flush=True)
                # print('')
                speaker_count_ += 1
                print(f'Speaker-wise data combination: ({speaker_count_}/{len(X.keys())})', end='\r', flush=True)
            print('')
            print(f'Development data shape={np.shape(X_combined_)} data_type={X_combined_.dtype}')


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
        

    def write_list_to_txt(self, txt_file, tmp_list):
        txt_data = ",".join(tmp_list)
        with open(txt_file, "w") as f:
            f.write(txt_data)


        
    def train_t_matrix(self, PARAMS):
        
        if not self.BACKGROUND_MODEL:
            ubm_fName_ = self.UBM_DIR + '/ubm.pkl'
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
        print('Background model loaded')

        leftlist, rightlist = sidekit_util.id_map_list(PARAMS['feat_dir'])
        self.write_list_to_txt(os.path.join(PARAMS['output_dir'], 'left_list.txt'), leftlist)
        self.write_list_to_txt(os.path.join(PARAMS['output_dir'], 'right_list.txt'), rightlist)
    
        tv_idmap = sidekit_util.get_id_map(leftlist, rightlist)
        
        stat_file = PARAMS['output_dir'] + '/stat_ubm_tv_' + str(PARAMS['distrib_nb']) + '.h5'
        self.train_tv_matrix(PARAMS, self.BACKGROUND_MODEL, stat_file)
        
        tv_stat = StatServer.read_subset(stat_file, tv_idmap)
        tv_mean, tv, _, __, tv_sigma = tv_stat.factor_analysis(
            rank_f=int(PARAMS['rank_TV']),
            rank_g=0,
            rank_h=None,
            re_estimate_residual=False,
            it_nb=(int(PARAMS['tv_iteration']), 0, 0),
            min_div=True,
            ubm=self.BACKGROUND_MODEL,
            batch_size=int(PARAMS['batch_size']),
            num_thread=int(PARAMS['nbThread']),
            save_partial=PARAMS['tv_path'],
            )
    
    
    def train_tv_matrix(self, PARAMS, ubm, stat_path):
        enroll_stat = sidekit_util.adapt_stats_with_feat_dir(PARAMS['feat_dir'], ubm, int(PARAMS['nbThread']))
        enroll_stat.write(stat_path)
