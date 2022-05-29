#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:21:17 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
from sklearn.mixture import GaussianMixture
import numpy as np
from lib.models.GMM_UBM.speaker_adaptation import SpeakerAdaptation
import pickle
import os


class GaussianBackground:
    NCOMP = 0
    MODEL_DIR = ''
    BACKGROUND_MODEL = None
    
    def __init__(self, model_dir, num_mixtures=128):
        self.NCOMP = num_mixtures
        self.MODEL_DIR = model_dir
    
    
    def train_ubm(self, X, cov_type='diag', max_iterations=100, num_init=3, verbose=1):
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
        verbose : int, optional
            Flag to indicate whether to print GMM training outputs. The default is 1.

        Returns
        -------
        None.

        '''
        ubm_fName = self.MODEL_DIR + '/ubm.pkl'
        if not os.path.exists(ubm_fName):
            X_combined = np.empty([], dtype=np.float32)
            for speaker_id_ in X.keys():
                if np.size(X_combined)<=1:
                    X_combined = X[speaker_id_]
                else:
                    X_combined = np.append(X_combined, X[speaker_id_], axis=0)
            print(f'X_combined={np.shape(X_combined)}')
            
            self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbose)
            self.BACKGROUND_MODEL.fit(X_combined)
            
            with open(ubm_fName, 'wb') as f:
                pickle.dump(self.BACKGROUND_MODEL, f, pickle.HIGHEST_PROTOCOL)
        else:
            print('The GMM Universal Background Model is already available')
        
        return
    
    
    def speaker_adaptation(self, X_ENR, cov_type='diag', use_adapt_w_cov=True):
        '''
        Adaptation of the UBM model for each enrolling speaker.

        Parameters
        ----------
        X_ENR : dict
            Dictionary containing the speaker-wise enrollment data.
        cov_type : str, optional
            Type of GMM covariance to be used for training. The default is 'diag'.
        use_adapt_w_cov : bool, optional
            Flag indicating whether to use speaker specific covariacne matrix
            during adaptation. The default is True.

        Returns
        -------
        None.

        '''
        if not self.BACKGROUND_MODEL:
            ubm_fName_ = self.MODEL_DIR + '/ubm.pkl'
            if not os.path.exists(ubm_fName_):
                print('Background model does not exist')
                return
            with open(ubm_fName_, 'rb') as f_:
                self.BACKGROUND_MODEL = pickle.load(f_)
        
        # print(f'{X_ENR.keys()}')
        for speaker_id_ in X_ENR.keys():
            speaker_opDir_ = self.MODEL_DIR + '/' + speaker_id_ + '/'
            if not os.path.exists(speaker_opDir_):
                os.makedirs(speaker_opDir_)
            speaker_model_fName_ = speaker_opDir_ + '/adapted_ubm.pkl'
            if os.path.exists(speaker_model_fName_):
                print(f'Adapted GMM model already available for speaker={speaker_id_}')
                continue

            fv_ = X_ENR[speaker_id_]
            adapt_ = SpeakerAdaptation().adapt_ubm(fv_.T, self.BACKGROUND_MODEL, use_adapt_w_cov)
            adapted_gmm_ = None
            adapted_gmm_ = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type)
            adapted_gmm_.means_ = adapt_['means']
            adapted_gmm_.weights_ = adapt_['weights']
            adapted_gmm_.covariances_ = adapt_['covariances']
            adapted_gmm_.precisions_ = adapt_['precisions']
            adapted_gmm_.precisions_cholesky_ = adapt_['precisions_cholesky']
            
            with open(speaker_model_fName_, 'wb') as f_:
                pickle.dump(adapted_gmm_, f_, pickle.HIGHEST_PROTOCOL)
            print(f'Adapted GMM model saved for speaker={speaker_id_}')
            
            
    def get_speaker_scores(self, X_TEST):
        '''
        Compute test speaker scores against all enrollment speaker models.

        Parameters
        ----------
        X_TEST : dict
            Dictionary containing the speaker-wise test data.

        Returns
        -------
        Scores_ : 2D array
            AN (N x N) array consisting of scores for each test speaer against
            each enrollment speaker. N is the number of speakers.

        '''
        Scores_ = np.zeros((len(X_TEST), len(X_TEST)))
        i = 0
        for speaker_id_i_ in X_TEST.keys():
            j = 0
            speaker_models_ = next(os.walk(self.MODEL_DIR))[1]
            for speaker_id_j_ in speaker_models_:
                speaker_opDir_ = self.MODEL_DIR + '/' + speaker_id_j_ + '/'
                speaker_model_fName_ = speaker_opDir_ + '/adapted_ubm.pkl'
                if not os.path.exists(speaker_model_fName_):
                    print(f'GMM model does not exist for speaker={speaker_id_j_}')
                    continue
                with open(speaker_model_fName_, 'rb') as f_:
                    speaker_model_ = pickle.load(f_)
                
                score_ = speaker_model_.score(X_TEST[speaker_id_j_])
                Scores_[i,j] = score_
                j += 1
            i += 1
            
        return Scores_
        