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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lib.metrics.performance_metrics import PerformanceMetrics
import psutil
import sys


class GaussianBackground:
    NCOMP = 0
    MODEL_DIR = ''
    BACKGROUND_MODEL = None
    FEATURE_SCALING = 0
    SCALER = None
    
    
    def __init__(self, model_dir, num_mixtures=128, feat_scaling=0):
        self.NCOMP = num_mixtures
        self.MODEL_DIR = model_dir
        self.FEATURE_SCALING = feat_scaling
    
    
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
        X_combined_ = np.empty([], dtype=np.float32)
        for speaker_id_ in X.keys():
            for split_id_ in X[speaker_id_].keys():
                if np.size(X_combined_)<=1:
                    X_combined_ = X[speaker_id_][split_id_]
                else:
                    X_combined_ = np.append(X_combined_, X[speaker_id_][split_id_], axis=0)
        print(f'X_combined={np.shape(X_combined_)}')
        ram_mem_avail_ = psutil.virtual_memory().available
        feat_memsize_ = sys.getsizeof(X_combined_)
        print(f'ram_mem_avail_={ram_mem_avail_} feat_memsize_={feat_memsize_} ratio={np.round(feat_memsize_/ram_mem_avail_,4)}')
        random_sample_idx_ = np.random.choice(list(range(np.shape(X_combined_)[0])), size=10000, replace=False)
        X_combined_ = X_combined_[random_sample_idx_, :]
        
        ''' Feature Scaling '''
        if self.FEATURE_SCALING==1:
            self.SCALER = StandardScaler(with_mean=True, with_std=False).fit(X_combined_)
            X_combined_ = self.SCALER.transform(X_combined_)
        elif self.FEATURE_SCALING==2:
            self.SCALER = StandardScaler(with_mean=True, with_std=True).fit(X_combined_)
            X_combined_ = self.SCALER.transform(X_combined_)
        
        self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbose)
        self.BACKGROUND_MODEL.fit(X_combined_)
        
        ubm_fName = self.MODEL_DIR + '/ubm.pkl'
        with open(ubm_fName, 'wb') as f:
            pickle.dump(self.BACKGROUND_MODEL, f, pickle.HIGHEST_PROTOCOL)
        
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

            fv_ = np.empty([], dtype=np.float32)
            for split_id_ in X_ENR[speaker_id_]:
                if np.size(fv_)<=1:
                    fv_ = X_ENR[speaker_id_][split_id_]
                else:
                    fv_ = np.append(fv_, X_ENR[speaker_id_][split_id_], axis=0)
            
            ''' Feature Scaling '''
            if self.FEATURE_SCALING==1:
                fv_ = self.SCALER.transform(fv_)
            elif self.FEATURE_SCALING==2:
                fv_ = self.SCALER.transform(fv_)

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
            
            
    def perform_testing(self, opDir, opFileName, X_TEST=None, feat_info=None, dim=39, duration=None):
        '''
        Compute test speaker scores against all enrollment speaker models.

        Parameters
        ----------
        opDir : str
            Output path.
        opFileName : str
            Output file name.
        X_TEST : dict, optional
            Dictionary containing the speaker-wise test data.
        feat_info : dict, optional
            Dictionary containing info about feature paths.
        dim : int, optional
            Dimension of input feature.
        duration : str, optional
            Selection of which utterance duration to test. Default, tests all
            utterances.

        Returns
        -------
        Scores_ : 2D array
            AN (N x N) array consisting of scores for each test speaer against
            each enrollment speaker. N is the number of speakers.

        '''
        if duration:
            score_fName_ = opDir + '/' + opFileName.split('.')[0] + '_' + str(duration) + 's.pkl'
        else:
            score_fName_ = opDir + '/' + opFileName.split('.')[0] + '.pkl'
        if not os.path.exists(score_fName_):
            if not self.BACKGROUND_MODEL:
                ubm_fName_ = self.MODEL_DIR + '/ubm.pkl'
                if not os.path.exists(ubm_fName_):
                    print('Background model does not exist')
                    return
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)
    
            enrolled_speakers_ = next(os.walk(self.MODEL_DIR))[1]
            Scores_ = {}
            true_lab_ = []
            pred_lab_ = []

            
            ''' Loading the speaker models '''
            enr_speaker_model_ = {}
            for enr_j_ in enrolled_speakers_:
                enr_speaker_opDir_ = self.MODEL_DIR + '/' + enr_j_ + '/'
                enr_speaker_model_fName_ = enr_speaker_opDir_ + '/adapted_ubm.pkl'
                if not os.path.exists(enr_speaker_model_fName_):
                    print(f'GMM model does not exist for speaker={enr_j_}')
                    continue
                with open(enr_speaker_model_fName_, 'rb') as f_:
                    enr_speaker_model_[enr_j_] = pickle.load(f_)

            
            if not X_TEST:
                ''' 
                Testing every test utterance one by one 
                '''
                split_count_ = 0
                for split_id_ in feat_info.keys():
                    split_count_ += 1
                    '''
                    Checking duration of utterance
                    '''
                    if duration:
                        if not split_id_.split('_')[-2]==str(duration):
                            continue
    
                    speaker_id_ = feat_info[split_id_]['speaker_id']
                    feature_path_ = feat_info[split_id_]['file_path']
                    fv_ = None
                    del fv_
                    fv_ = np.load(feature_path_, allow_pickle=True)
                    # The feature vectors must be stored as individual rows in the 2D array
                    if dim:
                        if np.shape(fv_)[0]==dim:
                            fv_ = fv_.T
                    elif np.shape(fv_)[1]>np.shape(fv_)[0]:
                        fv_ = fv_.T
    
                    ''' Feature Scaling '''
                    if self.FEATURE_SCALING==1:
                        fv_ = self.SCALER.transform(fv_)
                    elif self.FEATURE_SCALING==2:
                        fv_ = self.SCALER.transform(fv_)
                        
                    all_enr_speaker_scores_ = None
                    del all_enr_speaker_scores_
                    all_enr_speaker_scores_ = {}
                    max_score_ = None
                    del max_score_
                    max_score_ = -9999999
                    matched_speaker_id_ = ''
                    for enr_j_ in enr_speaker_model_:                        
                        all_enr_speaker_scores_[enr_j_] = enr_speaker_model_[enr_j_].score(fv_) - self.BACKGROUND_MODEL.score(fv_)
                        if all_enr_speaker_scores_[enr_j_]>max_score_:
                            max_score_ = all_enr_speaker_scores_[enr_j_]
                            matched_speaker_id_ = enr_j_
                    Scores_[split_id_] = {
                        'speaker_id':speaker_id_, 
                        'enr_speaker_scores': all_enr_speaker_scores_, 
                        'matched_speaker':matched_speaker_id_,
                        }
                    
                    ''' Displaying progress '''
                    # lab_true_ = np.squeeze(np.where(np.array(enrolled_speakers_)==str(speaker_id_)))
                    # lab_pred_ = np.squeeze(np.where(np.array(enrolled_speakers_)==str(matched_speaker_id_)))
                    # true_lab_.append(lab_true_)
                    # pred_lab_.append(lab_pred_)
                    true_lab_.append(str(speaker_id_))
                    pred_lab_.append(str(matched_speaker_id_))
                    accuracy_ = np.round(np.sum(np.array(true_lab_)==np.array(pred_lab_))/np.size(true_lab_)*100,2)
                    # print(f'\t{split_id_} splits=({split_count_}/{len(feat_info)}) true=({speaker_id_}|{lab_true_}) pred=({matched_speaker_id_}|{lab_pred_}) max score={np.round(max_score_,4)} accuracy={accuracy_}%')
                    print(f'\t{split_id_} splits=({split_count_}/{len(feat_info)}) true=({speaker_id_}) pred=({matched_speaker_id_}) max score={np.round(max_score_,4)} accuracy={accuracy_}%')


            if not feat_info:
                ''' 
                Testing each sub-utterance with each speaker model 
                '''
                speaker_count_ = 0
                num_speakers_ = len(X_TEST.keys())

                for speaker_id_i_ in X_TEST.keys():
                    speaker_count_ += 1
                    split_count_ = 0
                    num_splits_ = len(X_TEST[speaker_id_i_].keys())
                    for split_id_ in X_TEST[speaker_id_i_].keys():
                        split_count_ += 1
                        
                        '''
                        Checking duration of utterance
                        '''
                        if duration:
                            if not split_id_.split('_')[-2]==str(duration):
                                continue
                        
                        fv_ = None
                        del fv_
                        fv_ = X_TEST[speaker_id_i_][split_id_]
                    
                        ''' Feature Scaling '''
                        if self.FEATURE_SCALING==1:
                            fv_ = self.SCALER.transform(fv_)
                        elif self.FEATURE_SCALING==2:
                            fv_ = self.SCALER.transform(fv_)
                            
                        all_enr_speaker_scores_ = {}
                        max_score_ = -9999999
                        matched_speaker_id_ = ''
                        for enr_j_ in enr_speaker_model_:                        
                            all_enr_speaker_scores_[enr_j_] = enr_speaker_model_[enr_j_].score(fv_) #- self.BACKGROUND_MODEL.score(fv_)
                            if all_enr_speaker_scores_[enr_j_]>max_score_:
                                max_score_ = all_enr_speaker_scores_[enr_j_]
                                matched_speaker_id_ = enr_j_
                        Scores_[split_id_] = {'speaker_id':speaker_id_i_, 'enr_speaker_scores': all_enr_speaker_scores_, 'matched_speaker':matched_speaker_id_}
                        
                        ''' Displaying progress '''
                        lab_true_ = np.squeeze(np.where(np.array(enrolled_speakers_)==str(speaker_id_i_)))
                        lab_pred_ = np.squeeze(np.where(np.array(enrolled_speakers_)==str(matched_speaker_id_)))
                        true_lab_.append(lab_true_)
                        pred_lab_.append(lab_pred_)
                        accuracy_ = np.round(np.sum(np.array(true_lab_)==np.array(pred_lab_))/np.size(true_lab_)*100,2)
                        print(f'\t{split_id_} speakers=({speaker_count_}/{num_speakers_}) splits=({split_count_}/{num_splits_}) true={speaker_id_i_} ({lab_true_}) pred={matched_speaker_id_} ({lab_pred_}) max score={np.round(max_score_,4)} accuracy={accuracy_}%')

            with open(score_fName_, 'wb') as f_:
                pickle.dump(Scores_, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(score_fName_, 'rb') as f_:
                Scores_ = pickle.load(f_)
        
        return Scores_
    
    
    def evaluate_performance(self, res):
        '''
        Compute performance metrics.

        Parameters
        ----------
        res : dict
            Dictionary containing the sub-utterance wise scores.

        Returns
        -------
        Metrics_ : dict
            Disctionary containg the various evaluation metrics:
                accuracy, precision, recall, f1-score, eer

        '''
        all_speaker_id_ = next(os.walk(self.MODEL_DIR))[1]
        groundtruth_label_ = []
        ptd_labels_ = []
        groundtruth_scores_ = np.empty([])
        predicted_scores_ = np.empty([])
        for split_id_ in res.keys():
            true_speaker_id_ = res[split_id_]['speaker_id']
            pred_speaker_id_ = res[split_id_]['matched_speaker']
            true_label_ = np.squeeze(np.where(np.array(all_speaker_id_)==str(true_speaker_id_)))
            groundtruth_label_.append(true_label_)
            ptd_labels_.append(np.squeeze(np.where(np.array(all_speaker_id_)==str(pred_speaker_id_))))
            gt_score_ = np.zeros((1,np.size(all_speaker_id_)))
            gt_score_[0,true_label_] = 1
            ptd_scores_ = np.zeros((1,np.size(all_speaker_id_)))
            for speaker_id_ in res[split_id_]['enr_speaker_scores'].keys():
                lab_ = np.squeeze(np.where(np.array(all_speaker_id_)==str(speaker_id_)))
                ptd_scores_[0,lab_] = res[split_id_]['enr_speaker_scores'][speaker_id_]
            if np.size(groundtruth_scores_)<=1:
                groundtruth_scores_ = gt_score_
                predicted_scores_ = ptd_scores_
            else:
                groundtruth_scores_ = np.append(groundtruth_scores_, gt_score_, axis=0)
                predicted_scores_ = np.append(predicted_scores_, ptd_scores_, axis=0)
        
        label_list = list(range(np.size(all_speaker_id_)))
        confmat_, precision_, recall_, fscore_ = PerformanceMetrics().compute_identification_performance(groundtruth_label_, ptd_labels_, label_list)
        acc_ = np.sum(np.diag(confmat_))/np.sum(confmat_)

        FPR_, TPR_, EER_, EER_thresh_ = PerformanceMetrics().compute_eer(groundtruth_scores_.flatten(), predicted_scores_.flatten())
                
        Metrics_ = {
            'accuracy': acc_,
            'precision': precision_,
            'recall': recall_,
            'f1-score': fscore_,
            'fpr': FPR_,
            'tpr': TPR_,
            'eer': EER_,
            'eer_threshold': EER_thresh_,
            }
        
        return Metrics_