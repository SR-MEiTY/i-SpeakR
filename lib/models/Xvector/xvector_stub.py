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
# from sklearn.preprocessing import StandardScaler
# from lib.metrics.performance_metrics import PerformanceMetrics
import psutil
# import sys
import time
from lib.models.Xvector.gplda import GPLDA_computation, compute_gplda_score
import csv
from lib.metrics.performance_metrics import PerformanceMetrics
from joblib import Parallel, delayed



class XVector:
    NCOMP = 0
    UBM_DIR = ''
    MODEL_DIR = ''
    BACKGROUND_MODEL = None
    FEATURE_SCALING = 0
    SCALER = None
    N_BATCHES = 0
    
    
    def __init__(self, model_dir, opDir, xvec_dim=100, feat_scaling=0): #, lda=True, wccn=True):
        self.MODEL_DIR = model_dir
        self.OPDIR = opDir
        self.XVEC_DIM = xvec_dim
        self.FEATURE_SCALING = feat_scaling
        self.N_BATCHES = 1
        # self.LDA = lda
        # self.WCCN = wccn
    
    
    def save_features_required_format(self, dev_xvec_dir_, dev_xvec_dir_new_):
        subdirs = next(os.walk(dev_xvec_dir_))[1]
        # print(f'subdirs={subdirs}')
        for dir_i in subdirs:
            # print(f'dir_i={dir_i}')
            files = next(os.walk(dev_xvec_dir_+'/'+dir_i))[2]
            # files = librosa.util.find_files(dev_xvec_dir_+'/'+dir_i, ext=['npy'])
            # print(f'files={files}')
            X_vectors_ = np.empty([])
            speaker_id_ = ''
            for fl in files:
                speaker_id_ = fl.split('_')[0]
                fName = dev_xvec_dir_ + '/' + dir_i + '/' + fl #.split('/')[-1]
                # print(f'fName={fName}')
                xvec_temp_ = np.load(fName)
                if np.size(X_vectors_)<=1:
                    X_vectors_ = np.array(xvec_temp_, ndmin=2)
                else:
                    X_vectors_ = np.append(X_vectors_, np.array(xvec_temp_, ndmin=2), axis=0)
            print(f'speaker_id={dir_i} X_vectors_={X_vectors_.shape}')
            os.makedirs(dev_xvec_dir_new_+'/'+speaker_id_)
            xvec_fName_ = dev_xvec_dir_new_ + '/' + speaker_id_+ '/' + speaker_id_ + '.pkl'
            with open(xvec_fName_, 'wb') as f_:
                pickle.dump(X_vectors_, f_, pickle.HIGHEST_PROTOCOL)
        
    

    def cosine_kernal_scoring_on_spkr_factors(self, trn_y, tst_y):
        M_ =  trn_y
        F_ =  tst_y

        # Normalization        
        M_ = np.divide(M_, np.repeat(np.array(np.sqrt(np.sum(np.power(M_), axis=1))), ndmin=2).T, np.shape(M_)[1], axis=1)
        F_ = np.divide(F_, np.repeat(np.array(np.sqrt(np.sum(np.power(F_), axis=1))), ndmin=2).T, np.shape(F_)[1], axis=1)
        
        scores = np.inner(M_, F_)
        return scores



    def compute_expectations(self, s_, N_, n_feat_, T_mat_, TtimesInverseSSdiag_, I_, F_):
        time1 = time.process_time()
        Nc_ = np.repeat(np.array(np.tile(N_, [n_feat_]), ndmin=2), self.XVEC_DIM, axis=0)
        # print(f'I_={I_.shape} TtimesInverseSSdiag_={TtimesInverseSSdiag_.shape}, Nc_={Nc_.shape} T_mat_={T_mat_.shape}')
        L_ = I_ + (np.multiply(TtimesInverseSSdiag_, Nc_) @ T_mat_)
        # print(f'L_={L_.shape}')
        Linv_ = np.linalg.pinv(L_)
        # print(f'Linv_={Linv_[s_].shape}')
        Ey_ = Linv_ @ TtimesInverseSSdiag_ @ F_
        # print(f'Ey_={Ey_[s_].shape}')
        Eyy_ = Linv_ + Ey_ @ Ey_.T
        print(f'{s_} {np.round(time.process_time()-time1,2)}s\t\t\t')
        
        return s_, Linv_, Ey_, Eyy_




    
    def plda_training(self, X, model_dir, lda_dim=20, LDA=False, WCCN=False):
        '''
        Training PLDA model.

        Parameters
        ----------
        X : dict
            Dictionary containing speaker-wise data.
        model_dir : str
            Directory to store the PLDA model.

        Returns
        -------
        None.

        '''
        xvector_per_speaker_ = {}
        for speaker_id_ in X.keys():
            xvec_fName_ = model_dir + '/' + speaker_id_ + '/' + speaker_id_ + '.pkl'
            with open(xvec_fName_, 'rb') as f_:
                I_vectors_ = pickle.load(f_)
                xvector_per_speaker_[speaker_id_] = I_vectors_.T
                print(f'speaker={speaker_id_} xvector={np.shape(xvector_per_speaker_[speaker_id_])}')
        
        gpldaModel_, projectionMatrix_ = GPLDA_computation(xvector_per_speaker_, num_eigen_vectors=lda_dim, perform_LDA=LDA, perform_WCCN=WCCN, num_iter=50)
        
        return gpldaModel_, projectionMatrix_



    def perform_testing(self, enr_xvec_dir, tv_fName, classifier=None, opDir='', feat_info=None,  dim=None, test_key=None, duration=None):
        '''
        Test i-vector based speaker recognition system

        Parameters
        ----------
        enr_xvec_dir : str
            Enrollment i-vector directory.
        tv_fName : str
            Trained Total Variability matrix file.
        classifier : dict, optional
            G-PLDA model and projection matrix. The default is None.
        opDir : str, optional
            Path to dump output files. The default is ''.
        feat_info : dict, optional
            Dictionary containing speaker-wise feature info. The default is None.
        dim : int, optional
            Dimensions of input features. The default is None.
        test_key : str, optional
            Path to test key csv file. The default is None.
        duration : float, optional
            Test utterance chop duation. The default is None.

        Returns
        -------
        scores_ : dict
            Scores computed for all test utterances.

        '''
        if duration:
            score_fName_ = opDir + '/Test_Scores_' + str(duration) + 's.pkl'
        else:
            score_fName_ = opDir + '/Test_Scores.pkl'
    
        score_fName_ = opDir + '/Test_Scores.pkl'
        print(f'scores={os.path.exists(score_fName_)}')
        if not os.path.exists(score_fName_):    
            scores_ = {}
            # true_lab_ = []
            # pred_lab_ = []
            
            '''
            Loading Total Variability matrix
            '''
            with open(tv_fName, 'rb') as f_:
                TV_ = pickle.load(f_)
                TV_ = TV_['tv_mat']
            
            ''' Loading the speaker models '''
            enrolled_speakers_ = next(os.walk(enr_xvec_dir))[1]
            enr_xvectors_ = {}
            for speaker_id_ in enrolled_speakers_:
                xvec_fName_ = enr_xvec_dir + '/' + speaker_id_ + '/' + speaker_id_ + '.pkl'
                with open(xvec_fName_, 'rb') as f_:
                    I_vectors_ = pickle.load(f_)
                    enr_xvectors_[speaker_id_] = I_vectors_.T
                                
            # confusion_matrix_ = np.zeros((len(enrolled_speakers_), len(enrolled_speakers_)))
            # match_count_ = np.zeros(len(enrolled_speakers_))
            
            '''
            Testing every test utterance one by one 
            '''
            total_splits_ = 0
            
            speaker_id_list_ = []
            for split_id_ in feat_info.keys():
                if not split_id_.split('_')[-2]=='x':
                    split_dur_ = int(split_id_.split('_')[-2])
                else:
                    split_dur_ = duration
                if duration:
                    if not split_dur_==duration:
                        continue
                speaker_id_list_.append(feat_info[split_id_]['speaker_id'])
                total_splits_ += 1
                
            output_fName_ = opDir + '/' + test_key.split('/')[-1].split('.')[0] + '_predictions.csv'
            with open(output_fName_, 'a+', encoding='utf8') as fid_:
                writer_ = csv.writer(fid_)
                writer_.writerow([
                    'utterance_id',
                    'speaker_id', 
                    'score', 
                    ])
                
            split_count_ = 0
            with open(test_key, 'r' ) as test_meta_info_:
                reader_ = csv.DictReader(test_meta_info_)
                for row_ in reader_:
                    split_id_ = row_['split_id']
                    # print(row_['cohorts'])
                    cohort_speakers_ = row_['cohorts'].split('|')
                    # print(f'cohort_speakers_={cohort_speakers_}')
                            
                    split_count_ += 1    
                    speaker_id_ = feat_info[split_id_]['speaker_id']
                    test_xvec_opDir_ = opDir + '/xvectors/' + speaker_id_ + '/'
                    if not os.path.exists(test_xvec_opDir_):
                        os.makedirs(test_xvec_opDir_)
                        
                    test_xvec_fName_ = test_xvec_opDir_ + '/' + split_id_ + '.pkl'
                    test_xvec_ = np.empty([])
                    if not os.path.exists(test_xvec_fName_):
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
                        if self.FEATURE_SCALING>0:
                            fv_ = self.SCALER.transform(fv_)
                            fv_ = fv_.astype(np.float32)
                        
                        test_xvec_ = self.compute_xvector(fv_, TV_)
                        with open(test_xvec_fName_, 'wb') as xvec_fid_:
                            pickle.dump(test_xvec_, xvec_fid_, pickle.HIGHEST_PROTOCOL)
                        # print(f'Test {split_id_} xvector saved')
                    else:
                        with open(test_xvec_fName_, 'rb') as xvec_fid_:
                            test_xvec_ = pickle.load(xvec_fid_)
                        # print(f'Test {split_id_} xvector loaded')

                    print(f"GPLDA: {classifier['gplda_model'].keys()}")
                    test_xvec_ = classifier['projection_matrix'] @ test_xvec_                    
                    print(f'test_xvec_={np.shape(test_xvec_)}')

                    gplda_scores_ = {}
                    for index_i_ in enr_xvectors_.keys():
                        if index_i_ in cohort_speakers_:
                            
                            # spk_scores_ = []
                            # for utter_i_ in range(enr_xvectors_[index_i_].shape[1]):
                            #     enr_xvec_ = enr_xvectors_[index_i_][:,utter_i_]
                            #     if (self.LDA) or (self.WCCN):
                            #         enr_xvec_ = classifier['projection_matrix'] @ enr_xvec_
                            #     spk_scores_.append(compute_gplda_score(classifier['gplda_model'], enr_xvec_, test_xvec_))
                            # gplda_scores_[index_i_] = np.mean(spk_scores_)
                            

                            enr_xvec_ = classifier['projection_matrix'] @ enr_xvectors_[index_i_]
                            mean_enr_xvec_ = np.mean(enr_xvec_, axis=1)
                            gplda_scores_[index_i_] = compute_gplda_score(classifier['gplda_model'], mean_enr_xvec_, test_xvec_)
                            # print(f'{index_i_} gplda_scores_={np.shape(gplda_scores_[index_i_])}')
                    
                    # import sys
                    # sys.exit(0)
                    
                    scores_[split_id_] = {
                        'speaker_id':speaker_id_, 
                        'gplda_scores': gplda_scores_, 
                        'enrolled_speakers': enrolled_speakers_,
                        }

                    for cohort_id_ in gplda_scores_.keys():
                        output_fName_ = opDir+'/'+test_key.split('/')[-1].split('.')[0]+'_predictions.csv'
                        with open(output_fName_, 'a+', encoding='utf8') as fid_:
                            writer_ = csv.writer(fid_)
                            writer_.writerow([
                                row_['utterance_id'],
                                cohort_id_, 
                                gplda_scores_[cohort_id_], 
                                ])

                    '''
                    Displaying progress
                    '''
                    print(f'\t{split_id_}', end='\t', flush=True)
                    print(f'splits=({split_count_}/{total_splits_})', end='\t', flush=True)
                    print(f'true=({speaker_id_})', end='\t', flush=True)
                    print(f'cohort scores={gplda_scores_}', end='\n', flush=True)
                    
            '''
            Saving the scores for the selected duration
            '''
            with open(score_fName_, 'wb') as f_:
                pickle.dump(scores_, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(score_fName_, 'rb') as f_:
                scores_ = pickle.load(f_)
        
        return scores_



    def evaluate_performance(self, res, opDir, duration):
        '''
        Compute performance metrics.

        Parameters
        ----------
        res : dict
            Dictionary containing the sub-utterance wise scores.
        opDir : str
            Output path.
        duration : int
            Duration of the testing utterance.

        Returns
        -------
        metrics_ : dict
            Disctionary containg the various evaluation metrics:
                accuracy, precision, recall, f1-score, eer

        '''
        groundtruth_scores_ = np.empty([])
        predicted_scores_ = np.empty([])
        for split_id_ in res.keys():
            true_speaker_id_ = res[split_id_]['speaker_id']
            if true_speaker_id_=='0000':
                continue
            cohort_scores_ = res[split_id_]['gplda_scores']
            
            gt_score_ = []
            ptd_scores_ = []
            for cohort_id_ in cohort_scores_.keys():
                if cohort_id_==true_speaker_id_:
                    gt_score_.append(1)
                else:
                    gt_score_.append(0)
                ptd_scores_.append(cohort_scores_[cohort_id_])
                
            if np.size(groundtruth_scores_)<=1:
                groundtruth_scores_ = np.array(gt_score_)
                predicted_scores_ = np.array(ptd_scores_)
            else:
                groundtruth_scores_ = np.append(groundtruth_scores_, np.array(gt_score_), axis=0)
                predicted_scores_ = np.append(predicted_scores_, np.array(ptd_scores_), axis=0)
                
        if not np.sum(groundtruth_scores_)==0:            
            if not np.size(groundtruth_scores_)<=1:
                FPR_, TPR_, EER_, EER_thresh_ = PerformanceMetrics().compute_eer(groundtruth_scores_, predicted_scores_)
            else:
                FPR_ = np.empty([])
                TPR_ = np.empty([])
                EER_ = -1
                EER_thresh_ = np.empty([])                
        else:
            FPR_ = np.empty([])
            TPR_ = np.empty([])
            EER_ = -1
            EER_thresh_ = np.empty([])
                
        metrics_ = {
            'fpr': FPR_,
            'tpr': TPR_,
            'eer': EER_,
            'eer_threshold': EER_thresh_,
            }
            
        print(f'\n\nUtterance duration: {duration}s:\n__________________________________________')
        print(f"\tEER: {np.round(np.mean(metrics_['eer'])*100,2)}")

        with open(opDir+'/Performance.txt', 'a+') as f_:
            f_.write(f'Utterance duration: {duration}s:\n__________________________________________\n')
            f_.write(f"\tEER: {np.round(np.mean(metrics_['eer'])*100,2)}\n\n")
        
        return metrics_