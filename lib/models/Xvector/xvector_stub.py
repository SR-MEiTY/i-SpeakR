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
import torch
from lib.models.Xvector.x_vector_Indian_LID import X_vector
from tqdm import tqdm


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
        

        
    def compute_x_vectors(self, feat_path, model_dir, opDir, data_set, test=False):
        input_dim = 39
        num_classes = 50
        xvec_model_ = X_vector(input_dim, num_classes).to('cuda')
        xvec_model_.load_state_dict(torch.load(model_dir + '/xvector_model'))
        xvec_model_.eval()
        
        feat_files_ = next(os.walk(feat_path+'/'+data_set+'/'))[2]
        all_feat_ = {}
        for i_ in tqdm(range(len(feat_files_))):
            fName_ = feat_files_[i_]
            if not test:
                speaker_id_ = fName_.split('_')[0]
            else:
                speaker_id_ = fName_.split('.')[0]
            fv_ = np.load(feat_path + '/' + data_set + '/' + fName_)
            fv_ = np.expand_dims(fv_.T, axis=0)
            features_ = torch.from_numpy(fv_)
            features_ = features_.to('cuda')
            _, x_vectors_ = xvec_model_(features_)
            x_vectors_ = x_vectors_.detach().cpu().numpy()
            x_vectors_ = np.array(x_vectors_, ndmin=2)

            if speaker_id_ in all_feat_.keys():
                all_feat_[speaker_id_] = np.append(all_feat_[speaker_id_], x_vectors_.T, axis=1)
            else:
                all_feat_[speaker_id_] = x_vectors_.T
        
        for speaker_id_ in all_feat_.keys():
            speaker_feat_path_ = opDir + '/' + data_set + '/' + speaker_id_ + '/'
            if not os.path.exists(speaker_feat_path_):
                os.makedirs(speaker_feat_path_)
            np.save(speaker_feat_path_+speaker_id_+'.npy', all_feat_[speaker_id_])
            # print(f'{data_set} {fName_} xvec={np.shape(all_feat_[speaker_id_])}')

        return all_feat_


    

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




    
    def plda_training(self, XVEC_dev, lda_dim=20, LDA=False, WCCN=False):
        '''
        Training PLDA model.

        Parameters
        ----------
        XVEC_dev : dict
            Dictionary containing speaker-wise data.

        Returns
        -------
        None.

        '''
        
        gpldaModel_, projectionMatrix_ = GPLDA_computation(XVEC_dev, num_eigen_vectors=lda_dim, perform_LDA=LDA, perform_WCCN=WCCN, num_iter=50)
        
        return gpldaModel_, projectionMatrix_



    # def perform_testing(self, XVEC_enr, XVEC_test, classifier=None, opDir='', feat_info=None,  dim=None, test_key=None, duration=None):
    #     '''
    #     Test i-vector based speaker recognition system

    #     Parameters
    #     ----------
    #     XVEC_enr : dict
    #         Enrollment x-vector dictionary.
    #     classifier : dict, optional
    #         G-PLDA model and projection matrix. The default is None.
    #     opDir : str, optional
    #         Path to dump output files. The default is ''.
    #     feat_info : dict, optional
    #         Dictionary containing speaker-wise feature info. The default is None.
    #     dim : int, optional
    #         Dimensions of input features. The default is None.
    #     test_key : str, optional
    #         Path to test key csv file. The default is None.
    #     duration : float, optional
    #         Test utterance chop duation. The default is None.

    #     Returns
    #     -------
    #     scores_ : dict
    #         Scores computed for all test utterances.

    #     '''
    #     if duration:
    #         score_fName_ = opDir + '/Test_Scores_' + str(duration) + 's.pkl'
    #     else:
    #         score_fName_ = opDir + '/Test_Scores.pkl'
    
    #     score_fName_ = opDir + '/Test_Scores.pkl'
    #     # print(f'scores={os.path.exists(score_fName_)}')
    #     if not os.path.exists(score_fName_):    
    #         scores_ = {}
    #         # true_lab_ = []
    #         # pred_lab_ = []
                        
    #         ''' Loading the speaker models '''
    #         enrolled_speakers_ = [key for key in XVEC_enr.keys()]
            
    #         '''
    #         Testing every test utterance one by one 
    #         '''
    #         total_splits_ = 0
            
    #         # speaker_id_list_ = []
    #         # for split_id_ in feat_info.keys():
    #         #     if not split_id_.split('_')[-2]=='x':
    #         #         split_dur_ = int(split_id_.split('_')[-2])
    #         #     else:
    #         #         split_dur_ = duration
    #         #     if duration:
    #         #         if not split_dur_==duration:
    #         #             continue
    #         #     speaker_id_list_.append(feat_info[split_id_]['speaker_id'])
    #         #     total_splits_ += 1
                
    #         output_fName_ = opDir + '/' + test_key.split('/')[-1].split('.')[0] + '_predictions.csv'
    #         with open(output_fName_, 'a+', encoding='utf8') as fid_:
    #             writer_ = csv.writer(fid_)
    #             writer_.writerow([
    #                 'utterance_id',
    #                 'speaker_id', 
    #                 'score', 
    #                 ])
                
    #         split_count_ = 0
    #         with open(test_key, 'r' ) as test_meta_info_:
    #             reader_ = csv.DictReader(test_meta_info_)
    #             for row_ in reader_:
    #                 split_id_ = row_['split_id']
    #                 # print(row_['cohorts'])
    #                 cohort_speakers_ = row_['cohorts'].split('|')
    #                 # print(f'cohort_speakers_={cohort_speakers_}')
                            
    #                 split_count_ += 1    
    #                 speaker_id_ = feat_info[split_id_]['speaker_id']
    #                 # test_xvec_opDir_ = opDir + '/xvectors/' + speaker_id_ + '/'
    #                 # if not os.path.exists(test_xvec_opDir_):
    #                 #     os.makedirs(test_xvec_opDir_)
                        
    #                 test_xvec_ = XVEC_test[split_id_]
    #                 # print(f'{split_id_} test_xvec_={np.shape(test_xvec_)}')
                        
    #                 # test_xvec_fName_ = test_xvec_opDir_ + '/' + split_id_ + '.pkl'
    #                 # test_xvec_ = np.empty([])
    #                 # if not os.path.exists(test_xvec_fName_):
    #                 #     feature_path_ = feat_info[split_id_]['file_path']
    #                 #     fv_ = None
    #                 #     del fv_
    #                 #     fv_ = np.load(feature_path_, allow_pickle=True)
    #                 #     # The feature vectors must be stored as individual rows in the 2D array
    #                 #     if dim:
    #                 #         if np.shape(fv_)[0]==dim:
    #                 #             fv_ = fv_.T
    #                 #     elif np.shape(fv_)[1]>np.shape(fv_)[0]:
    #                 #         fv_ = fv_.T
        
    #                 #     ''' Feature Scaling '''
    #                 #     if self.FEATURE_SCALING>0:
    #                 #         fv_ = self.SCALER.transform(fv_)
    #                 #         fv_ = fv_.astype(np.float32)
                        
    #                 #     model = X_vector(fv_.shape[1], len(enrolled_speakers_)).to('cpu')
    #                 #     PATH = '/DATA/jagabandhu/i-SpeakR_output/I-MSV/models/MFCC_x_vector/save_model/train_best_check_point_19_0.7463340487331152_0.06842262705783253'
    #                 #     model.load_state_dict(torch.load(PATH))
    #                 #     model.eval()

    #                 #     print(f'fv_ = {np.shape(fv_)}')
    #                 #     import sys
    #                 #     sys.exit(0)
    #                 #     features = torch.from_numpy(fv_)
    #                 #     features.requires_grad = False
    #                 #     pred_logits, test_x_vec_ = model(features)
    #                 #     # test_xvec_ = self.compute_xvector(fv_, TV_)

    #                 #     with open(test_xvec_fName_, 'wb') as xvec_fid_:
    #                 #         pickle.dump(test_xvec_, xvec_fid_, pickle.HIGHEST_PROTOCOL)
    #                 #     # print(f'Test {split_id_} xvector saved')
    #                 # else:
    #                 #     with open(test_xvec_fName_, 'rb') as xvec_fid_:
    #                 #         test_xvec_ = pickle.load(xvec_fid_)
    #                 #     # print(f'Test {split_id_} xvector loaded')

    #                 # print(f"GPLDA: {classifier['gplda_model'].keys()}")
    #                 test_xvec_ = classifier['projection_matrix'] @ test_xvec_                    
    #                 # print(f'test_xvec_={np.shape(test_xvec_)}')

    #                 gplda_scores_ = {}
    #                 for index_i_ in XVEC_enr.keys():
    #                     if index_i_ in cohort_speakers_:
                            
    #                         # spk_scores_ = []
    #                         # for utter_i_ in range(enr_xvectors_[index_i_].shape[1]):
    #                         #     enr_xvec_ = enr_xvectors_[index_i_][:,utter_i_]
    #                         #     if (self.LDA) or (self.WCCN):
    #                         #         enr_xvec_ = classifier['projection_matrix'] @ enr_xvec_
    #                         #     spk_scores_.append(compute_gplda_score(classifier['gplda_model'], enr_xvec_, test_xvec_))
    #                         # gplda_scores_[index_i_] = np.mean(spk_scores_)
                            

    #                         enr_xvec_ = classifier['projection_matrix'] @ XVEC_enr[index_i_]
    #                         # print(f'{index_i_} enr_xvec_={np.shape(enr_xvec_)}')
    #                         mean_enr_xvec_ = np.array(np.mean(enr_xvec_, axis=1), ndmin=2).T
    #                         # print(f'mean_enr_xvec_={np.shape(mean_enr_xvec_)} test_xvec_={np.shape(test_xvec_)}')
    #                         gplda_scores_[index_i_] = compute_gplda_score(classifier['gplda_model'], mean_enr_xvec_, test_xvec_)
    #                         # print(f'{index_i_} gplda_scores_={np.shape(gplda_scores_[index_i_])}')
    #                         # print(f'gplda_scores_={gplda_scores_[index_i_]}\n\n\n')
                    
    #                 # import sys
    #                 # sys.exit(0)
                    
    #                 scores_[split_id_] = {
    #                     'speaker_id':speaker_id_, 
    #                     'gplda_scores': gplda_scores_, 
    #                     'enrolled_speakers': enrolled_speakers_,
    #                     }

    #                 for cohort_id_ in gplda_scores_.keys():
    #                     output_fName_ = opDir+'/'+test_key.split('/')[-1].split('.')[0]+'_predictions.csv'
    #                     with open(output_fName_, 'a+', encoding='utf8') as fid_:
    #                         writer_ = csv.writer(fid_)
    #                         writer_.writerow([
    #                             row_['utterance_id'],
    #                             cohort_id_, 
    #                             gplda_scores_[cohort_id_], 
    #                             ])

    #                 '''
    #                 Displaying progress
    #                 '''
    #                 print(f'\t{split_id_}', end='\t', flush=True)
    #                 print(f'splits=({split_count_}/{total_splits_})', end='\t', flush=True)
    #                 print(f'true=({speaker_id_})', end='\t', flush=True)
    #                 print(f'cohort scores={gplda_scores_}', end='\n', flush=True)
                    
    #         '''
    #         Saving the scores for the selected duration
    #         '''
    #         with open(score_fName_, 'wb') as f_:
    #             pickle.dump(scores_, f_, pickle.HIGHEST_PROTOCOL)
    #     else:
    #         with open(score_fName_, 'rb') as f_:
    #             scores_ = pickle.load(f_)
        
    #     return scores_






    def perform_testing(self, XVEC_enr, XVEC_test, classifier=None, opDir='', feat_info=None,  dim=None, test_key=None, duration=None):
        '''
        Test x-vector based speaker recognition system

        Parameters
        ----------
        XVEC_enr : dict
            Enrollment x-vector dictionary.
        XVEC_test : dict
            Test x-vector dictionary.
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
        if not os.path.exists(score_fName_):    
            scores_ = {}
                        
            ''' Loading the speaker models '''
            enrolled_speakers_ = [key for key in XVEC_enr.keys()]
            
            '''
            Testing every test utterance one by one 
            '''
            total_splits_ = 0
            
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
                    cohort_speakers_ = row_['cohorts'].split('|')
                            
                    split_count_ += 1    
                    speaker_id_ = feat_info[split_id_]['speaker_id']
                        
                    test_xvec_ = XVEC_test[split_id_]
                    test_xvec_ = classifier['projection_matrix'] @ test_xvec_
                    # test_xvec_ = test_xvec_.T

                    gplda_scores_ = {}
                    for index_i_ in XVEC_enr.keys():
                        if index_i_ in cohort_speakers_:
                            enr_xvec_ = classifier['projection_matrix'] @ XVEC_enr[index_i_]
                            mean_enr_xvec_ = np.array(np.mean(enr_xvec_, axis=1), ndmin=2).T
                            # print(f'xvec enr={mean_enr_xvec_.shape} test={test_xvec_.shape}')
                            gplda_scores_[index_i_] = np.sum(compute_gplda_score(classifier['gplda_model'], mean_enr_xvec_, test_xvec_))
                            # print(f'gplda_scores_={len(gplda_scores_[index_i_])}')
                            
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