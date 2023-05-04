#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 18:16:13 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad

"""

from lib.feature_computation.compute_mfcc import MFCC
from lib.feature_computation.load_features import LoadFeatures
from lib.models.GMM_UBM.gaussian_background import GaussianBackground
from lib.models.IVector.ivector_extraction import IVector
from lib.models.Xvector.xvector_stub import XVector
from lib.models.Xvector.xvector_extraction import XvectorTraining
from lib.metrics.performance_metrics import PerformanceMetrics
import os
import pickle
import configparser
import argparse
import json



'''
X-Vector :: Speaker Verification System
'''
def xvector_sv(PARAMS, feat_info_):

    input_dim = 39
    num_classes = 50
    win_length = 78
    n_fft = 78
    batch_size = 10
    use_gpu = True
    num_epochs = 50
    
    XvectorTraining(input_dim, num_classes, win_length, n_fft, batch_size, use_gpu, num_epochs, PARAMS['feat_dir'], PARAMS['model_dir']).train_tdnn()


    XVec_ = XVector(
        model_dir=PARAMS['model_dir'], 
        opDir=PARAMS['output_dir'],
        xvec_dim=int(PARAMS['rank_tv']),
        feat_scaling=int(PARAMS['feature_scaling']),
        )
    
    xvector_opDir_ = PARAMS['output_dir'] + '/results/x_vector/'
    if not os.path.exists(xvector_opDir_):
        os.makedirs(xvector_opDir_)
        
    dev_key_ = PARAMS['dev_key'].split('/')[-1].split('.')[0]
    # dev_xvec_dir_ = xvector_opDir_ + '/' + dev_key_ + '/'
    if not os.path.exists(xvector_opDir_+'/'+dev_key_+'_xvectors.pkl'):
        print('\n\nComputing development X-Vectors:')
        XVEC_dev_ = XVec_.compute_x_vectors(PARAMS['feat_dir'], PARAMS['model_dir'], xvector_opDir_, dev_key_)
        with open(xvector_opDir_+'/'+dev_key_+'_xvectors.pkl', 'wb') as dev_fid_:
            pickle.dump(XVEC_dev_, dev_fid_, pickle.HIGHEST_PROTOCOL)
    else:
        with open(xvector_opDir_+'/'+dev_key_+'_xvectors.pkl', 'rb') as dev_fid_:
            XVEC_dev_ = pickle.load(dev_fid_)

    enr_key_ = PARAMS['enr_key'].split('/')[-1].split('.')[0]
    # enr_xvec_dir_ = xvector_opDir_ + '/' + enr_key_ + '/'
    if not os.path.exists(xvector_opDir_+'/'+enr_key_+'_xvectors.pkl'):
        print('\n\nComputing enrollment X-Vectors:')
        XVEC_enr_ = XVec_.compute_x_vectors(PARAMS['feat_dir'], PARAMS['model_dir'], xvector_opDir_, enr_key_)
        with open(xvector_opDir_+'/'+enr_key_+'_xvectors.pkl', 'wb') as enr_fid_:
            pickle.dump(XVEC_enr_, enr_fid_, pickle.HIGHEST_PROTOCOL)
    else:
        with open(xvector_opDir_+'/'+enr_key_+'_xvectors.pkl', 'rb') as enr_fid_:
            XVEC_enr_ = pickle.load(enr_fid_)

    test_key_ = PARAMS['test_key'].split('/')[-1].split('.')[0]
    # test_xvec_dir_ = xvector_opDir_ + '/' + test_key_ + '/'
    if not os.path.exists(xvector_opDir_+'/'+test_key_+'_xvectors.pkl'):
        print('\n\nComputing test X-Vectors:')
        XVEC_test_ = XVec_.compute_x_vectors(PARAMS['feat_dir'], PARAMS['model_dir'], xvector_opDir_, test_key_, test=True)
        with open(xvector_opDir_+'/'+test_key_+'_xvectors.pkl', 'wb') as test_fid_:
            pickle.dump(XVEC_test_, test_fid_, pickle.HIGHEST_PROTOCOL)
    else:
        with open(xvector_opDir_+'/'+test_key_+'_xvectors.pkl', 'rb') as test_fid_:
            XVEC_test_ = pickle.load(test_fid_)


    '''
    G-PLDA Training
    '''
    gplda_fName_ = PARAMS['model_dir'] + '/G_PLDA.pkl'
    if not os.path.exists(gplda_fName_):        
        gplda_model_, projection_matrix_ = XVec_.plda_training(XVEC_enr_, lda_dim=20, LDA=True, WCCN=True)
        gplda_classifier_ = {'gplda_model':gplda_model_, 'projection_matrix': projection_matrix_}
        with open(gplda_fName_, 'wb') as fid_:
            pickle.dump(gplda_classifier_, fid_, pickle.HIGHEST_PROTOCOL)
    else:
        print('G-PLDA model already trained')
        with open(gplda_fName_, 'rb') as fid_:
            gplda_classifier_ = pickle.load(fid_)



    ''' 
    Testing the trained models 
    '''
    print('Testing the trained models')
    test_opDir_ = xvector_opDir_ + '/Performance_' + PARAMS['test_key'].split('/')[-1].split('.')[0] + '/'
    if not os.path.exists(test_opDir_):
        os.makedirs(test_opDir_)
    FPR_ = {}
    TPR_ = {}
    test_chop = json.loads(PARAMS.get('test_chop'))
    for utter_dur_ in test_chop:
        res_fName = test_opDir_ + '/Result_' + str(utter_dur_) + 's.pkl'
        if not os.path.exists(res_fName):            
            '''
            Utterance-wise testing
            '''
            scores_ = XVec_.perform_testing(
                XVEC_enr_,
                XVEC_test_,
                classifier=gplda_classifier_,
                opDir=test_opDir_,
                feat_info=feat_info_[PARAMS['test_set']], 
                dim=int(PARAMS['num_dim']), 
                test_key=PARAMS['data_info_dir']+'/'+PARAMS['test_key'].split('/')[-1],
                duration=utter_dur_,
                )
            
            with open(res_fName, 'wb') as f_:
                pickle.dump({'scores':scores_}, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(res_fName, 'rb') as f_:
                scores_ = pickle.load(f_)['scores']

        ''' 
        Computing the performance metrics 
        '''
        metrics_ = XVec_.evaluate_performance(scores_, test_opDir_, utter_dur_)
        FPR_[utter_dur_] = metrics_['fpr']
        TPR_[utter_dur_] = metrics_['tpr']
            
    roc_opFile_ = test_opDir_ + '/ROC.png'
    PerformanceMetrics().plot_roc(FPR_, TPR_, roc_opFile_)




'''
I-Vector :: Speaker Verification System
'''
def ivector_sv(PARAMS, feat_info_):
    IVec_ = IVector(
        ubm_dir=PARAMS['ubm_dir'], 
        model_dir=PARAMS['model_dir'], 
        opDir=PARAMS['output_dir'],
        num_mixtures=int(PARAMS['ubm_ncomponents']),
        ivec_dim=int(PARAMS['rank_tv']),
        tv_iteration=int(PARAMS['tv_iteration']),
        feat_scaling=int(PARAMS['feature_scaling']),
        mem_limit=int(PARAMS['ubm_memory_limit']),
        )
    
    '''
    Training the GMM-UBM model
    '''
    print('Training the GMM-UBM model')
    ubm_fName = PARAMS['ubm_dir'] + '/ubm.pkl'
    if not os.path.exists(ubm_fName):
        FV_dev_, ram_mem_req_ = LoadFeatures(
            info=feat_info_[PARAMS['dev_set']], 
            feature_name=PARAMS['feature_name']
            ).load(dim=int(PARAMS['num_dim']))
        IVec_.train_ubm(
            FV_dev_,
            ram_mem_req_,
            cov_type=PARAMS['covariance_type'],
            ubm_dir=PARAMS['ubm_dir'],
            )
    else:
        print('The GMM-UBM is already available')
    print('\n\n')


    '''
    Estimating the Total Variability matrix
    '''
    tv_mat_fName_ = PARAMS['model_dir'] + '/tv_mat.pkl'
    if not os.path.exists(tv_mat_fName_):
        FV_dev_, ram_mem_req_ = LoadFeatures(
            info=feat_info_[PARAMS['dev_set']], 
            feature_name=PARAMS['feature_name']
            ).load(dim=int(PARAMS['num_dim']))
        IVec_.train_tv_matrix(FV_dev_, tv_mat_fName_)
    else:
        print('The Total Variability matrix is already trained')
    print('\n\n')
    
    
    '''
    Extracting Development I-Vectors
    '''
    FV_dev_, ram_mem_req_ = LoadFeatures(
        info=feat_info_[PARAMS['dev_set']], 
        feature_name=PARAMS['feature_name']
        ).load(dim=int(PARAMS['num_dim']))
    dev_ivec_dir_ = PARAMS['model_dir'] + '/' + PARAMS['dev_set'] + '_ivector/'
    IVec_.extract_ivector(FV_dev_, tv_mat_fName_, dev_ivec_dir_)


    '''
    G-PLDA Training
    '''
    gplda_fName_ = PARAMS['model_dir'] + '/G_PLDA.pkl'
    if not os.path.exists(gplda_fName_):
        gplda_model_, projection_matrix_ = IVec_.plda_training(FV_dev_, dev_ivec_dir_, lda_dim=20, LDA=True, WCCN=True)
        gplda_classifier_ = {'gplda_model':gplda_model_, 'projection_matrix': projection_matrix_}
        with open(gplda_fName_, 'wb') as fid_:
            pickle.dump(gplda_classifier_, fid_, pickle.HIGHEST_PROTOCOL)
    else:
        print('G-PLDA model already trained')
        with open(gplda_fName_, 'rb') as fid_:
            gplda_classifier_ = pickle.load(fid_)


    '''
    Extracting Enrollment I-Vectors
    '''
    FV_enr_, ram_mem_req_ = LoadFeatures(
        info=feat_info_[PARAMS['enr_set']], 
        feature_name=PARAMS['feature_name']
        ).load(dim=int(PARAMS['num_dim']))
    enr_ivec_dir_ = PARAMS['model_dir']+'/'+PARAMS['enr_set']+'_ivector/'
    IVec_.extract_ivector(FV_enr_, tv_mat_fName_, enr_ivec_dir_)


    ''' 
    Testing the trained models 
    '''
    print('Testing the trained models')
    test_opDir_ = PARAMS['output_dir'] + '/' + PARAMS['test_key'].split('/')[-1].split('.')[0] + '_' + PARAMS['model_type'] + '/'
    if not os.path.exists(test_opDir_):
        os.makedirs(test_opDir_)
    FPR_ = {}
    TPR_ = {}
    test_chop = json.loads(PARAMS.get('test_chop'))
    for utter_dur_ in test_chop:
        res_fName = test_opDir_ + '/Result_' + str(utter_dur_) + 's.pkl'
        if not os.path.exists(res_fName):            
            '''
            Utterance-wise testing
            '''
            scores_ = IVec_.perform_testing(
                enr_ivec_dir_, 
                tv_fName=tv_mat_fName_,
                classifier=gplda_classifier_,
                opDir=test_opDir_,
                feat_info=feat_info_[PARAMS['test_set']], 
                dim=int(PARAMS['num_dim']), 
                test_key=PARAMS['data_info_dir']+'/'+PARAMS['test_key'].split('/')[-1],
                duration=utter_dur_,
                )
            
            with open(res_fName, 'wb') as f_:
                pickle.dump({'scores':scores_}, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(res_fName, 'rb') as f_:
                scores_ = pickle.load(f_)['scores']

        ''' 
        Computing the performance metrics 
        '''
        metrics_ = IVec_.evaluate_performance(scores_, test_opDir_, utter_dur_)
        FPR_[utter_dur_] = metrics_['fpr']
        TPR_[utter_dur_] = metrics_['tpr']
            
    roc_opFile_ = test_opDir_ + '/ROC.png'
    PerformanceMetrics().plot_roc(FPR_, TPR_, roc_opFile_)







'''
GMM-UBM :: Speaker Verification System
'''
def gmm_ubm_sv(PARAMS, feat_info_):
    print('Creating a GaussianBackground model object')
    GB_ = GaussianBackground(
        model_dir=PARAMS['ubm_dir'], 
        opDir=PARAMS['output_dir'],
        num_mixtures=int(PARAMS['ubm_ncomponents']), 
        feat_scaling=int(PARAMS['feature_scaling']),
        mem_limit=int(PARAMS['ubm_memory_limit'])
        )
    
    '''
    Training the GMM-UBM model
    '''
    print('Training the GMM-UBM model')
    ubm_fName = PARAMS['ubm_dir'] + '/ubm.pkl'
    if not os.path.exists(ubm_fName):
        FV_dev_, ram_mem_req_ = LoadFeatures(
            info=feat_info_[PARAMS['dev_set']], 
            feature_name=PARAMS['feature_name']
            ).load(dim=int(PARAMS['num_dim']))
        GB_.train_ubm(
            FV_dev_,
            ram_mem_req_,
            cov_type=PARAMS['covariance_type']
            )
    else:
        print('The GMM-UBM is already available')
    print('\n\n')
    
    
    ''' 
    Speaker-wise adaptation 
    '''
    print('Speaker-wise adaptation')
    FV_enr_, ram_mem_req_ = LoadFeatures(
        info=feat_info_[PARAMS['enr_set']], 
        feature_name=PARAMS['feature_name']
        ).load(dim=int(PARAMS['num_dim']))
            
    GB_.speaker_adaptation(
        FV_enr_, 
        cov_type=PARAMS['covariance_type'], 
        use_adapt_w_cov=bool(PARAMS['adapt_weight_cov']),
        )
    print('\n\n')
            
        
    ''' 
    Testing the trained models 
    '''
    print('Testing the trained models')
    test_opDir_ = PARAMS['output_dir'] + '/' + PARAMS['test_key'].split('/')[-1].split('.')[0]+ '_' + PARAMS['model_type'] + '/'
    if not os.path.exists(test_opDir_):
        os.makedirs(test_opDir_)
    
    FPR_ = {}
    TPR_ = {}
    test_chop = json.loads(PARAMS.get('test_chop'))
    for utter_dur_ in test_chop:
        res_fName = test_opDir_ + '/Result_' + str(utter_dur_) + 's.pkl'
        if not os.path.exists(res_fName):
            '''
            All test-data loaded at-once
            '''
            '''
            FV_test_, ram_mem_req_ = LoadFeatures(
                info=feat_info_[PARAMS['test_set']], 
                feature_name=PARAMS['feature_name']
                ).load(dim=int(PARAMS['num_dim']))
            scores_ = GB_.perform_testing(
                opDir=PARAMS['output_dir'], 
                opFileName='Test_Scores', 
                X_TEST=FV_test_, 
                duration=int(utter_dur_)
                )
            '''
            
            '''
            Utterance-wise testing
            '''
            scores_ = GB_.perform_testing(
                opDir=test_opDir_, 
                feat_info=feat_info_[PARAMS['test_set']], 
                test_key=PARAMS['data_info_dir']+'/'+PARAMS['test_key'].split('/')[-1],
                dim=int(PARAMS['num_dim']), 
                duration=utter_dur_,
                )
            
            with open(res_fName, 'wb') as f_:
                pickle.dump({'scores':scores_}, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(res_fName, 'rb') as f_:
                scores_ = pickle.load(f_)['scores']

        ''' 
        Computing the performance metrics 
        '''
        metrics_ = GB_.evaluate_performance(scores_, test_opDir_, utter_dur_)
        FPR_[utter_dur_] = metrics_['fpr']
        TPR_[utter_dur_] = metrics_['tpr']
            
    roc_opFile_ = test_opDir_ + '/ROC.png'
    PerformanceMetrics().plot_roc(FPR_, TPR_, roc_opFile_)




def __init__():
    '''
    This function initiates the i-SpeakR system with the environment settings.

    Returns
    -------
    args : namespace
        This namespace contains the environment variables as attributes to 
        run the experiments.

    '''
    # Setting up the argparse object
    parser = argparse.ArgumentParser(
        prog='i-SpeakR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='SPEECH TECHNOLOGIES IN INDIAN LANGUAGES\n-----------------------------------------\nDeveloping Speaker Recognition systems for Indian scenarios',
        epilog='The above syntax needs to be strictly followed.',
        )
    # Adding the version information of the i-SpeakR toolkit
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1.1')

    # parser.add_argument(
    #     '--model_type',
    #     type=str,
    #     choices=['GMM-UBM', 'I-Vector'],
    #     default='GMM-UBM',
    #     help='Currently supports only GMM-UBM model',
    #     required=True,
    #     )

    args = parser.parse_args()
    
    return args



    
if __name__ == '__main__':
    args = __init__()
    
    base_config = configparser.ConfigParser()
    base_config.read('config.ini')
    opDir_ = base_config['MAIN']['output_path'] + '/i-SpeakR_output/' + base_config['MAIN']['dataset_name'] + '/'

    base_config = configparser.ConfigParser()
    base_config.read(opDir_+'/setup.ini')
    PARAMS = dict(base_config['EXECUTION_SETUP']).copy()
    
    print('Feature details..')
    if PARAMS['feature_name']=='MFCC':
        feature_info_ = MFCC(config=PARAMS).get_feature_details()
        print('Feature details obtained')
    
    if PARAMS['model_type']=='gmm_ubm':
        gmm_ubm_sv(PARAMS, feature_info_)
        
    if PARAMS['model_type']=='i_vector':
        ivector_sv(PARAMS, feature_info_)
        
    if PARAMS['model_type']=='x_vector':
        xvector_sv(PARAMS, feature_info_)
        