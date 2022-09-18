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
from lib.metrics.performance_metrics import PerformanceMetrics
import os
import pickle
import configparser
import argparse
import json




'''
I-Vector :: Speaker Verification System
'''
def ivector_sv(PARAMS):
        IVec_ = IVector(
            model_dir=PARAMS['model_dir'], 
            opDir=PARAMS['output_dir'],
            num_mixtures=int(PARAMS['ubm_ncomponents']), 
            feat_scaling=int(PARAMS['feature_scaling'])
            )
        
        '''
        Training the GMM-UBM model
        '''
        ubm_fName = PARAMS['model_dir'] + '/ubm.pkl'
        if not os.path.exists(ubm_fName):
            FV_dev_ = LoadFeatures(
                info=feat_info_[PARAMS['dev_set']], 
                feature_name=PARAMS['feature_name']
                ).load(dim=int(PARAMS['num_dim']))
            IVec_.train_ubm(
                FV_dev_, 
                cov_type=PARAMS['covariance_type']
                )
        else:
            print('The GMM-UBM is already available')
        print('\n\n')






'''
GMM-UBM :: Speaker Verification System
'''
def gmm_ubm_sv(PARAMS):
    print('Creating a GaussianBackground model object')
    GB_ = GaussianBackground(
        model_dir=PARAMS['model_dir'], 
        opDir=PARAMS['output_dir'],
        num_mixtures=int(PARAMS['ubm_ncomponents']), 
        feat_scaling=int(PARAMS['feature_scaling'])
        )
    
    '''
    Training the GMM-UBM model
    '''
    print('Training the GMM-UBM model')
    ubm_fName = PARAMS['model_dir'] + '/ubm.pkl'
    if not os.path.exists(ubm_fName):
        FV_dev_ = LoadFeatures(
            info=feat_info_[PARAMS['dev_set']], 
            feature_name=PARAMS['feature_name']
            ).load(dim=int(PARAMS['num_dim']))
        GB_.train_ubm(
            FV_dev_, 
            cov_type=PARAMS['covariance_type']
            )
    else:
        print('The GMM-UBM is already available')
    print('\n\n')
    
    
    ''' 
    Speaker-wise adaptation 
    '''
    print('Speaker-wise adaptation')
    FV_enr_ = LoadFeatures(
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
    test_opDir_ = PARAMS['output_dir'] + '/' + PARAMS['test_set'].split('.')[0] + '/'
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
            FV_test_ = LoadFeatures(
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
    # Adding a switch for the path to a csv file containing the meta information of the dataset
    # Adding switch to indicate the type of data
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['GMM-UBM'],
        default='GMM-UBM',
        help='Currently supports only GMM-UBM model',
        required=True,
        )
    args = parser.parse_args()
    
    return args



    
if __name__ == '__main__':
    args = __init__()
    
    base_config = configparser.ConfigParser()
    base_config.read('config.ini')
    opDir_ = base_config['MAIN']['output_path'] + '/i-SpeakR_output/' + base_config['MAIN']['dataset_name'] + '/'

    PARAMS = configparser.ConfigParser()
    PARAMS.read(opDir_+'/setup.ini')
    PARAMS = PARAMS['EXECUTION_SETUP']    
    
    print('Feature details..')
    if PARAMS['feature_name']=='MFCC':
        feat_info_ = MFCC(config=PARAMS).get_feature_details()
        print('Feature details obtained')
    
    
    if args.model_type=='GMM-UBM':    
        gmm_ubm_sv(PARAMS)
        
    if args.model_type=='I-Vector':    
        ivector_sv(PARAMS)        