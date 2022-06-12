#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:47:48 2022

@author: 
    Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
@collaborators: 
    Jagabandhu Mishra, Ph.D. Scholar, Dept. of EE, IIT Dharwad

"""

import argparse
from lib.data_io.metafile_reader import GetMetaInfo
from lib.data_io.chop_utterances import ChopUtterances
from lib.feature_computation.compute_mfcc import MFCC
from lib.feature_computation.load_features import LoadFeatures
from lib.models.GMM_UBM.gaussian_background import GaussianBackground
from lib.metrics.performance_metrics import PerformanceMetrics
import configparser
import json
import sys
import datetime
import os
import numpy as np
import pickle


def select_data_sets(info):
    '''
    Function to select development, enrollment and test sets when multiple 
    options are available.

    Parameters
    ----------
    info : dict
        Dict contains the information about various sets in the dataset.

    Returns
    -------
    info : dict
        A pruned dict that contains only those sets that are selected by the
        user.

    '''

    # Select set names starting with "DEV"
    dev_sets = [set_name if set_name.startswith('DEV') else '' for set_name in info.keys()] 
    # Remove empty set names
    dev_sets = list(filter(None, dev_sets))                                                 
    # Convert the list of names to a dict 
    dev_sets = {i:dev_sets[i] for i in range(len(dev_sets))}                                
    # If number of DEV sets greater than 1, ask user to selct one
    if len(dev_sets)>1:
        print('DEV sets found: ', end=' ', flush=True)
        print(dev_sets, end=' ', flush=True)
        dev_choice = input('Select one to be used: ')
        try:
            dev_choice = int(dev_choice)
            for key in dev_sets.keys():
                if not key==dev_choice:
                    del info[dev_sets[key]]
        except:
            print('Wrong choice')
            sys.exit(0)
    
    # Select set names starting with "ENR"
    enr_sets = [set_name if set_name.startswith('ENR') else '' for set_name in info.keys()]
    # Remove empty set names
    enr_sets = list(filter(None, enr_sets))
    # Convert the list of names to a dict 
    enr_sets = {i:enr_sets[i] for i in range(len(enr_sets))}
    # If number of ENR sets greater than 1, ask user to selct one
    if len(enr_sets)>1:
        print('ENR sets found: ', end=' ', flush=True)
        print(enr_sets, end=' ', flush=True)
        enr_choice = input('Select one to be used: ')
        try:
            enr_choice = int(enr_choice)
            for key in enr_sets.keys():
                if not key==enr_choice:
                    del info[enr_sets[key]]
        except:
            print('Wrong choice')
            sys.exit(0)
    
    # Select set names starting with "TEST"
    test_sets = [set_name if set_name.startswith('TEST') else '' for set_name in info.keys()]
    # Remove empty set names
    test_sets = list(filter(None, test_sets))
    # Convert the list of names to a dict 
    test_sets = {i:test_sets[i] for i in range(len(test_sets))}
    # If number of TEST sets greater than 1, ask user to selct one
    if len(test_sets)>1:
        print('TEST sets found: ', end=' ', flush=True)
        print(test_sets, end=' ', flush=True)
        test_choice = input('Select one to be used: ')
        try:
            test_choice = int(test_choice)
            for key in test_sets.keys():
                if not key==test_choice:
                    del info[test_sets[key]]
        except:
            print('Wrong choice')
            sys.exit(0)
    
    return info


def get_configurations(args):
    '''
    This function reads the config.ini file that lists the fixed variables to
    be used.

    Returns
    -------
    CFG : dict
        This dict file contains the global variables set for running the 
        experiments.

    '''
    config = configparser.ConfigParser()
    config.read('config.ini')
    section = config['MAIN']
    CFG = {
        'today': datetime.datetime.now().strftime("%Y-%m-%d"),
        'SAMPLING_RATE': int(section['sampling_rate']),             # Sampling rate to be used for the audio files
        'NFFT': int(section['n_fft']),                              # Number of DFT points to be used
        'FRAME_SIZE': int(section['frame_size']),                   # Short-term frame size in miliseconds
        'FRAME_SHIFT': int(section['frame_shift']),                 # Short-term frame shift in miliseconds 
        'DEV_CHOP': [json.loads(config.get('MAIN', 'DEV_chop'))],   # Development sub-utterance sizes
        'ENR_CHOP': [json.loads(config.get('MAIN', 'ENR_chop'))],     # Enrollment sub-utterance sizes
        'TEST_CHOP': json.loads(config.get('MAIN', 'TEST_chop')),   # Test sub-utterance sizes
        'PREEMPHASIS': section.getboolean('preemphasis'),           # Boolean flag indicating pre-emphasis required or not. True indicates pre-emphasis is required, False indicates pre-emphasis not required.
        'N_MELS': int(section['n_mels']),                           # Number of Mel filters to be used
        'N_MFCC': int(section['n_mfcc']),                           # Number of MFCC coefficients to be computed. If EXCL_C0=True, N_MFCC+1 coefficients are computed and c0 is ignored
        'DELTA_WIN': int(section['delta_win']),                     # Context window to be used for computing Delta features
        'EXCL_C0': section.getboolean('excl_c0'),                   # Boolean flag indicating whether MFCC c0 to be used or not. True indicates c0 is included. False indicates c0 is to be ignored and N_MFCC+1 coefficients to be computed
        'FEATURE_NAME': section['feature_name'],                    # Parameter to indicate which feature to compute
        'FEATURE_SCALING': int(section['feature_scaling']),         # Type of feature scaling to be used.
                                                                    # 0: no scaling, 
                                                                    # 1: only mean subtraction,      
                                                                    # 2: mean and variance scaling
        'MODEL_TYPE': section['model'],                             # Parameter indicating which model to use
        'UBM_NCOMPONENTS': int(section['UBM_ncomp']),               # Number of Gaussian components for the UBM model
        'COVARIANCE_TYPE': section['covariance_type'],              # Type of covariance: 'full', 'diag', 'tied'
        'ADAPT_WEIGHT_COV': section.getboolean('adapt_weight_cov'), # Flag to indicate whether to adapt the weights and covariances of the speaker models
        }

    CFG['OUTPUT_DIR'] = args.output_path + '/i-SpeakR_output/' + args.data_path.split('/')[-2] + '_' + CFG['today'] + '/'
    if not os.path.exists(CFG['OUTPUT_DIR']):
        os.makedirs(CFG['OUTPUT_DIR'])
        
    CFG['SPLITS_DIR'] = CFG['OUTPUT_DIR'] + '/sub_utterance_info/'
    if not os.path.exists(CFG['SPLITS_DIR']):
        os.makedirs(CFG['SPLITS_DIR'])

    CFG['FEAT_DIR'] = CFG['OUTPUT_DIR'] + '/features/' + CFG['FEATURE_NAME'] + '/'
    if not os.path.exists(CFG['FEAT_DIR']):
        os.makedirs(CFG['FEAT_DIR'])

    CFG['MODEL_DIR'] = CFG['OUTPUT_DIR'] + '/models/' + CFG['FEATURE_NAME'] + '_' + CFG['MODEL_TYPE'] + '/'
    if not os.path.exists(CFG['MODEL_DIR']):
        os.makedirs(CFG['MODEL_DIR'])

    CFG['FIG_DIR'] = CFG['OUTPUT_DIR'] + '/figures/'
    if not os.path.exists(CFG['FIG_DIR']):
        os.makedirs(CFG['FIG_DIR'])
    
    return CFG
    

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
        '--data_info',
        type=str,
        choices=['infer', 'specify'],
        default='infer',
        help='Switch to select how to obtain the dataset details',
        required=True,
        )
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to the meta file. If <data_info="infer">, expects three folders in <data_path>, viz. DEV, ENR, TEST. All wav files for DEV, ENR or TEST sets needs to be kept within each respective directories without any sub-directories. If <data_info="specify">, searches for DEV*.csv, ENR*.csv and TEST*.csv in <meta_path>',
        required=True,
        )
    # Adding a switch for the output path to be used by the system
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path to store the ouputs',
        default='../',
        )
    args = parser.parse_args()

    print(f'meta_path: {args.data_path}')
    print(f'output_path: {args.output_path}')
    
    return args


if __name__ == '__main__':
    args = __init__()
    
    metaobj = GetMetaInfo(data_info=args.data_info, path=args.data_path, duration=True, gender=True)
    print('\nData sets:')
    for f in metaobj.INFO.keys():
        print(f"\t{f}", end=' ', flush=True)
        print(f"Duration={metaobj.TOTAL_DURATION[f]}", end=' ', flush=True)
        try:
            print(f"{metaobj.GENDER_DISTRIBUTION[f]}")
        except:
            print('')
    print('\n\n')
        
    metaobj.INFO = select_data_sets(metaobj.INFO)
    print(f'\nSelected sets: {metaobj.INFO.keys()}\n')
    # print('DEV utterance ID: ', metaobj.INFO['DEV'].keys())
    # print('ENR utterance ID: ', metaobj.INFO['ENR'].keys())
    # print('TEST utterance ID: ', metaobj.INFO['TEST'].keys())

    CFG = get_configurations(args)
    print('Global variables:')
    for key in CFG:
        print(f'\t{key}={CFG[key]}')
    print('\n\n')
    
    print('Utterance chopping..')
    ChopUtterances(config=CFG).create_splits(metaobj.INFO, args.data_path)
    print('\n\n')

    print('Feature computation..')
    if CFG['FEATURE_NAME']=='MFCC':
        feat_info_ = MFCC(config=CFG).compute(args.data_path, metaobj.INFO, CFG['SPLITS_DIR'], CFG['FEAT_DIR'], delta=True)
        print('\n\n')
            
    if CFG['MODEL_TYPE']=='GMM_UBM':
        GB_ = GaussianBackground(
            model_dir=CFG['MODEL_DIR'], 
            opDir=CFG['OUTPUT_DIR'],
            num_mixtures=CFG['UBM_NCOMPONENTS'], 
            feat_scaling=CFG['FEATURE_SCALING']
            )
        
        '''
        Training the GMM-UBM model
        '''
        ubm_fName = CFG['MODEL_DIR'] + '/ubm.pkl'
        if not os.path.exists(ubm_fName):
            dev_key_ = list(filter(None, [key if key.startswith('DEV') else '' for key in feat_info_.keys()]))
            if not os.path.exists(CFG['OUTPUT_DIR']+'/DEV_Data.pkl'):
                FV_dev_ = LoadFeatures(info=feat_info_[dev_key_[0]], feature_name=CFG['FEATURE_NAME']).load(dim=3*CFG['N_MFCC'])
                # with open(CFG['OUTPUT_DIR']+'/DEV_Data.pkl', 'wb') as f_:
                #     pickle.dump(FV_dev_, f_, pickle.HIGHEST_PROTOCOL)
            else:
                with open(CFG['OUTPUT_DIR']+'/DEV_Data.pkl', 'rb') as f_:
                    FV_dev_ = pickle.load(f_)
            GB_.train_ubm(FV_dev_, cov_type=CFG['COVARIANCE_TYPE'])
        else:
            print('The GMM-UBM is already available')
        print('\n\n')
        
        
        ''' 
        Speaker-wise adaptation 
        '''
        enr_key_ = list(filter(None, [key if key.startswith('ENR') else '' for key in feat_info_.keys()]))
        if not os.path.exists(CFG['OUTPUT_DIR']+'/ENR_Data.pkl'):
            FV_enr_ = LoadFeatures(info=feat_info_[enr_key_[0]], feature_name=CFG['FEATURE_NAME']).load(dim=3*CFG['N_MFCC'])
            # with open(CFG['OUTPUT_DIR']+'/ENR_Data.pkl', 'wb') as f_:
            #     pickle.dump(FV_enr_, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(CFG['OUTPUT_DIR']+'/ENR_Data.pkl', 'rb') as f_:
                FV_enr_ = pickle.load(f_)
                
        GB_.speaker_adaptation(
            FV_enr_, 
            cov_type=CFG['COVARIANCE_TYPE'], 
            use_adapt_w_cov=CFG['ADAPT_WEIGHT_COV']
            )
        print('\n\n')
                
            
        ''' 
        Computing the performance metrics 
        '''
        for utter_dur_ in [30]: # CFG['TEST_CHOP']:
            res_fName = CFG['OUTPUT_DIR']+'/Result_'+str(utter_dur_)+'s.pkl'
            if not os.path.exists(res_fName):
                ''' 
                Testing the trained models 
                '''
                test_key_ = list(filter(None, [key if key.startswith('TEST') else '' for key in feat_info_.keys()]))
                
                '''
                All test-data loaded at-once
                '''
                '''
                FV_test_ = LoadFeatures(info=feat_info_[test_key_[0]], feature_name=CFG['FEATURE_NAME']).load(dim=3*CFG['N_MFCC'])
                scores_ = GB_.perform_testing(opDir=CFG['OUTPUT_DIR'], opFileName='Test_Scores', X_TEST=FV_test_, duration=utter_dur_)
                '''
                
                '''
                Utterance-wise testing
                '''
                scores_ = GB_.perform_testing(opDir=CFG['OUTPUT_DIR'], opFileName='Test_Scores', feat_info=feat_info_[test_key_[0]], duration=utter_dur_)
                
                with open(res_fName, 'wb') as f_:
                    pickle.dump({'scores':scores_}, f_, pickle.HIGHEST_PROTOCOL)
            else:
                with open(res_fName, 'rb') as f_:
                    scores_ = pickle.load(f_)['scores']

            metrics_ = GB_.evaluate_performance(scores_)
            roc_opFile = CFG['FIG_DIR'] + '/ROC_' + str(utter_dur_) + 's.png'
            PerformanceMetrics().plot_roc(metrics_['fpr'], metrics_['tpr'], roc_opFile)
                
            print(f'\n\nUtterance duration: {utter_dur_}s:\n__________________________________________')
            print(f"\tAccuracy: {np.round(metrics_['accuracy']*100,2)}")
            print(f"\tMacro Average Precision: {np.round(metrics_['precision']*100,2)}")
            print(f"\tMacro Average Recall: {np.round(metrics_['recall']*100,2)}")
            print(f"\tMacro Average F1-score: {np.round(metrics_['f1-score']*100,2)}")
            print(f"\tEER: {np.round(np.mean(metrics_['eer'])*100,2)}")
            with open(CFG['OUTPUT_DIR']+'/Performance.txt', 'a+') as f_:
                f_.write(f'Utterance duration: {utter_dur_}s:\n__________________________________________\n')
                f_.write(f"\tAccuracy: {np.round(metrics_['accuracy']*100,2)}\n")
                f_.write(f"\tMacro Average Precision: {np.round(metrics_['precision']*100,2)}\n")
                f_.write(f"\tMacro Average Recall: {np.round(metrics_['recall']*100,2)}\n")
                f_.write(f"\tMacro Average F1-score: {np.round(metrics_['f1-score']*100,2)}\n")
                f_.write(f"\tEER: {np.round(np.mean(metrics_['eer'])*100,2)}\n\n")
                
                