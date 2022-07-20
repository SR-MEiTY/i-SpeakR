#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:39:42 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""


import argparse
import configparser
import json
import sys
import datetime
import os




def update_execution_config(CFG, PARAMS):
    opFile_ = CFG['OUTPUT_DIR']+'/setup.ini'
    config = configparser.ConfigParser()
    config.read(opFile_)
    section_name_ = 'EXECUTION_SETUP'
    if section_name_ not in config.sections():
        config.add_section(section_name_)
    
    for name_ in PARAMS.keys():
        config.set(section_name_, name_, str(PARAMS[name_]))

    # Writing our configuration file to 'example.ini'
    with open(opFile_, 'w') as f_:
        config.write(f_)
    



def select_data_sets(info, CFG):
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


def get_configurations():
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
        'TODAY': datetime.datetime.now().strftime("%Y-%m-%d"),
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
        'COMPUTE_DELTA_FEAT': section.getboolean('compute_delta_feat'), # Boolean flag indicating whether delta features are computed
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
        'data_info': section['data_info'],                          # Data type is either "infer" or "specify"
        'data_path': section['data_path'],                          # Path to data set
        'output_path': section['output_path'],                      # Path where program output is to be stored
        'dataset_name': section['dataset_name'],                    # Name of the dataset
        }
    
    if CFG['FEATURE_NAME']=='MFCC':
        if CFG['COMPUTE_DELTA_FEAT']:
            CFG['NUM_DIM'] = 3*CFG['N_MFCC']
        else:
            CFG['NUM_DIM'] = CFG['N_MFCC']

    CFG['OUTPUT_DIR'] = CFG['output_path'] + '/i-SpeakR_output/' + CFG['dataset_name'] + '/'
    if not os.path.exists(CFG['OUTPUT_DIR']):
        os.makedirs(CFG['OUTPUT_DIR'])
        
    CFG['DATA_INFO_DIR'] = CFG['OUTPUT_DIR'] + '/data_info/'
    if not os.path.exists(CFG['DATA_INFO_DIR']):
        os.makedirs(CFG['DATA_INFO_DIR'])

    CFG['FEAT_DIR'] = CFG['OUTPUT_DIR'] + '/features/' + CFG['FEATURE_NAME'] + '/'
    if not os.path.exists(CFG['FEAT_DIR']):
        os.makedirs(CFG['FEAT_DIR'])

    CFG['MODEL_DIR'] = CFG['OUTPUT_DIR'] + '/models/' + CFG['FEATURE_NAME'] + '_' + CFG['MODEL_TYPE'] + '/'
    if not os.path.exists(CFG['MODEL_DIR']):
        os.makedirs(CFG['MODEL_DIR'])
    
    return CFG
    

def parse_commandline_arguments():
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
