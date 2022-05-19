#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:47:48 2022

@author: 
    Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
@collaborators: 
    Jagabandhu Mishra, Ph.D. Scholar, Dept. of EE, IIT Dharwad

Naming conventions followed:
    Local variable:   Should have all lower-case alphabets [a-z]
                      Should start with an alphabet
                      Can include numerals [0-9]
                      Can include special characters [@, _, $]
                      Should end with an underscore "_"
    
    Global variables: Should have all upper-case alphabets [A-Z]
                      Should start with an alphabet
                      Can include numerals [0-9]
                      Can include special characters [@, _, $]
                      Should NOT end with an underscore "_"
    
    Function names:   Should have all lower-case alphabets [a-z]
                      Should follow snake case. <this_is_a_snake_case>
                      Should not include numerals [0-9]
                      Can include underscore "_"
    
    Class names:      Can have a combination of lower-case [a-z] and upper-case 
                      alphabets [A-Z]
                      Should follow Pascal case - ThisIsPascalCase
                      Words must start with an upper-case alphabet
                      Consecutive words must be concatenated
                      Should not include numerals [0-9] or special characters
                      
    Commenting:       All functions must include Doc Strings
                      All significant line of code must have an inline comment
                      All variable declarations must be commented
                    
                    
"""

import argparse
from lib.data_io.metafile_reader import GetMetaInfo
from lib.data_io.chop_utterances import ChopUtterances
from lib.feature_computation.compute_mfcc import MFCC
import configparser
import json
import sys


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
                    del metaobj.info[dev_sets[key]]
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
    test_sets = [set_name if set_name.startswith('TEST') else '' for set_name in metaobj.info.keys()]
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
        'SAMPLING_RATE': int(section['sampling_rate']),             # Sampling rate to be used for the audio files
        'FRAME_SIZE': int(section['frame_size']),                   # Short-term frame size in miliseconds
        'FRAME_SHIFT': int(section['frame_shift']),                 # Short-term frame shift in miliseconds 
        'DEV_CHOP': [json.loads(config.get('MAIN', 'DEV_chop'))],   # Development sub-utterance sizes
        'EN_CHOP': [json.loads(config.get('MAIN', 'EN_chop'))],     # Enrollment sub-utterance sizes
        'TEST_CHOP': json.loads(config.get('MAIN', 'TEST_chop')),   # Test sub-utterance sizes
        'PREEMPHASIS': section.getboolean('preemphasis'),           # Boolean flag indicating pre-emphasis required or not. True indicates pre-emphasis is required, False indicates pre-emphasis not required.
        'N_MELS': int(section['n_mels']),                           # Number of Mel filters to be used
        'N_MFCC': int(section['n_mfcc']),                           # Number of MFCC coefficients to be computed. If EXCL_C0=True, N_MFCC+1 coefficients are computed and c0 is ignored
        'DELTA_WIN': int(section['delta_win']),                     # Context window to be used for computing Delta features
        'EXCL_C0': section.getboolean('excl_c0')                    # Boolean flag indicating whether MFCC c0 to be used or not. True indicates c0 is included. False indicates c0 is to be ignored and N_MFCC+1 coefficients to be computed
        }
    
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
        epilog='The above systax needs to be strictly followed.',
        )
    # Adding the version information of the i-SpeakR toolkit
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 0.1.1')
    # Adding a switch for the path to a csv file containing the meta information of the dataset
    # Adding switch to indicate the type of data
    parser.add_argument(
        '--data_info',
        type=str,
        choices=['infer', 'specify'],
        default='infer',
        help='Switch to select how to obtain the dataset details'
        )
    parser.add_argument(
        '--meta_path',
        type=str,
        help='Path to the meta file. Ignored if data_info="infer". Searches for DEV*.csv, ENR*.csv and TEST*.csv in <meta_path> when <data_info="specify">'
        )
    # Adding a switch to the base path of the dataset
    parser.add_argument(
        '--base_path',
        type=str,
        help='Base path to the data. If <data_info="infer">, expects three folders in <base_path>, viz. DEV, ENR, TEST. All wav files for DEV, ENR or TEST sets needs to be kept within each respective directories without any sub-directories.'
        )
    # Adding a switch for the output path to be used by the system
    parser.add_argument(
        '--output_path',
        type=str,
        help='Path to store the ouputs'
        )
    args = parser.parse_args()

    print(f'meta_path: {args.meta_path}')
    print(f'base_path: {args.base_path}')
    print(f'output_path: {args.output_path}')
    
    return args


if __name__ == '__main__':
    args = __init__()
    CFG = get_configurations()
    print('Global variables:')
    for key in CFG:
        print(f'\t{key}={CFG[key]}')
    print('\n')
    
    metaobj = GetMetaInfo(data_info=args.data_info, path=args.meta_path, duration=True, gender=True)
    print('\nData sets:')
    for f in metaobj.info.keys():
        print(f"\t{f}", end=' ', flush=True)
        print(f"Duration={metaobj.total_duration[f]}", end=' ', flush=True)
        try:
            print(f"{metaobj.gender_distribution[f]}")
        except:
            print('')
    print('\n')
        
    metaobj.info = select_data_sets(metaobj.info)
    print(f'\nSelected sets: {metaobj.info.keys()}\n')

    sys.exit(0)
    
    if args.data_type=='DEV':
        segment_duration = CFG['DEV_chop'] # Duration of the segments extracted from utterances
    elif args.data_type=='EN':
        segment_duration = CFG['EN_chop'] # Duration of the segments extracted from utterances
    else:
        segment_duration = CFG['TEST_chop'] # Duration of the segments extracted from utterances
    splitsDir = args.output_path + '/segment_info/' + args.data_type + '/'
        
    ChopUtterances(chop_size=segment_duration, config=CFG).create_splits(metaobj.info, args.base_path, splitsDir)

    feat_dir = args.output_path + '/features/MFCC/' + args.data_type + '/'
    MFCC(config=CFG).compute(args.base_path, metaobj.info, splitsDir, feat_dir, delta=True)