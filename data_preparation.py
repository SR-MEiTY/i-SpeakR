#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:38:16 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad

"""

from lib.data_io.metafile_reader import GetMetaInfo
from lib.data_io.chop_utterances import ChopUtterances
from lib import init_funcs as InFunc
import argparse


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
        description='Data preparation',
        epilog='',
        )
    # Adding the version information of the i-SpeakR toolkit
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1.1')
    # Adding a switch for the path to a csv file containing the meta information of the dataset
    # Adding switch to indicate the type of data

    parser.add_argument(
        '--dev_path',
        type=str,
        help='Path to development-set data',
        required=True,
        )

    parser.add_argument(
        '--dev_key',
        type=str,
        help='Path to development-set key file (*.csv)',
        required=True,
        )

    parser.add_argument(
        '--enr_path',
        type=str,
        help='Path to enrollment-set data',
        required=True,
        )

    parser.add_argument(
        '--enr_key',
        type=str,
        help='Path to enrollment-set key file (*.csv)',
        required=True,
        )

    parser.add_argument(
        '--test_path',
        type=str,
        help='Path to test-set data',
        required=True,
        )

    parser.add_argument(
        '--test_key',
        type=str,
        help='Path to test-set key file (*.csv)',
        required=True,
        )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    CFG = InFunc.get_configurations()
    print('Global variables:')
    for key in CFG:
        print(f'\t{key}={CFG[key]}')
    print('\n\n')
    
    # 17 Sep 22
    args = __init__()
    CFG['dev_path'] = args.dev_path
    CFG['dev_key'] = args.dev_key
    CFG['enr_path'] = args.enr_path
    CFG['enr_key'] = args.enr_key
    CFG['test_path'] = args.test_path
    CFG['test_key'] = args.test_key

    InFunc.update_execution_config(CFG, CFG)
    InFunc.update_execution_config(CFG, {
        'output_path': CFG['output_path'], 
        # 'data_path': CFG['data_path'], 
        'data_info': CFG['data_info'],
        # 17 Sep 22
        'dev_path': CFG['dev_path'],
        'dev_key': CFG['dev_key'],
        'enr_path': CFG['enr_path'],
        'enr_key': CFG['enr_key'],
        'test_path': CFG['test_path'],
        'test_key': CFG['test_key'],
        })

    metaobj = GetMetaInfo(
        data_info=CFG['data_info'], 
        path={'DEV': CFG['dev_path'], 'ENR': CFG['enr_path'], 'TEST': CFG['test_path']}, 
        key={'DEV': CFG['dev_key'], 'ENR': CFG['enr_key'], 'TEST': CFG['test_key']}, 
        # duration=True, 
        # gender=True
        )    
    print('\nData sets:')
    # for f in metaobj.INFO.keys():
    #     print(f"\t{f}", end=' ', flush=True)
    #     print(f"Duration={metaobj.TOTAL_DURATION[f]}", end=' ', flush=True)
    #     try:
    #         print(f"{metaobj.GENDER_DISTRIBUTION[f]}")
    #     except:
    #         print('')
    print('\n\n')

    
    metaobj.INFO = InFunc.select_data_sets(metaobj.INFO, CFG)
    print(f'\nSelected sets: {metaobj.INFO.keys()}\n')
    for key in metaobj.INFO.keys():
        if key.startswith('DEV'):
            InFunc.update_execution_config(CFG, {'dev_set':key})
        if key.startswith('ENR'):
            InFunc.update_execution_config(CFG, {'enr_set':key})
        if key.startswith('TEST'):
            InFunc.update_execution_config(CFG, {'test_set':key})
    
    print('Utterance chopping..')
    ChopUtterances(config=CFG).create_splits(
        metaobj.INFO, 
        key={'DEV': CFG['dev_key'], 'ENR': CFG['enr_key'], 'TEST': CFG['test_key']}, 
        )
    print('\n\n')
