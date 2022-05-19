#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:47:48 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""

import argparse
from lib.data_io.metafile_reader import ReadMetaFile
from lib.data_io.chop_utterances import ChopUtterances
from lib.feature_computation.compute_mfcc import MFCC
import configparser
import json


def get_configurations():
    config = configparser.ConfigParser()
    config.read('config.ini')
    section = config['MAIN']
    CFG = {
        'sampling_rate': int(section['sampling_rate']),
        'frame_size': int(section['frame_size']),
        'frame_shift': int(section['frame_shift']),
        'DEV_chop': [json.loads(config.get('MAIN', 'DEV_chop'))],
        'EN_chop': [json.loads(config.get('MAIN', 'EN_chop'))],
        'TEST_chop': json.loads(config.get('MAIN', 'TEST_chop')),
        'preemphasis': section.getboolean('preemphasis'),
        'n_mels': int(section['n_mels']),
        'n_mfcc': int(section['n_mfcc']),
        'delta_win': int(section['delta_win']),
        'excl_c0': section.getboolean('excl_c0')
        }
    
    return CFG
    

def __init__():
    parser = argparse.ArgumentParser(
        prog='InSpeakR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='SPEECH TECHNOLOGIES IN INDIAN LANGUAGES\n-----------------------------------------\nDeveloping Speaker Recognition systems for Indian scenarios',
        epilog='The above systax needs to be strictly followed.',
        )
    parser.add_argument('--version', '-v', action='version', version='%(prog)s 0.1')
    parser.add_argument('--meta_path', type=str, help='Path to the meta file')
    parser.add_argument('--base_path', type=str, help='Base path to the data')
    parser.add_argument('--output_path', type=str, help='Path to store the ouputs')
    parser.add_argument('--os_type', choices=['Windows', 'Linux'], help='Type of operating system', default='Linux')
    parser.add_argument('--data_type', type=str, choices=['DEV', 'EN', 'TEST'], default='dev', help='Type of data being processed')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = __init__()
    CFG = get_configurations()
    print(CFG)
    print(f'meta_path: {args.meta_path}')
    print(f'base_path: {args.base_path}')
    print(f'output_path: {args.output_path}')
    
    metaobj = ReadMetaFile(path=args.meta_path, os=args.os_type, duration=True, gender=True)
    print(f"\nMetafile read ({args.meta_path.split('/')[-1]})) total_duration={metaobj.total_duration} gender_distribution={metaobj.gender_distribution}")
    
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