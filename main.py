#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:47:48 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""

import argparse
from lib.data_io.metafile_reader import ReadMetaFile
from lib.data_io.utterance_splitter import SplitUtterances

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
    parser.add_argument('--preemphasis', action='store_true', help='Pre-emphasis flag')
    parser.add_argument('--data_type', type=str, choices=['DEV', 'EN', 'TEST'], default='dev', help='Type of data being processed')
    parser.add_argument('--sr', type=str, choices=['up', 'down', 'same'], default='same', help='Changing audio sampling rate')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = __init__()
    metaobj = ReadMetaFile(path=args.meta_path, os=args.os_type, duration=True, gender=True)
    print(f"\nMetafile read ({args.meta_path.split('/')[-1]})) total_duration={metaobj.total_duration} gender_distribution={metaobj.gender_distribution}")
    
    if (args.data_type=='DEV') or (args.data_type=='EN'):
        segment_duration = [50] # Duration of the segments extracted from utterances
    else:
        segment_duration = list(range(10,61,10)) # Duration of the segments extracted from utterances
    opDir = args.output_path + '/' + args.data_type + '/'
        
    SplitUtterances(segment_duration).create_splits(metaobj.info, args.base_path, opDir)
    

