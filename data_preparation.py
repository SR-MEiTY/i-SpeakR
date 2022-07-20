#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:38:16 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad

"""

from lib.data_io.metafile_reader import GetMetaInfo
from lib.data_io.chop_utterances import ChopUtterances
from lib import init_funcs as InFunc


if __name__ == '__main__':
    CFG = InFunc.get_configurations()
    print('Global variables:')
    for key in CFG:
        print(f'\t{key}={CFG[key]}')
    print('\n\n')

    InFunc.update_execution_config(CFG, CFG)
    InFunc.update_execution_config(CFG, {
        'output_path': CFG['output_path'], 
        'data_path': CFG['data_path'], 
        'data_info': CFG['data_info'],
        })

    metaobj = GetMetaInfo(data_info=CFG['data_info'], path=CFG['data_path'], duration=True, gender=True)    
    print('\nData sets:')
    for f in metaobj.INFO.keys():
        print(f"\t{f}", end=' ', flush=True)
        print(f"Duration={metaobj.TOTAL_DURATION[f]}", end=' ', flush=True)
        try:
            print(f"{metaobj.GENDER_DISTRIBUTION[f]}")
        except:
            print('')
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
    ChopUtterances(config=CFG).create_splits(metaobj.INFO, CFG['data_path'])
    print('\n\n')
