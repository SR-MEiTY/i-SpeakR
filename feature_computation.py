#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:45:19 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad

"""

from lib.feature_computation.compute_mfcc import MFCC
import configparser


if __name__ == '__main__':
    base_config = configparser.ConfigParser()
    base_config.read('config.ini')
    opDir_ = base_config['MAIN']['output_path'] + '/i-SpeakR_output/' + base_config['MAIN']['dataset_name'] + '/'

    PARAMS = configparser.ConfigParser()
    PARAMS.read(opDir_+'/setup.ini')
    PARAMS = PARAMS['EXECUTION_SETUP']
    
    print('Feature computation..')
    if PARAMS['feature_name']=='MFCC':
        feat_info_ = MFCC(config=PARAMS).compute()
        print('\n\n')
            
