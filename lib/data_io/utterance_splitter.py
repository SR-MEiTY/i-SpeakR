#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:14:22 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""

import os
from datetime import datetime
import csv
import numpy as np

class SplitUtterances:
    split_durations = []
    
    def __init__(self, dur):
        self.split_durations = dur
        
    def create_splits(self, meta_info, base_path, opDir):
        for spldur in self.split_durations:
            for sl_no in meta_info.keys():
                fName_path = base_path + '/' + meta_info[sl_no]['Local_Path']
                fName = fName_path.split('/')[-1] # meta_info[sl_no]['File Name']
    
                opDir_path = opDir + '/' + '/'.join(meta_info[sl_no]['Local_Path'].split('/')[:-1])
                if not os.path.exists(opDir_path):
                    os.makedirs(opDir_path)
                opFile = opDir_path + '/' + fName.split('.')[0] + '.csv'
                
                duration = datetime.strptime(meta_info[sl_no]['Duration'], '%H:%M:%S').time()
                NSec = duration.hour*3600 + duration.minute*60 + duration.second
                
                line_count = 0
                if os.path.exists(opFile):
                    with open(opFile, 'r+', encoding='utf8') as fid:
                        reader = csv.reader(fid)
                        for row in reader:
                            line_count += 1
                if line_count==0:
                    with open(opFile, 'a+', encoding='utf8') as fid:
                        writer = csv.writer(fid)
                        writer.writerow(['start','duration'])
                for seg_start in range(0, NSec, spldur):
                    with open(opFile, 'a+', encoding='utf8') as fid:
                        writer = csv.writer(fid)
                        writer.writerow([seg_start, np.min([spldur, NSec-seg_start])])