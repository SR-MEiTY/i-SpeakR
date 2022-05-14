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
import librosa
from lib.preprocessing.normalize import Normalize


class ChopUtterances:
    chop_size = []
    sampling_rate = 0
    frame_length = 0
    hop_length = 0
    
    def __init__(self, chop_size, config):
        self.chop_size = chop_size
        self.sampling_rate = config['sampling_rate']
        self.frame_length = int(config['frame_size']*self.sampling_rate/1000)
        self.hop_length = int(config['frame_shift']*self.sampling_rate/1000)
        
    def add_header(self, opFile):
        line_count = 0
        if os.path.exists(opFile):
            with open(opFile, 'r+', encoding='utf8') as fid:
                reader = csv.reader(fid)
                for row in reader:
                    line_count += 1
        if line_count==0:
            with open(opFile, 'a+', encoding='utf8') as fid:
                writer = csv.writer(fid)
                writer.writerow(['split_id', 'first_sample', 'last_sample', 'duration'])
        
    def create_splits(self, meta_info, base_path, splitsDir):
        for sl_no in meta_info.keys():
            fName = meta_info[sl_no]['Local_Path'].split('/')[-1] # meta_info[sl_no]['File Name']
            data_path = base_path + '/' + meta_info[sl_no]['Local_Path']
            if not os.path.exists(data_path):
                print('WAV file does not exist ', data_path)
                continue

            opDir_path = splitsDir + '/' + '/'.join(meta_info[sl_no]['Local_Path'].split('/')[:-1])
            if not os.path.exists(opDir_path):
                os.makedirs(opDir_path)
            opFile = opDir_path + '/' + fName.split('.')[0] + '.csv'
            if os.path.exists(opFile):
                print(f'{fName} chopping details already stored')
                continue

            Xin, fs = librosa.load(data_path, mono=True, sr=self.sampling_rate)
            Xin = Normalize().mean_max_normalize(Xin)
            top_dB = -10*np.log10(0.6*np.mean(Xin**2)) # Threshold is 60% of average energy of the utterance
            nonsil_intervals = librosa.effects.split(Xin, top_db=top_dB, frame_length=self.frame_length, hop_length=self.hop_length)
                
            self.add_header(opFile)
            for spldur in self.chop_size:
                seg_count = 1
                smpStart = 0
                smpEnd = 0
                for intvl_i in range(np.shape(nonsil_intervals)[0]):
                    smpEnd = nonsil_intervals[intvl_i,1]
                    if intvl_i==np.shape(nonsil_intervals)[0]:
                        smpEnd = len(Xin)
                    if (smpEnd-smpStart)/fs>spldur:
                        with open(opFile, 'a+', encoding='utf8') as fid:
                            writer = csv.writer(fid)
                            split_id = meta_info[sl_no]['Uttrence_ID'] + '_' + str(spldur) + '_' + format(seg_count, '03d')
                            writer.writerow([split_id, smpStart, smpEnd, np.round((smpEnd-smpStart)/fs,2)])
                        seg_count += 1
                        smpStart = smpEnd
                    else:
                        continue
            print(f'{sl_no}/{len(meta_info.keys())} {fName} chop details stored')