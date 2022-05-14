#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:34:50 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
import librosa
import os
import csv
import numpy as np
from lib.preprocessing.normalize import Normalize

class VAD:
    sampling_rate = 0
    frame_length = 0
    hop_length = 0
    
    def __init__(self, config):
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
                writer.writerow(['start_sample','end_sample'])
    
    def find_silences(self, base_path, meta_info, vad_dir):
        for sl_no in meta_info.keys():
            fName = meta_info[sl_no]['Local_Path'].split('/')[-1] # meta_info[sl_no]['File Name']
            data_path = base_path + '/' + meta_info[sl_no]['Local_Path']
            if not os.path.exists(data_path):
                print('WAV file does not exist ', data_path)
                continue

            opDir_path = vad_dir + '/' + '/'.join(meta_info[sl_no]['Local_Path'].split('/')[:-1])
            if not os.path.exists(opDir_path):
                os.makedirs(opDir_path)
            opFile = opDir_path + '/' + fName.split('.')[0] + '.csv'
            if os.path.exists(opFile):
                print(f'{fName} VAD details already stored')
                continue
            
            Xin, fs = librosa.load(data_path, mono=True, frame_length=self.frame_length, hop_length=self.hop_length, sr=self.sampling_rate)
            Xin = Normalize().mean_max_normalize(Xin)
            intervals = librosa.effects.split(Xin)
            print(f'intervals: {intervals} {np.min(Xin)} {np.max(Xin)}')
            
            self.add_header(opFile)
            seg_count = 1
            for i in range(np.shape(intervals)[0]):
                with open(opFile, 'a+', encoding='utf8') as fid:
                    writer = csv.writer(fid)
                    writer.writerow([intervals[i,0], intervals[i,1]])
                    seg_count += 1            
        
        return
    