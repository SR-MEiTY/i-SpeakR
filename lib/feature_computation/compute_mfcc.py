#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 22:41:37 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
import librosa
import os
import csv
import numpy as np
from lib.preprocessing.normalize import Normalize

class MFCC:
    sampling_rate = 0
    frame_length = 0
    hop_length = 0
    n_mels = 0
    n_mfcc = 0
    delta_win = 0
    
    def __init__(self, config):
        self.sampling_rate = int(config['sampling_rate'])
        self.frame_length = int(config['frame_size']*self.sampling_rate/1000)
        self.hop_length = int(config['frame_shift']*self.sampling_rate/1000)
        self.n_mels = int(config['n_mels'])
        self.n_mfcc = int(config['n_mfcc'])
        self.delta_win = int(config['delta_win'])
        self.excl_c0 =config['excl_c0']
    
    def ener_mfcc(self, Xin_split, mfcc):
        astf=abs(librosa.stft(y=Xin_split,n_fft=2048, win_length=self.frame_length, hop_length=self.hop_length, window='hann', center=False))
        sastf=np.square(astf)# squared absolute short term frequency
        enerf=np.sum(sastf,axis=0) # short term energy 
        enerfB=enerf>(0.06*np.mean(enerf)) # voiced frame selection with 6% average eneregy
        mfcc=mfcc[:,enerfB]# selected MFCC frames with energy threshold
        return mfcc

    def compute(self, base_path, meta_info, split_dir, feat_dir, delta=False):
        for sl_no in meta_info.keys():
            fName = meta_info[sl_no]['Local_Path'].split('/')[-1] # meta_info[sl_no]['File Name']
            data_path = base_path + '/' + meta_info[sl_no]['Local_Path']
            if not os.path.exists(data_path):
                print('WAV file does not exist ', data_path)
                continue

            opDir_path = feat_dir + '/'
            if not os.path.exists(opDir_path):
                os.makedirs(opDir_path)
            
            chop_details_fName = split_dir + '/' + '/'.join(meta_info[sl_no]['Local_Path'].split('/')[:-1]) + '/' + fName.split('.')[0] + '.csv'
            if not os.path.exists(chop_details_fName):
                print(f'{chop_details_fName} Utterance chop details unavailable')
                continue

            Xin, fs = librosa.load(data_path, mono=True, sr=self.sampling_rate)
            Xin = Normalize().mean_max_normalize(Xin)
            with open(chop_details_fName, 'r', encoding='utf8') as fid:
                reader = csv.DictReader(fid)
                for row in reader:
                    split_id = row['split_id']
                    first_sample = int(row['first_sample'])
                    last_sample = int(row['last_sample'])
                    duration = float(row['duration'])
            
                    opFile = opDir_path + '/' + split_id + '.npy'
                    if os.path.exists(opFile):
                        print(f'{sl_no}/{len(meta_info.keys())} {fName} {split_id} feature already computed')
                        continue
                    
                    Xin_split = Xin[first_sample:last_sample]
                    
                    if self.excl_c0: # exclude c0 from mfcc computation
                        mfcc = librosa.feature.mfcc(y=Xin_split, sr=fs, n_mfcc=self.n_mfcc+1, dct_type=2, norm='ortho', lifter=0, win_length=self.frame_length, hop_length=self.hop_length, window='hann', center=False, n_mels=self.n_mels)
                        mfcc= mfcc[1:,:] # excluding c0
                    else:
                        mfcc = librosa.feature.mfcc(y=Xin_split, sr=fs, n_mfcc=self.n_mfcc, dct_type=2, norm='ortho', lifter=0, win_length=self.frame_length, hop_length=self.hop_length, window='hann', center=False, n_mels=self.n_mels)
                        
                    if delta:
                        delta_mfcc = librosa.feature.delta(mfcc, width=self.delta_win, order=1, axis=-1)
                        delta_delta_mfcc = librosa.feature.delta(mfcc, width=self.delta_win, order=2, axis=-1)
                        mfcc = np.append(mfcc, delta_mfcc, axis=0)
                        mfcc = np.append(mfcc, delta_delta_mfcc, axis=0)
                    mfcc=self.ener_mfcc(Xin_split, mfcc)
                    np.save(opFile, mfcc)
                    print(f'{sl_no}/{len(meta_info.keys())} {fName} {split_id} mfcc={np.shape(mfcc)} duration={duration}')
                    
        return
    