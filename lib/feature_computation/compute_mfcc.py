#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 22:41:37 2022

@author: 
    Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
@collaborators: 
    Jagabandhu Mishra, Ph.D. Scholar, Dept. of EE, IIT Dharwad
"""

import librosa
import os
import csv
import numpy as np
from lib.preprocessing.normalize import Normalize

class MFCC:
    SAMPLING_RATE = 0
    NFFT = 0
    FRAME_LENGTH = 0
    HOP_LENGTH = 0
    N_MELS = 0
    N_MFCC = 0
    DELTA_WIN = 0
    
    def __init__(self, config):
        self.SAMPLING_RATE = int(config['sampling_rate'])
        self.NFFT = int(config['nfft'])
        self.FRAME_LENGTH = int(int(config['frame_size'])*self.SAMPLING_RATE/1000)
        self.HOP_LENGTH = int(int(config['frame_shift'])*self.SAMPLING_RATE/1000)
        self.N_MELS = int(config['n_mels'])
        self.N_MFCC = int(config['n_mfcc'])
        self.DELTA_WIN = int(config['delta_win'])
        self.EXCL_C0 = config['excl_c0']
        self.DATA_INFO_DIR = config['data_info_dir']
        self.FEAT_DIR = config['feat_dir']
        self.DEV_KEY = config['dev_key']
        self.ENR_KEY = config['enr_key']
        self.TEST_KEY = config['test_key']
        self.DELTA_FEAT = config.getboolean('compute_delta_feat')
    
    
    def ener_mfcc(self, y, mfcc):
        '''
        Selection of MFCC feature vectors based on VAD threshold of 60% of
        average energy

        Parameters
        ----------
        y : 1D array
            Audio samples for sub-utterance.
        mfcc : 2D array
            MFCC feature vectors for the sub-utterance.

        Returns
        -------
        mfcc : 2D array
            Voiced frame MFCC feature vectors selected based on energy 
            threshold.

        '''
        astf_ = np.abs(librosa.stft(y=y, n_fft=self.NFFT, win_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH, window='hann', center=False))
        # squared absolute short term frequency
        sastf_ = np.square(astf_)
        # short term energy 
        enerf_ = np.sum(sastf_,axis=0)
        # voiced frame selection with 6% average eneregy
        enerfB_ = enerf_>(0.06*np.mean(enerf_))
        # enerfB_ = enerf_>(0.6*np.mean(enerf_))
        # selected MFCC frames with energy threshold
        voiced_mfcc_ = mfcc[:,enerfB_]
        return voiced_mfcc_


    def compute(self):
        '''
        Computing the MFCC features

        Feature filename structure:
            <Feature folder>/<Split-ID>.npy
            
        Split-ID structure:
            <Utterance-ID>_<Chop Size>_<Split count formatted as a 3-digit number>

        Utterance-ID structure:
            "infer" mode:
                <DEV/ENR/TEST>_<Speaker-ID>_<File Name>
            "specify" mode:
                <Speaker-ID>_<File Name>

        Returns
        -------
        feature_details_ : dict 
            A dictionary containing the information of all features. The
            following name-value pairs are available:
                'DEV': {split_id:{'feature_name':<>, 'utterance_id':<>, 'file_path':<>, 'speaker_id':<>}}
                'ENR': {split_id:{'feature_name':<>, 'utterance_id':<>, 'file_path':<>, 'speaker_id':<>}}
                'TEST': {split_id:{'feature_name':<>, 'utterance_id':<>, 'file_path':<>, 'speaker_id':<>}}

        '''
        feature_details_ = {}
        for key_fName_ in [self.DEV_KEY, self.ENR_KEY, self.TEST_KEY]:
            if key_fName_==self.DEV_KEY:
                data_type_ = 'DEV'
            elif key_fName_==self.ENR_KEY:
                data_type_ = 'ENR'
            elif key_fName_==self.TEST_KEY:
                data_type_ = 'TEST'
            
            feature_details_[data_type_] = {}
            
            data_info_fName_ = self.DATA_INFO_DIR + '/' + key_fName_.split('/')[-1]
            print(f'data_info_fName: {data_info_fName_}')
            
            nFiles_ = 0
            with open(data_info_fName_, 'r') as f_:
                nFiles_ = len(f_.readlines())
                
            with open(data_info_fName_, 'r') as meta_file_:
                reader_ = csv.DictReader(meta_file_)
                utter_count_ = 0
                for row_ in reader_:
                    # utterance_id_ = row_['utterance_id']
                    speaker_id_ = row_['speaker_id']
                    split_id_ = row_['split_id']
                    data_path_ = row_['wav_path']
                    fName_ = data_path_.split('/')[-1]
                    first_sample_ = int(row_['first_sample'])
                    last_sample_ = int(row_['last_sample'])
                    
                    if not os.path.exists(data_path_):
                        print('\tWAV file does not exist ', data_path_)
                        continue
                    
                    if data_type_.startswith('DEV'):
                        opDir_path_ = self.FEAT_DIR + '/DEV/'
                    if data_type_.startswith('ENR'):
                        opDir_path_ = self.FEAT_DIR + '/ENR/'
                    if data_type_.startswith('TEST'):
                        opDir_path_ = self.FEAT_DIR + '/TEST/'
                    if not os.path.exists(opDir_path_):
                        os.makedirs(opDir_path_)

                    opFile_ = opDir_path_ + '/' + split_id_ + '.npy'
                    feature_details_[data_type_][split_id_] = {
                        'feature_name': 'MFCC', 
                        'utterance_id': row_['utterance_id'], 
                        'file_path': opFile_, 
                        'speaker_id': speaker_id_,
                        }

                    # Check if feature file already exists
                    if os.path.exists(opFile_):
                        continue
                    
                    sys_state_ = locals()
                    if 'utterance_id_' in sys_state_:
                        if not sys_state_['utterance_id_']==row_['utterance_id']:
                            Xin_, fs_ = librosa.load(data_path_, mono=True, sr=self.SAMPLING_RATE)
                            Xin_ = Normalize().mean_max_normalize(Xin_)
                            utterance_id_ = row_['utterance_id']
                    else:
                        utterance_id_ = row_['utterance_id']
                        Xin_, fs_ = librosa.load(data_path_, mono=True, sr=self.SAMPLING_RATE)
                        Xin_ = Normalize().mean_max_normalize(Xin_)
                        
                    Xin_split_ = None
                    del Xin_split_
                    Xin_split_ = np.array(Xin_[first_sample_:last_sample_], copy=True)
                    if len(Xin_split_)<=self.NFFT:
                        del feature_details_[data_type_][split_id_]
                        continue
                    
                    mfcc_ = None
                    del mfcc_
                    if self.EXCL_C0: # Exclude c0 from mfcc computation
                        mfcc_ = librosa.feature.mfcc(y=Xin_split_, sr=fs_, n_mfcc=self.N_MFCC+1, dct_type=2, norm='ortho', lifter=0, n_fft=self.NFFT, win_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH, window='hann', center=False, n_mels=self.N_MELS)
                        mfcc_ = mfcc_[1:,:] # excluding c0
                    else:
                        mfcc_ = librosa.feature.mfcc(y=Xin_split_, sr=fs_, n_mfcc=self.N_MFCC, dct_type=2, norm='ortho', lifter=0, n_fft=self.NFFT, win_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH, window='hann', center=False, n_mels=self.N_MELS)

                    if np.shape(mfcc_)[1]<=self.DELTA_WIN:
                        del feature_details_[data_type_][split_id_]
                        continue
                        
                    if self.DELTA_FEAT:
                        delta_mfcc_ = librosa.feature.delta(mfcc_, width=self.DELTA_WIN, order=1, axis=-1)
                        delta_delta_mfcc_ = librosa.feature.delta(mfcc_, width=self.DELTA_WIN, order=2, axis=-1)
                        mfcc_ = np.append(mfcc_, delta_mfcc_, axis=0)
                        mfcc_ = np.append(mfcc_, delta_delta_mfcc_, axis=0)
                    
                    # Selection of voiced frames
                    voiced_mfcc_ = self.ener_mfcc(Xin_split_, mfcc_)
                    np.save(opFile_, voiced_mfcc_)
                    
                    utter_count_ += 1
                    print(f'\t({utter_count_}/{nFiles_})\t{split_id_} MFCC shape={np.shape(voiced_mfcc_)}')
                        
        return feature_details_



    def get_feature_details(self):
        '''
        Returns
        -------
        feature_details_ : dict 
            A dictionary containing the information of all features. The
            following name-value pairs are available:
                'DEV': {split_id:{'feature_name':<>, 'utterance_id':<>, 'file_path':<>, 'speaker_id':<>}}
                'ENR': {split_id:{'feature_name':<>, 'utterance_id':<>, 'file_path':<>, 'speaker_id':<>}}
                'TEST': {split_id:{'feature_name':<>, 'utterance_id':<>, 'file_path':<>, 'speaker_id':<>}}

        '''
        feature_details_ = {}
        for key_fName_ in [self.DEV_KEY, self.ENR_KEY, self.TEST_KEY]:
            if key_fName_==self.DEV_KEY:
                data_type_ = 'DEV'
            elif key_fName_==self.ENR_KEY:
                data_type_ = 'ENR'
            elif key_fName_==self.TEST_KEY:
                data_type_ = 'TEST'
            
            feature_details_[data_type_] = {}
            
            data_info_fName_ = self.DATA_INFO_DIR + '/' + key_fName_.split('/')[-1]
            print(f'data_info_fName: {data_info_fName_}')

            with open(data_info_fName_, 'r') as meta_file_:
                reader_ = csv.DictReader(meta_file_)
                for row_ in reader_:
                    speaker_id_ = row_['speaker_id']
                    utterance_id_ = row_['utterance_id']
                    split_id_ = row_['split_id']
                    
                    if data_type_.startswith('DEV'):
                        opDir_path_ = self.FEAT_DIR + '/DEV/'
                    if data_type_.startswith('ENR'):
                        opDir_path_ = self.FEAT_DIR + '/ENR/'
                    if data_type_.startswith('TEST'):
                        opDir_path_ = self.FEAT_DIR + '/TEST/'
                    if not os.path.exists(opDir_path_):
                        os.makedirs(opDir_path_)

                    opFile_ = opDir_path_ + '/' + split_id_ + '.npy'

                    # Check if feature file already exists
                    if os.path.exists(opFile_):
                        feature_details_[data_type_][split_id_] = {
                            'feature_name': 'MFCC', 
                            'utterance_id': utterance_id_, 
                            'file_path': opFile_, 
                            'speaker_id': speaker_id_,
                            }
                        
        return feature_details_
    