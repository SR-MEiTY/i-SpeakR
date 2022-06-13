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
    SAMPLING_RATE = 0
    NFFT = 0
    FRAME_LENGTH = 0
    HOP_LENGTH = 0
    N_MELS = 0
    N_MFCC = 0
    DELTA_WIN = 0
    
    def __init__(self, config):
        self.SAMPLING_RATE = int(config['SAMPLING_RATE'])
        self.NFFT = int(config['NFFT'])
        self.FRAME_LENGTH = int(config['FRAME_SIZE']*self.SAMPLING_RATE/1000)
        self.HOP_LENGTH = int(config['FRAME_SHIFT']*self.SAMPLING_RATE/1000)
        self.N_MELS = int(config['N_MELS'])
        self.N_MFCC = int(config['N_MFCC'])
        self.DELTA_WIN = int(config['DELTA_WIN'])
        self.EXCL_C0 = config['EXCL_C0']
    
    
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


    def compute(self, base_path, meta_info, split_dir, feat_dir, delta=False):
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

        Parameters
        ----------
        base_path : str
            Path to dataset root.
        meta_info : dict
            Dataset information.
        split_dir : str
            Path to details of utterance chopping.
        feat_dir : str
            Path to store the feature vectors.
        delta : bool, optional
            Flag to indicate whether delta features are to be computed. 
            The default is False.

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
        for data_type_ in meta_info.keys():
            feature_details_[data_type_] = {}
            utter_count_ = 0
            for utterance_id_ in meta_info[data_type_].keys():
                # print(f'{data_type_}\t{utterance_id_}\t({utter_count_}/{len(meta_info[data_type_].keys())}):')
                fName_ = meta_info[data_type_][utterance_id_]['wav_path'].split('/')[-1]
                data_path_ = base_path + '/' + meta_info[data_type_][utterance_id_]['wav_path']
                
                # speaker_id_ = utterance_id_.split('_')[1] # This way of obtaining speaker_id was wrong. Corrected on  01-Jun-22 
                speaker_id_ = meta_info[data_type_][utterance_id_]['speaker_id']
                
                if not os.path.exists(data_path_):
                    print('\tWAV file does not exist ', data_path_)
                    continue
                
                opDir_path_ = feat_dir + '/'
                if not os.path.exists(opDir_path_):
                    os.makedirs(opDir_path_)
                
                chop_details_fName_ = split_dir + '/' + '/'.join(meta_info[data_type_][utterance_id_]['wav_path'].split('/')[:-1]) + '/' + fName_.split('.')[0] + '.csv'
                if not os.path.exists(chop_details_fName_):
                    print(f'\t{chop_details_fName_} Utterance chop details unavailable')
                    continue

                # Xin_, fs_ = librosa.load(data_path_, mono=True, sr=self.SAMPLING_RATE)
                # Xin_ = Normalize().mean_max_normalize(Xin_)
                Xin_ = None
                del Xin_
                
                utter_count_ += 1
                with open(chop_details_fName_, 'r', encoding='utf8') as fid_:
                    reader_ = csv.DictReader(fid_)
                    for row_ in reader_:
                        split_id_ = row_['split_id']
                        first_sample_ = int(row_['first_sample'])
                        last_sample_ = int(row_['last_sample'])
                        # duration_ = float(row_['duration'])
                
                        opFile_ = opDir_path_ + '/' + split_id_ + '.npy'
                        feature_details_[data_type_][split_id_] = {
                            'feature_name': 'MFCC', 
                            'utterance_id': utterance_id_, 
                            'file_path': opFile_, 
                            'speaker_id': speaker_id_
                            }

                        # Check if feature file already exists
                        if os.path.exists(opFile_):
                            # print(f'\t{split_id_} feature available')
                            continue
                        
                        if not 'Xin_' in locals(): # Check if the wav has already been loaded
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
                            
                        if delta:
                            delta_mfcc_ = librosa.feature.delta(mfcc_, width=self.DELTA_WIN, order=1, axis=-1)
                            delta_delta_mfcc_ = librosa.feature.delta(mfcc_, width=self.DELTA_WIN, order=2, axis=-1)
                            mfcc_ = np.append(mfcc_, delta_mfcc_, axis=0)
                            mfcc_ = np.append(mfcc_, delta_delta_mfcc_, axis=0)
                        
                        # Selection of voiced frames
                        voiced_mfcc_ = self.ener_mfcc(Xin_split_, mfcc_)
                        np.save(opFile_, voiced_mfcc_)
                        
                        print(f'\t({utter_count_}/{len(meta_info[data_type_].keys())})\t{split_id_} MFCC feature_shape={np.shape(voiced_mfcc_)}')
                        
        return feature_details_
    