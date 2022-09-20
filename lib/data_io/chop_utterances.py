#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:14:22 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""

import os
import csv
import numpy as np
import librosa
from lib.preprocessing.normalize import Normalize


class ChopUtterances:
    SAMPLING_RATE = 0
    FRAME_LENGTH = 0
    HOP_LENGTH = 0
    DATA_INFO_DIR = ''
    DEV_CHOP_SIZE = []
    ENR_CHOP_SIZE = []
    TEST_CHOP_SIZE = []
    
    def __init__(self, config):
        self.DEV_CHOP_SIZE.extend(config['DEV_CHOP'])
        self.ENR_CHOP_SIZE.extend(config['ENR_CHOP'])
        self.TEST_CHOP_SIZE.extend(config['TEST_CHOP'])
        self.SAMPLING_RATE = config['SAMPLING_RATE']
        self.FRAME_LENGTH = int(config['FRAME_SIZE']*self.SAMPLING_RATE/1000)
        self.HOP_LENGTH = int(config['FRAME_SHIFT']*self.SAMPLING_RATE/1000)
        self.DATA_INFO_DIR = config['DATA_INFO_DIR']
            
            
    def get_non_silence_intervals(self, file_path):
        '''
        Load the specified wav file and compute non-silent intervals using a 
        VAD threshold of 60% of average utterance energy

        Parameters
        ----------
        file_path : str
            Path of the wav file.

        Returns
        -------
        nonsil_intervals : array
            Array contains the start and end samples of consecutive non-silent
            intervals.

        '''
        # Load the wav file and perform VAD
        Xin_, fs_ = librosa.load(file_path, mono=True, sr=self.SAMPLING_RATE)
        Xin_ = Normalize().mean_max_normalize(Xin_)
        # Threshold is 60% of average energy of the utterance
        top_dB_ = -10*np.log10(0.6*np.mean(Xin_**2)) 
        nonsil_intervals_ = librosa.effects.split(Xin_, top_db=top_dB_, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH)
        
        return nonsil_intervals_, len(Xin_)
        
        
    def add_header(self, opFile):
        '''
        Adding header to the csv file that stores the utterrance wise chopping
        details.

        Parameters
        ----------
        opFile : str
            Path to the csv file.

        Returns
        -------
        None.

        '''
        line_count_ = 0
        if os.path.exists(opFile):
            with open(opFile, 'r+', encoding='utf8') as fid_:
                reader_ = csv.reader(fid_)
                for row_ in reader_:
                    line_count_ += 1
                    
        # If the file is empty, then add a header line
        if line_count_==0: 
            with open(opFile, 'a+', encoding='utf8') as fid_:
                writer_ = csv.writer(fid_)
                # The csv file has four columns
                writer_.writerow([
                    'speaker_id',
                    'utterance_id',
                    'split_id', 
                    'first_sample', 
                    'last_sample', 
                    'duration',
                    'wav_path',
                    'cohorts'
                    ])
        
        
    def create_splits(self, meta_info, key):
        '''
        Parse the meta_info object to obtain the paths to all wav files for 
        DEV, ENR and TEST sets. Record the split durations for each wav file 
        from the respective sets based on the predefined chop sizes for each
        set. 
        
        Split-ID structure:
            <Utterance-ID>_<Chop Size>_<Split count formatted as a 3-digit number>
        
        Utterance-ID structure:
            "infer" mode:
                <DEV/ENR/TEST>_<Speaker-ID>_<File Name>
            "specify" mode:
                <Speaker-ID>_<File Name>

        Parameters
        ----------
        meta_info : dict
            Contains the dataset details.
        path : str
            Base path to the dataset.

        Returns
        -------
        None.

        '''
        for data_type_ in meta_info.keys():
            chop_size_ = None
            if data_type_.startswith('DEV'):
                opFile_ = self.DATA_INFO_DIR + '/' + key['DEV'].split('/')[-1]
                chop_size_ = self.DEV_CHOP_SIZE
            elif data_type_.startswith('ENR'):
                opFile_ = self.DATA_INFO_DIR + '/' + key['ENR'].split('/')[-1]
                chop_size_ = self.ENR_CHOP_SIZE
            elif data_type_.startswith('TEST'):
                opFile_ = self.DATA_INFO_DIR + '/' + key['TEST'].split('/')[-1]
                chop_size_ = self.TEST_CHOP_SIZE

            if os.path.exists(opFile_):
                continue
            
            utter_count_ = 0
            csv_output_string_ = ''
            for utterance_id_ in meta_info[data_type_].keys():
                utter_count_ += 1
                # print(f'{data_type_} {utterance_id_} ({utter_count_}/{len(meta_info[data_type_].keys())})')
                fName_ = meta_info[data_type_][utterance_id_]['wav_path'].split('/')[-1]
                data_path_ = meta_info[data_type_][utterance_id_]['wav_path']
                speaker_id_ = meta_info[data_type_][utterance_id_]['speaker_id']
                if data_type_.startswith('TEST'):
                    cohorts_ = '|'.join([str(meta_info[data_type_][utterance_id_]['c'+str(i)]) for i in range(1,6)])
                else:
                    cohorts_ = ''
                if not os.path.exists(data_path_):
                    print('\tWAV file does not exist ', data_path_)
                    continue

                nonsil_intervals_, nSamples_ = self.get_non_silence_intervals(data_path_)

                self.add_header(opFile_)

                # If spldur_=='x', chopping is not done
                if -1 in chop_size_:
                    with open(opFile_, 'a+', encoding='utf8') as fid_:
                        writer_ = csv.writer(fid_)
                        split_id_ = utterance_id_ + '_x_000'
                        writer_.writerow([
                            speaker_id_,
                            utterance_id_,
                            split_id_, 
                            0, 
                            nSamples_, 
                            np.round(nSamples_/self.SAMPLING_RATE,2),
                            data_path_,
                            cohorts_
                            ])
                    
                    # split_id_ = utterance_id_ + '_x_000'
                    # csv_output_string_ += speaker_id_ + ',' + utterance_id_ + ',' + split_id_ + ',' + '0' + ',' + str(nSamples_) + ',' + str(np.round(nSamples_/self.SAMPLING_RATE,2)) + ',' + data_path_ + ',' + cohorts_ + '\n'
                    print(f'\t({utter_count_}/{len(meta_info[data_type_])}) {fName_} not chopped')
                    continue                    
                    
                # Compute the sub-utterances and store the details in csv files
                for spldur_ in chop_size_:
                    seg_count_ = 1
                    smpStart_ = 0
                    smpEnd_ = 0
                    for intvl_i_ in range(np.shape(nonsil_intervals_)[0]):
                        smpEnd_ = nonsil_intervals_[intvl_i_,1]
                        if intvl_i_==np.shape(nonsil_intervals_)[0]:
                            smpEnd_ = nSamples_

                        if ((smpEnd_-smpStart_)/self.SAMPLING_RATE>spldur_) or (smpEnd_==nSamples_):
                            with open(opFile_, 'a+', encoding='utf8') as fid_:
                                writer_ = csv.writer(fid_)
                                split_id_ = utterance_id_ + '_' + str(spldur_) + '_' + format(seg_count_, '03d')
                                writer_.writerow([
                                    speaker_id_,
                                    utterance_id_,
                                    split_id_, 
                                    smpStart_, 
                                    smpEnd_, 
                                    np.round((smpEnd_-smpStart_)/self.SAMPLING_RATE,2),
                                    data_path_,
                                    cohorts_
                                    ])
                                
                            # split_id_ = utterance_id_ + '_' + str(spldur_) + '_' + format(seg_count_, '03d')
                            # csv_output_string_ += speaker_id_ + ',' + utterance_id_ + ',' + split_id_ + ',' + str(smpStart_) + ',' + str(smpEnd_) + ',' + str(np.round((smpEnd_-smpStart_)/self.SAMPLING_RATE,2)) + ',' + data_path_ + ',' + cohorts_ + '\n'
                            
                            seg_count_ += 1
                            smpStart_ = smpEnd_
                        else:
                            continue
                        
                    print(f'\t({utter_count_}/{len(meta_info[data_type_])}) {fName_} chop details stored for duration={spldur_}s')

            with open(opFile_, 'w+', encoding='utf8') as fid_:
                fid_.write(csv_output_string_)
                
