#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:32:06 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
import csv
from datetime import datetime
import numpy as np
import os
import librosa
import collections


class GetMetaInfo:
    DATA_INFO = ''
    DATA_PATH = ''
    INFO = {}
    DURATION_FLAG = False
    GENDER_FLAG = False
    TOTAL_DURATION = {}
    GENDER_DISTRIBUTION = {}
    if os.name=='posix': # Linux
        PATH_DELIM = '/'
    elif os.name=='nt': # Windows
        PATH_DELIM = '\\'
    

    def __init__(self, data_info, path, duration=True, gender=True):
        '''
        This function initializes the ReadMetaFile object

        Parameters
        ----------
        path : string
            Path to data or metafile.
        duration : bool, optional
            Flag for computing duration summary of the files in the list. 
            The default is True.
        gender : bool, optional
            Flag for computing speaker gender distribution of the files in the 
            list. The default is True.

        Returns
        -------
        None.

        '''
        self.DURATION_FLAG = duration
        self.GENDER_FLAG = gender
        self.DATA_INFO = data_info
        if self.DATA_INFO=='specify':
            self.DATA_PATH = path
            self.read_info()
            if self.DURATION_FLAG:
                for set_name_ in self.INFO.keys():
                    self.TOTAL_DURATION[set_name_] = self.get_total_duration(set_name_)
            if self.GENDER_FLAG:
                for set_name_ in self.INFO.keys():
                    self.GENDER_DISTRIBUTION[set_name_] = self.get_gender_dist(set_name_)

        elif self.DATA_INFO=='infer':
            self.DATA_PATH = path
            self.get_info()
            

    def read_info(self):
        '''
        Read the meta file and load the information.

        Returns
        -------
        None.

        '''
        csv_files_ = [f_.split(self.PATH_DELIM)[-1] for f_ in librosa.util.find_files(self.DATA_PATH, ext=['csv'])]
        for f_ in csv_files_:
            self.INFO[f_] = {}
            with open(self.DATA_PATH + '/' + f_, 'r' ) as meta_file_:
                reader_ = csv.DictReader(meta_file_)
                for row_ in reader_:
                    utterance_id_ = row_['utterance_id']
                    del row_['utterance_id']
                    if os.name=='posix': # Linux
                        row_['wav_path'] = '/'.join(row_['wav_path'].split('\\'))
                    elif os.name=='nt': # Windows
                        row_['wav_path'] = row_['wav_path'].replace('/', '\\')
                    self.INFO[f_][utterance_id_] = row_
    
    
    def get_info(self):
        '''
        This function is used to infer the different datasets from the path
        provided. The DEV, ENR and TEST folders must contain separate 
        sub-directories for each speaker. The sub-directories must be named 
        according to the speaker IDs. Each speaeker sub-directory may contain 
        multiple utterances. 
        
        Utterance-ID structure:
            <DEV/ENR/TEST>_<Speaker-ID>_<File Name>

        Returns
        -------
        None.

        '''
        if not os.path.exists(self.DATA_PATH+'/DEV/'):
            print('DEV folder does not exist')
            return
        self.INFO['DEV'] = {}
        for speaker_id_ in next(os.walk(self.DATA_PATH+'/DEV/'))[1]:
            for f_ in librosa.util.find_files(self.DATA_PATH + '/DEV/' + speaker_id_ + '/'):
                f_splits_ = f_.split('/')
                part_path_start = np.squeeze(np.where(np.array(f_splits_)=='DEV'))
                f_part_ = '/'.join(f_splits_[part_path_start:])
                row_ = [('speaker_id', speaker_id_), ('wav_path', f_part_)]
                row_ = collections.OrderedDict(row_)
                utterance_id_ = 'DEV_' + speaker_id_ + '_' + f_.split('/')[-1].split('.')[0]
                self.INFO['DEV'][utterance_id_] = row_
        if self.DURATION_FLAG:
            self.TOTAL_DURATION['DEV'] = self.get_total_duration('DEV')
            
        if not os.path.exists(self.DATA_PATH+'/ENR/'):
            print('ENR folder does not exist')
            return
        self.INFO['ENR'] = {}
        for speaker_id in next(os.walk(self.DATA_PATH+'/ENR/'))[1]:
            for f_ in librosa.util.find_files(self.DATA_PATH+'/ENR/'+speaker_id+'/'):
                f_splits_ = f_.split('/')
                part_path_start = np.squeeze(np.where(np.array(f_splits_)=='ENR'))
                f_part_ = '/'.join(f_splits_[part_path_start:])
                row_ = [('speaker_id', speaker_id_), ('wav_path', f_part_)]
                row_ = collections.OrderedDict(row_)
                utterance_id_ = 'ENR_' + speaker_id_ + '_' + f_.split('/')[-1].split('.')[0]
                self.INFO['ENR'][utterance_id_] = row_
        if self.DURATION_FLAG:
            self.TOTAL_DURATION['ENR'] = self.get_total_duration('ENR')

        if not os.path.exists(self.DATA_PATH+'/TEST/'):
            print('TEST folder does not exist')
            return
        self.INFO['TEST'] = {}
        for speaker_id_ in next(os.walk(self.DATA_PATH+'/TEST/'))[1]:
            for f_ in librosa.util.find_files(self.DATA_PATH+'/TEST/'+speaker_id_+'/'):
                f_splits_ = f_.split('/')
                part_path_start = np.squeeze(np.where(np.array(f_splits_)=='TEST'))
                f_part_ = '/'.join(f_splits_[part_path_start:])
                row_ = [('speaker_id', speaker_id_), ('wav_path', f_part_)]
                row_ = collections.OrderedDict(row_)
                utterance_id_ = 'TEST_' + speaker_id_ + '_' + f_.split('/')[-1].split('.')[0]
                self.INFO['TEST'][utterance_id_] = row_
        if self.DURATION_FLAG:
            self.TOTAL_DURATION['TEST'] = self.get_total_duration('TEST')


    def get_total_duration(self, set_name):
        '''
        Compute the duration distribution of the files in the list.

        Returns
        -------
        None.

        '''
        if len(self.INFO[set_name])==0:
            if self.DATA_INFO=='specify':
                self.read_info()
            elif self.DATA_INFO=='infer':
                self.get_info()

        hours_ = 0
        minutes_ = 0
        seconds_ = 0
        for utterance_id_ in self.INFO[set_name].keys():
            try:
                dur = datetime.strptime(self.INFO[set_name][utterance_id_]['Duration'], '%H:%M:%S').time()
                hours_ += dur.hour
                minutes_ += dur.minute
                seconds_ += dur.second
            except:
                Xin_, fs_ = librosa.load(self.DATA_PATH+'/'+self.INFO[set_name][utterance_id_]['wav_path'], mono=True)
                seconds_ += int(len(Xin_)/fs_)
            
        q_, seconds_ = np.divmod(seconds_, 60)
        minutes_ += q_
        q_, minutes_ = np.divmod(minutes_, 60)
        hours_ += q_
        return {'hours':hours_, 'minutes':minutes_, 'seconds':seconds_}
        
        
    def get_gender_dist(self, set_name):
        '''
        Compute the gender distribution of the files in the list.

        Returns
        -------
        None.

        '''
        m_spk_ = [] # keep track of male speaker id
        f_spk_ = [] # keep track of female speaker id
        if len(self.INFO[set_name])==0:
            if self.DATA_INFO=='specify':
                self.read_info()
            elif self.DATA_INFO=='infer':
                self.get_info()
                
        for utterance_id in self.INFO[set_name].keys():
            try:
                if self.INFO[set_name][utterance_id]['gender']=='M':
                    m_spk_.append(self.INFO[set_name][utterance_id]['speaker_id'])
                if self.INFO[set_name][utterance_id]['gender']=='F':
                    f_spk_.append(self.INFO[set_name][utterance_id]['speaker_id'])
            except:
                continue
        return {'Male':len(np.unique(m_spk_)), 'Female':len(np.unique(f_spk_))}