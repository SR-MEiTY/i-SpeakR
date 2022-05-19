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
    data_info = ''
    meta_path = ''
    base_path = ''
    info = {}
    duration_flag = False
    gender_flag = False
    total_duration = {}
    gender_distribution = {}
    if os.name=='posix': # Linux
        path_delim = '/'
    elif os.name=='nt': # Windows
        path_delim = '\\'
    

    def __init__(self, data_info, path, duration=True, gender=True):
        '''
        This function initializes the ReadMetaFile object

        Parameters
        ----------
        path : string
            Path to metafile.
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
        self.duration_flag = duration
        self.gender_flag = gender
        self.data_info = data_info
        if self.data_info=='specify':
            self.meta_path = path
            self.read_info()
            if self.duration_flag:
                for set_name in self.info.keys():
                    self.total_duration[set_name] = self.get_total_duration(set_name)
            if self.gender_flag:
                for set_name in self.info.keys():
                    self.gender_distribution[set_name] = self.get_gender_dist(set_name)

        elif self.data_info=='infer':
            self.base_path = path
            self.get_info()
            

    def read_info(self):
        '''
        Read the meta file and load the information.

        Returns
        -------
        None.

        '''
        csv_files = [f.split(self.path_delim)[-1] for f in librosa.util.find_files(self.meta_path, ext=['csv'])]
        for f in csv_files:
            self.info[f] = {}
            with open(self.meta_path+'/'+f, 'r' ) as meta_file:
                reader = csv.DictReader(meta_file)
                for row in reader:
                    utterance_id = row['utterance_id']
                    del row['utterance_id']
                    if os.name=='posix': # Linux
                        row['wav_path'] = '/'.join(row['wav_path'].split('\\'))
                    elif os.name=='nt': # Windows
                        row['wav_path'] = row['wav_path'].replace('/', '\\')
                    self.info[f][utterance_id] = row
    
    
    def get_info(self):
        if not os.path.exists(self.base_path+'/DEV/'):
            print('DEV folder does not exist')
            return
        self.info['DEV'] = {}
        utterance_id = 0
        for f in librosa.util.find_files(self.base_path+'/DEV/'):
            row = [('speaker_id', str(utterance_id)), ('wav_path', f)]
            row = collections.OrderedDict(row)
            self.info['DEV'][str(utterance_id)] = row
            utterance_id += 1
        if self.duration_flag:
            self.total_duration['DEV'] = self.get_total_duration('DEV')
            
        if not os.path.exists(self.base_path+'/ENR/'):
            print('ENR folder does not exist')
            return
        self.info['ENR'] = {}
        utterance_id = 0
        for f in librosa.util.find_files(self.base_path+'/ENR/'):
            row = [('speaker_id', str(utterance_id)), ('wav_path', f)]
            row = collections.OrderedDict(row)
            self.info['ENR'][str(utterance_id)] = row
            utterance_id += 1
        if self.duration_flag:
            self.total_duration['ENR'] = self.get_total_duration('ENR')

        if not os.path.exists(self.base_path+'/TEST/'):
            print('TEST folder does not exist')
            return
        self.info['TEST'] = {}
        utterance_id = 0
        for f in librosa.util.find_files(self.base_path+'/TEST/'):
            row = [('speaker_id', str(utterance_id)), ('wav_path', f)]
            row = collections.OrderedDict(row)
            self.info['TEST'][str(utterance_id)] = row
            utterance_id += 1
        if self.duration_flag:
            self.total_duration['TEST'] = self.get_total_duration('TEST')


    def get_total_duration(self, set_name):
        '''
        Compute the duration distribution of the files in the list.

        Returns
        -------
        None.

        '''
        if len(self.info[set_name])==0:
            if self.data_info=='specify':
                self.read_info()
            elif self.data_info=='infer':
                self.get_info()

        hours = 0
        minutes = 0
        seconds = 0
        for utterance_id in self.info[set_name].keys():
            try:
                dur = datetime.strptime(self.info[set_name][utterance_id]['Duration'], '%H:%M:%S').time()
                hours += dur.hour
                minutes += dur.minute
                seconds += dur.second
            except:
                Xin, fs = librosa.load(self.info[set_name][utterance_id]['wav_path'], mono=True)
                seconds += int(len(Xin)/fs)
            
        q, seconds = np.divmod(seconds, 60)
        minutes += q
        q, minutes = np.divmod(minutes, 60)
        hours += q
        return {'hours':hours, 'minutes':minutes, 'seconds':seconds}
        
        
    def get_gender_dist(self, set_name):
        '''
        Compute the gender distribution of the files in the list.

        Returns
        -------
        None.

        '''
        m_spk=[] # keep track of male speaker id
        f_spk=[] #keep track of female speaker id
        if len(self.info[set_name])==0:
            if self.data_info=='specify':
                self.read_info()
            elif self.data_info=='infer':
                self.get_info()
                
        for utterance_id in self.info[set_name].keys():
            try:
                if self.info[set_name][utterance_id]['gender']=='M':
                    m_spk.append(self.info[set_name][utterance_id]['speaker_id'])
                if self.info[set_name][utterance_id]['gender']=='F':
                    f_spk.append(self.info[set_name][utterance_id]['speaker_id'])
            except:
                continue
        return {'Male':len(np.unique(m_spk)), 'Female':len(np.unique(f_spk))}