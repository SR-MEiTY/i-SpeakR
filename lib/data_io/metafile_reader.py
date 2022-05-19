#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:32:06 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
import csv
from datetime import datetime
import numpy as np

class ReadMetaFile:
    meta_path = ''
    os_type = ''
    info = {}
    total_duration = None
    gender_distribution = {'Male':None, 'Female':None}
    
    def __init__(self, path, os, duration=True, gender=True):
        self.meta_path = path
        self.os_type = os
        self.read_info()
        if duration:
            self.get_total_duration()
        if gender:
            self.get_gender_dist()
        
    def read_info(self):
        with open(self.meta_path, 'r' ) as meta_file:
            reader = csv.DictReader(meta_file)
            for row in reader:
                sl_no = int(row['Sl.No'])
                del row['Sl.No']
                if self.os_type=='Linux':
                    row['Local_Path'] = '/'.join(row['Local_Path'].split('\\'))
                self.info[sl_no] = row

    def get_total_duration(self):
        if len(self.info)==0:
            self.read_info()
        hours = 0
        minutes = 0
        seconds = 0
        for sl_no in self.info.keys():
            dur = datetime.strptime(self.info[sl_no]['Duration'], '%H:%M:%S').time()
            hours += dur.hour
            minutes += dur.minute
            seconds += dur.second
        q, seconds = np.divmod(seconds, 60)
        minutes += q
        q, minutes = np.divmod(minutes, 60)
        hours += q
        self.total_duration = {'hours':hours, 'minutes':minutes, 'seconds':seconds}
        
    def get_gender_dist(self):
        if len(self.info)==0:
            self.read_info()
        male = 0
        female = 0
        for sl_no in self.info.keys():
            if self.info[sl_no]['Gender']=='M':
                male += 1
            if self.info[sl_no]['Gender']=='F':
                female += 1
        self.gender_distribution['Male'] = male
        self.gender_distribution['Female'] = female