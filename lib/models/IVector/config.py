"""
Created on 6th September 2022

@authors: 
    Dr. Joyanta Basu, Madhab Pal, Soma Khan, CDAC Kolkata
@collaborators: 
    Jagabandhu Mishra, Ph.D. Scholar, Dept. of EE, IIT Dharwad
    Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad

Ref: SIDEKIT is an open source package for Speaker and Language recognition (https://projets-lium.univ-lemans.fr/sidekit/tutorial/sre10_iv.html)
"""


import os
from os.path import join
import time
import configparser


def time_count(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        result = fn(*args, **kwargs)
        print(">>[Time Count]: Funtion '%s' Costs %fs" % (fn.__name__, time.clock() - start))
        return result

    return _wrapper


def CreatPathIfNotExists(fn):
    def _wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if not os.path.exists(result):
            os.makedirs(result)
        return result

    return _wrapper


class Config:
    def __init__(self):
        self.root = os.getcwd()
        self.parser = configparser.ConfigParser()
        self.parser.read(join(self.root, 'config.ini'))

    @property
    def distrib_nb(self):
        return parser['TRAIN']['distrib_nb']

    @property
    @CreatPathIfNotExists
    def base_dir(self):
        return join(self.root, parser['PATH']['BASE_DIR'])

    @property
    def wave_dir(self):
        return join(self.root,
                    parser['PATH']['BASE_DIR'],
                    parser['PATH']['WAVE_DIR']
                    )

    @property
    def temp_audio(self):
        return join(self.wave_dir, 'temp.wav')

    @property
    @CreatPathIfNotExists
    def feat_dir(self):
        return join(self.root,
                    parser['PATH']['BASE_DIR'],
                    parser['PATH']['FEATURE_DIR']
                    )

    @property
    @CreatPathIfNotExists
    def ubm_dir(self):
        return join(self.root,
                    parser['PATH']['BASE_DIR'],
                    parser['PATH']['UBM_DIR']
                    )

    @property
    def ubm_path(self):
        return join(self.ubm_dir,
                    "ubm_{}.h5".format(int(self.TrainingParamsParser['distrib_nb']))
                    )

    @property
    def stat_path(self):
        return join(self.ubm_dir,
                    "stat_ubm_tv_{}.h5".format(int(self.TrainingParamsParser['distrib_nb']))
                    )

    @property
    @CreatPathIfNotExists
    def data_dir(self):
        return join(self.root,
                    parser['PATH']['BASE_DIR'],
                    parser['PATH']['DATA_DIR']
                    )

    @property
    def tv_path(self):
        return join(self.data_dir,
                    "tv_{}".format(int(self.TrainingParamsParser['distrib_nb']))
                    )

    @property
    @CreatPathIfNotExists
    def get_enroll_feat_dir(self):
        return join(self.root,
                    parser['PATH']['BASE_DIR'],
                    parser['PATH']['FEATURE_DIR'],
                    "enroll"
                    )
    @property
    @CreatPathIfNotExists
    def get_ubm_feat_dir(self):
        return join(self.root,
                    parser['PATH']['BASE_DIR'],
                    parser['PATH']['FEATURE_DIR'],
                    "ubm"
                    )

    @property
    @CreatPathIfNotExists
    def get_test_feat_dir(self):
        return join(self.root,
                    parser['PATH']['BASE_DIR'],
                    parser['PATH']['FEATURE_DIR'],
                    "test"
                    )

    @property
    def get_enroll_data(self):
        return join(self.data_dir,
                    "enroll_data"
                    )

    @property
    def TrainingParamsParser(self):
        return self.parser['TRAIN']

    @property
    def TestingParamsParser(self):
        return self.parser['TEST']

    @property
    def ConfigParser(self):
        return self.parser['CONFIG']


config = Config()
parser = config.parser