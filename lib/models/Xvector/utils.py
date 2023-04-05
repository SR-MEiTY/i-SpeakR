# Third Party
import librosa
import numpy as np
import os
import pandas as pd
import random
# ===============================================
#       code from Arsha for loading data.
# This code extract features for a give audio file
# ===============================================
def load_wav(audio_filepath, sr, input_length_time=60):
    data,fs  = librosa.load(audio_filepath, sr=sr)
    input_length = input_length_time * sr
    #len_file = len(audio_data)
    #if len_file <int(min_dur_sec*sr):
    #    dummy=np.zeros((1,int(min_dur_sec*sr)-len_file))
    #    extened_wav = np.concatenate((audio_data,dummy[0]))
    #else:
        
    #    extened_wav = audio_data
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
    return data

def lin_mel_from_wav(wav, hop_length, win_length, n_mels):
    linear = librosa.feature.melspectrogram(wav, n_mels=n_mels, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=512):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def lin_mfcc_from_wav(wav, hop_length, win_length, n_mfcc, n_fft=512):
    linear = librosa.feature.mfcc(y=wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mfcc=n_mfcc) # mfcc
    return linear.T


def feature_extraction(filepath,sr=16000, min_dur_sec=4,win_length=400,hop_length=80, n_mfcc=40, n_mels=40, spec_len=400,mode='dev'):
    audio_data = load_wav(filepath, sr=sr,input_length_time=60)
    #linear_spect = lin_spectogram_from_wav(audio_data, hop_length, win_length, n_fft=512)
    linear_spect = lin_mfcc_from_wav(audio_data, hop_length, win_length, n_mfcc, n_fft=512)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    mu = np.mean(mag_T, 0, keepdims=True)
    std = np.std(mag_T, 0, keepdims=True)
    return (mag_T - mu) / (std + 1e-5)
    
    
    
    
def load_data(filepath,sr=16000, min_dur_sec=4,win_length=400,hop_length=80, n_fft=400, n_mels=40, spec_len=400,n_mfcc=40, mode='train'):
    audio_data = load_wav(filepath, sr=sr, min_dur_sec=min_dur_sec)
    #linear_spect = lin_spectogram_from_wav(audio_data, hop_length, win_length, n_mels)
    #linear_spect = lin_spectogram_from_wav(audio_data, hop_length, win_length, n_fft=n_fft)
    linear_spect = lin_mfcc_from_wav(audio_data, hop_length, win_length, n_mfcc, n_fft=512)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    
    if mode=='dev':
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)
    


def load_npy_data(filepath,spec_len=400,mode='train'):
    mag_T = np.load(filepath)
    if mode=='train':
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    return spec_mag
    
    



def speech_collate(batch):
    targets = []
    specs = []
    paths = []
    for sample in batch:
        specs.append(sample['features'])
        targets.append((sample['labels']))
        paths.append((sample['path']))
    return specs, targets, paths


def writeXvectors(xvec_feat_path , df, labels, x_vec, paths):
    for i in range(len(labels)):

        xvec_feat_label_dir = os.path.join(xvec_feat_path, str(labels[i]))

        if not os.path.exists(xvec_feat_label_dir):
            os.mkdir(xvec_feat_label_dir)
        filename = os.path.basename(os.path.splitext(paths[i])[0])
        filename = filename + ".npy"
        xvec_feat_file_path = os.path.join(xvec_feat_label_dir, filename)
        np.save(xvec_feat_file_path, x_vec[i].cpu().detach().numpy())
        newPd = pd.DataFrame({'x_vec_path': [xvec_feat_file_path], 'label': [labels[i]]})
        df = pd.concat((df, newPd), axis=0)
    return df


def writeFeatures(feat_path, labels, features, paths, mode):
    file_path = os.path.join(feat_path, mode + ".text")
    lines = []
    for i in range(len(labels)):
            feat_label_dir = os.path.join(feat_path, str(labels[i]))
            if not os.path.exists(feat_label_dir):
                os.mkdir(feat_label_dir)
            filename = os.path.basename(os.path.splitext(paths[i])[0])
            filename = filename + ".npy"
            feat_file_path = os.path.join(feat_label_dir, filename)
            np.save(feat_file_path, features[i].cpu().detach().numpy())
            lines.append(feat_file_path + " " + str(labels[i]) + "\n")
    with open(file_path, "a") as myfile:
     myfile.writelines(lines)


