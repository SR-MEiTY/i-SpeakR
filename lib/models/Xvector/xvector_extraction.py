#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu March 23 2023

@contributers: 
    Dr. Gayadhar Pradhan, Fatima Zaheera, National Institute of Technology Patna
    Mrinmoy Bhattacharjee, Indian Institute of Technology Dharwad
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from XvecSpeechGenerator import XvecSpeechGenerator
import torch.nn as nn
import os
from torch import optim
from x_vector_Indian_LID import X_vector
from sklearn.metrics import accuracy_score
from utils import speech_collate
from utils import writeXvectors
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import sys



class XvectorComputation:
    trainloss = []
    train_acc = []
    dataloader_dev = None
    dataloader_train = None
    dataloader_test = None
    model = None
    test_model = None
    optimizer = None
    loss_fun = None

    def __init__(self, input_dim, num_classes, win_length, n_fft, batch_size, use_gpu, num_epochs, feat_path, speaker_xvec_path):    
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.win_length = win_length
        self.n_fft = n_fft
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.num_epochs = num_epochs
        self.feat_path = feat_path
        self.speaker_xvec_path = speaker_xvec_path

        all_speakers_dev = {'count': 0}
        self.dev_filepath = self.feat_path + '/Dev_data.txt'
        with open(self.dev_filepath, 'w+') as fid:
            fid.write('')
        # files = next(os.walk(self.feat_path+'/DEV/'))[2]
        files = next(os.walk(self.feat_path+'/Dev_data/'))[2]
        for fl in files:
            speaker_id = fl.split('.')[0].split('_')[0]
            if not speaker_id in all_speakers_dev.keys():
                all_speakers_dev[speaker_id] = str(all_speakers_dev['count'])
                all_speakers_dev['count'] += 1
            with open(self.dev_filepath, 'a+') as fid:
                # fid.write(self.feat_path+'/DEV/'+fl+' '+all_speakers[speaker_id]+'\n')
                fid.write(self.feat_path+'/Dev_data/'+fl+' '+all_speakers_dev[speaker_id]+'\n')
        
        all_speakers_training = {'count': 0}
        self.training_filepath = self.feat_path + '/Enr_data.txt'
        with open(self.training_filepath, 'w+') as fid:
            fid.write('')
        # files = next(os.walk(self.feat_path+'/ENR/'))[2]
        files = next(os.walk(self.feat_path+'/Enr_data/'))[2]
        for fl in files:
            speaker_id = fl.split('.')[0].split('_')[0]
            if not speaker_id in all_speakers_training.keys():
                all_speakers_training[speaker_id] = str(all_speakers_training['count'])
                all_speakers_training['count'] += 1
            with open(self.training_filepath, 'a+') as fid:
                # fid.write(self.feat_path+'/ENR/'+fl+' '+all_speakers[speaker_id]+'\n')
                fid.write(self.feat_path+'/Enr_data/'+fl+' '+all_speakers_training[speaker_id]+'\n')
                
        self.testing_filepath = self.feat_path + '/public_test_cohart_edited.txt'
        with open(self.testing_filepath, 'w+') as fid:
            fid.write('')
        # files = next(os.walk(self.feat_path+'/TEST/'))[2]
        files = next(os.walk(self.feat_path+'/public_test_cohart_edited/'))[2]
        # files = next(os.walk(self.feat_path+'/private_test_cohort_final/'))[2]
        for fl in files:
            speaker_id = fl.split('.')[0].split('_')[0]
            if not speaker_id in all_speakers_training.keys():
                all_speakers_training[speaker_id] = str(all_speakers_training['count'])
                all_speakers_training['count'] += 1
            with open(self.testing_filepath, 'a+') as fid:
                # fid.write(self.feat_path+'/TEST/'+fl+' '+all_speakers[speaker_id]+'\n')
                fid.write(self.feat_path+'/public_test_cohart_edited/'+fl+' '+all_speakers_training[speaker_id]+'\n')
        
        ### Data related
        self.dataset_dev = XvecSpeechGenerator(manifest=self.dev_filepath, mode='dev', win_length=self.win_length, n_fft=self.n_fft)
        self.dataloader_dev = DataLoader(self.dataset_dev, batch_size=self.batch_size, shuffle=True, collate_fn=speech_collate)
        
        self.dataset_train = XvecSpeechGenerator(manifest=self.training_filepath, mode='train', win_length=self.win_length, n_fft=self.n_fft)
        self.dataloader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, collate_fn=speech_collate)
        
        self.dataset_test = XvecSpeechGenerator(manifest=self.testing_filepath, mode='test', win_length=self.win_length, n_fft=self.n_fft)
        self.dataloader_test = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=True, collate_fn=speech_collate)
        
        ## Model related
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        print(self.input_dim)
        print(self.num_classes)
        self.test_model = X_vector(self.input_dim, self.num_classes)
        print(self.test_model)
        
        self.model = X_vector(self.input_dim, self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
        self.loss_fun = nn.CrossEntropyLoss()
    
    
    def compute_xvectors(self):
        for epoch_i in range(self.num_epochs):
            loss, acc = self.train(self.dataloader_dev, epoch_i)
            self.trainloss.append(loss)
            self.train_acc.append(acc)
            self.validation(self.dataloader_train, epoch_i, 'train')
            self.validation(self.dataloader_test, epoch_i, 'test')
            # plt.plot(trainloss)
        # plt.show()
        
        self.validation(self.dataloader_dev, self.num_epochs, 'dev')
        self.validation(self.dataloader_train, self.num_epochs, 'train')
        self.validation(self.dataloader_test, self.num_epochs, 'test')
        
    
        
    def train(self, vdataloader_train, epoch):
        train_loss_list=[]
        full_preds=[]
        full_gts=[]
        self.model.train()
        count = 1
        x_vec = ""
        # path = "xvector"
        df = pd.DataFrame(columns=['x_vec_path', 'label'])
    
        if os.path.exists(self.speaker_xvec_path) == False:
            os.mkdir(self.speaker_xvec_path)
    
        xvec_feat_path = os.path.join(self.speaker_xvec_path, 'Dev_data')
        if os.path.exists(xvec_feat_path):
            shutil.rmtree(xvec_feat_path)
        os.mkdir(xvec_feat_path)
    
        for i_batch, sample_batched in enumerate(self.dataloader_train):
            print('Processing batch', 'Dev ', count)
            feat = np.empty([])
            for torch_tensor in sample_batched[0]:
                # print(np.shape(torch_tensor))
                fv = torch_tensor.numpy()
                if np.size(feat)<=1:
                    feat = np.expand_dims(fv.T, axis=0)
                else:
                    if np.shape(feat)[1]>np.shape(fv)[1]:
                        feat = feat[:,:np.shape(fv)[1],:]
                    elif np.shape(fv)[1]>np.shape(feat)[1]:
                        fv = fv[:,:np.shape(feat)[1]]
                    feat = np.append(feat, np.expand_dims(fv.T, axis=0), axis=0)
            # feat = np.asarray(feat)
            # print(f'feat = {np.shape(feat)}')
            
            #count = count + 1
            # features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]])).float()
            features = torch.from_numpy(feat)
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            paths = np.asarray([torch_tensor for torch_tensor in sample_batched[2]])
    
            # print(f'features = {np.shape(features)}')
            # print(f'labels = {np.shape(labels)}')
            
            #print("Processing files ", paths)
            features, labels = features.to(self.device),labels.to(self.device)
            numpy_features = features.detach().cpu().numpy()
            numpy_labels = labels.detach().cpu().numpy()
            features.requires_grad = True
            self.optimizer.zero_grad()
            pred_logits,x_vec = self.model(features)
            #################################################
            ############ Writing X Vectors ##################
            if epoch == self.num_epochs - 1:
                print("Started writing xvecs at epoch count", epoch)
                df = writeXvectors(xvec_feat_path , df, numpy_labels, x_vec, paths)
            #################################################
            #### CE loss
            loss = self.loss_fun(pred_logits, labels.long())
            loss.backward()
            self.optimizer.step()
            train_loss_list.append(loss.item())
            count = count + 1
            #train_acc_list.append(accuracy)
            #if i_batch%10==0:
            #    print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))
    
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
    
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(train_loss_list))
        #print("Training xvector ", x_vec.detach().numpy())
        path = os.path.join(self.speaker_xvec_path, 'Dev_data_xvector.npy')
        np.save(path, x_vec.cpu().detach().numpy())
        df.to_csv(os.path.join(xvec_feat_path, 'Dev_data_xvector.csv'), encoding='utf-8')
        print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))
        rand = np.random.rand()
        model_save_path = os.path.join(self.speaker_xvec_path, 'save_model')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        path = os.path.join(model_save_path, 'train_best_check_point_' + str(epoch) + '_' + str(mean_loss) + '_' + str(rand))
        state_dict = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
        torch.save(self.model.state_dict(), path)
        return (mean_loss, mean_acc)
    
    
    
    def validation(self, dataloader_val, epoch, mode):
        #model1 = torch.load("C:\\Users\\Talib\\Downloads\\talib\\talib\\X-vector_codes\\save_model\\best_check_point_1_1.395266056060791_0.761649240146505")
        model = X_vector(self.input_dim, self.num_classes).to(self.device)
        #model.state_dict(torch.load("/home/fathima/Desktop/x_py/x_vector-25-10/save_model/train_best_check_point_2_0.6299841274817785_0.6732803042366898"))
        print("Processing mode {}".format( mode))
        #model.eval()
    
        # path = "xvector"
        count = 0
        xvec_feat_path = ""
        df = pd.DataFrame(columns=['x_vec_path', 'label'])
        if mode == 'dev':
            xvec_feat_path = os.path.join(self.speaker_xvec_path, 'Dev_data')
        elif mode == 'train':
            xvec_feat_path = os.path.join(self.speaker_xvec_path, 'Enr_data')
        else:
            xvec_feat_path = os.path.join(self.speaker_xvec_path, 'public_test_cohart_edited')
        if os.path.exists(xvec_feat_path):
            shutil.rmtree(xvec_feat_path)
        os.mkdir(xvec_feat_path)
        
        with torch.no_grad():
            val_loss_list=[]
            full_preds=[]
            full_gts=[]
            for i_batch, sample_batched in enumerate(dataloader_val):
                feat = np.empty([])
                for torch_tensor in sample_batched[0]:
                    # print(np.shape(torch_tensor))
                    fv = torch_tensor.numpy()
                    if np.size(feat)<=1:
                        feat = np.expand_dims(fv.T, axis=0)
                    else:
                        if np.shape(feat)[1]>np.shape(fv)[1]:
                            feat = feat[:,:np.shape(fv)[1],:]
                        elif np.shape(fv)[1]>np.shape(feat)[1]:
                            fv = fv[:,:np.shape(feat)[1]]
                        feat = np.append(feat, np.expand_dims(fv.T, axis=0), axis=0)
                # feat = np.asarray(feat)
                # print(f'feat = {np.shape(feat)}')
                features = torch.from_numpy(feat)
    
                # features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
                
                labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
                paths = np.asarray([torch_tensor for torch_tensor in sample_batched[2]])
                print("Processing  batch count {}".format(count))
                numpy_features = features.detach().cpu().numpy()
                numpy_labels = labels.detach().cpu().numpy()
                features, labels = features.to(self.device),labels.to(self.device)
                pred_logits,x_vec = model(features)
                #################################################
                ############ Writing X Vectors ##################
                df = writeXvectors(xvec_feat_path, df, numpy_labels, x_vec, paths)
                #################################################
                #### CE loss
                loss = self.loss_fun(pred_logits,labels.long())
                val_loss_list.append(loss.item())
                #train_acc_list.append(accuracy)
                count = count + 1
                predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
                for pred in predictions:
                    full_preds.append(pred)
                for lab in labels.detach().cpu().numpy():
                    full_gts.append(lab)
    
    
            mean_acc = accuracy_score(full_gts,full_preds)
            mean_loss = np.mean(np.asarray(val_loss_list))
            #print("Training xvector ", x_vec.detach().numpy())
            if mode == 'dev':
                path = os.path.join(self.speaker_xvec_path, 'Dev_data_xvector.npy')
            elif mode == 'test':
                path = os.path.join(self.speaker_xvec_path, 'public_test_cohart_edited_xvector.npy')
            else:
                path = os.path.join(self.speaker_xvec_path, 'Enr_data_xvector.npy')
            np.save(path, x_vec.cpu().detach().numpy())
            df.to_csv(os.path.join(xvec_feat_path, 'train_xvect_feat.csv'), encoding='utf-8')
            print('Total validation loss {} and Validation accuracy {} after'.format(mean_loss,mean_acc))

    
    
if __name__ == '__main__':
    # print(args)

    input_dim = 39
    num_classes = 50
    win_length = 78
    n_fft = 78
    batch_size = 10
    use_gpu = True
    num_epochs = 20
    
    # feat_path = '/home/mrinmoy/Documents/Professional/Senior_Project_Engineer_IIT_Dharwad_IndicASV/Toolkit/i-SpeakR_output/Test_Dataset/features/MFCC/'
    # speaker_xvec_path = '/home/mrinmoy/Documents/Professional/Senior_Project_Engineer_IIT_Dharwad_IndicASV/Toolkit/i-SpeakR_output/Test_Dataset/TEST_Xvector/xvectors/'

    feat_path = '/DATA/jagabandhu/i-SpeakR_output/I-MSV/features/MFCC/'
    speaker_xvec_path = '/DATA/jagabandhu/i-SpeakR_output/I-MSV/models/MFCC_x_vector/'
    
    xvec = XvectorComputation(input_dim, num_classes, win_length, n_fft, batch_size, use_gpu, num_epochs, feat_path, speaker_xvec_path)
    xvec.compute_xvectors()
   

    
