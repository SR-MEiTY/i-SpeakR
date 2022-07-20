#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 13:52:57 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

class PerformanceMetrics:

    def __init__(self):
        return

    def compute_eer(self, groundtruth, scores):
        '''
        Compute the Equal Error Rate.

        Parameters
        ----------
        groundtruth : 1D array
            Array of groundtruths.
        scores : 1D array
            Array of predicted scores.

        Returns
        -------
        eer_ : float
            EER value.
        eer_threshold_ : float
            EER threshold.

        '''
        fpr_, tpr_, thresholds_ = roc_curve(y_true=groundtruth, y_score=scores, pos_label=1)
        fnr_ = 1 - tpr_
        # the threshold of fnr == fpr
        eer_threshold_ = thresholds_[np.nanargmin(np.absolute((fnr_ - fpr_)))]
        eer_1_ = fpr_[np.nanargmin(np.absolute((fnr_ - fpr_)))]
        eer_2_ = fnr_[np.nanargmin(np.absolute((fnr_ - fpr_)))]
        eer_ = (eer_1_+eer_2_)/2
        
        return fpr_, tpr_, eer_, eer_threshold_
    
    
    def compute_identification_performance(self, groundtruth, ptd_labels, labels):
        '''
        Compute the speaker identification performance.

        Parameters
        ----------
        groundtruth : 1D array
            Array of groundtruth labels.
        ptd_labels : 1D array
            Array of predicted speaker labels.
        labels : list
            List of all speaker labels.

        Returns
        -------
        ConfMat : 2D array
            Confusion Matrix.
        precision : 1D array
            Speaker-wise precisions.
        recall : 1D array
            Speaker-wise recall.
        fscore : 1D array
            Speaker-wise f1 scores.

        '''
        ConfMat = confusion_matrix(y_true=groundtruth, y_pred=ptd_labels)
        precision, recall, fscore, support = precision_recall_fscore_support(y_true=groundtruth, y_pred=ptd_labels, labels=labels, average='macro', zero_division=0)
        
        return ConfMat, precision, recall, fscore
    
    
    def plot_roc(self, fpr, tpr, opFile):
        '''
        Plot the Reciever Operating Characteristics (ROC) curve.

        Parameters
        ----------
        fpr : dict
            Dictionary variable containing duration wise False Positive Rate.
        tpr : dict
            Dictionary variable containing duration wise True Positive Rate.
        opFile : str
            Path to save the figure.

        Returns
        -------
        None.

        '''
        legend_text_ = []
        for dur_ in fpr.keys():
            plt.plot(fpr[dur_], tpr[dur_])
            legend_text_.append(str(dur_)+'s')
        plt.legend(legend_text_)
        plt.title('ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(opFile)
        
        return
        