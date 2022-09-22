#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:36:46 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, EE Dept., IIT Dharwad, Karnataka, India
"""

import argparse
import csv
import numpy as np
import os
from lib.metrics.performance_metrics import PerformanceMetrics


def generate_random_test_predictions(gt_fName, pred_fName):
    with open(pred_fName, 'a+', encoding='utf8') as fid_:
        writer_ = csv.writer(fid_)
        # The csv file has four columns
        writer_.writerow([
            'utterance_id',
            'speaker_id', 
            'score', 
            ])

    with open(gt_fName, 'r' ) as gt_fid_:
        reader_ = csv.DictReader(gt_fid_)
        for row_ in reader_:
            # Utterance_ID	c1	c2	c3	c4	c5	rand_ID
            Utterance_ID_ = row_['true_utterance_id']
            rand_ID_ = row_['utterance_id']
            cohort_ = []
            cohort_.append(row_['c1'])
            cohort_.append(row_['c2'])
            cohort_.append(row_['c3'])
            cohort_.append(row_['c4'])
            cohort_.append(row_['c5'])
            speaker_id_ = Utterance_ID_.split('_')[0]
            for i in range(5):
                if cohort_[i] == speaker_id_:
                    # score = np.random.uniform(low=0.7, high=1.0)
                    score = np.random.normal(loc=0.7, scale=0.1)
                else:
                    # score = np.random.uniform(low=0.0, high=0.3)
                    score = np.random.normal(loc=0.3, scale=0.1)
                with open(pred_fName, 'a+', encoding='utf8') as pred_fid_:
                    writer_ = csv.writer(pred_fid_)
                    writer_.writerow([
                        rand_ID_,
                        cohort_[i],
                        score, 
                        ])
    


def compute_performance(gt_fName, pred_fName):
    all_speaker_id_ = {}
    speaker_count_ = 0
    groundtruth_ = {}
    with open(gt_fName, 'r' ) as gt_fid_:
        reader_ = csv.DictReader(gt_fid_)
        for row_ in reader_:
            # Utterance_ID	c1	c2	c3	c4	c5	rand_ID
            groundtruth_[row_['utterance_id']] = row_
            speaker_id_ = row_['true_utterance_id'].split('_')[0]
            if not speaker_id_ in all_speaker_id_:
                all_speaker_id_[speaker_id_] = speaker_count_
                speaker_count_ += 1
    
    predictions_ = {}
    with open(pred_fName, 'r' ) as pred_fid_:
        reader_ = csv.DictReader(pred_fid_)
        for row_ in reader_:
            # rand_utterance_id speaker_id score', 
            if not row_['utterance_id'] in predictions_:
                predictions_[row_['utterance_id']] = {row_['speaker_id']:row_['score']}
            else:
                predictions_[row_['utterance_id']][row_['speaker_id']] = row_['score']
        
    gt_scores_ = np.zeros((len(groundtruth_), len(all_speaker_id_)))
    gt_scores_final_ = [] # np.empty([])
    pred_scores_ = np.ones((len(groundtruth_), len(all_speaker_id_)))*-9999999
    pred_scores_final_ = [] # np.empty([])
    utterance_count_ = 0
    not_tested_with_target_ = 0
    for rand_id_ in predictions_.keys():
        Utterance_ID_ = groundtruth_[rand_id_]['true_utterance_id']
        gt_speaker_id_ = Utterance_ID_.split('_')[0]
        gt_scores_[utterance_count_, all_speaker_id_[gt_speaker_id_]] = 1
        for test_speaker_id_ in predictions_[rand_id_].keys():
            pred_scores_[utterance_count_, all_speaker_id_[test_speaker_id_]] = float(predictions_[rand_id_][test_speaker_id_])
            if gt_speaker_id_==test_speaker_id_:
                gt_scores_final_.append(1)
            else:
                gt_scores_final_.append(0)
            pred_scores_final_.append(float(predictions_[rand_id_][test_speaker_id_]))

        if pred_scores_[utterance_count_, all_speaker_id_[gt_speaker_id_]]==0:
            not_tested_with_target_ += 1
        
        utterance_count_ += 1

    FPR_, TPR_, EER_, EER_thresh_ = PerformanceMetrics().compute_eer(gt_scores_final_, pred_scores_final_)
    
    ConfMat, precision, recall, fscore = PerformanceMetrics().compute_identification_performance(np.argmax(gt_scores_, axis=1), np.argmax(pred_scores_, axis=1), labels=list(range(len(all_speaker_id_))))
    
    print('\n\nResults:')
    print(f'\tPrecision={np.round(precision*100,2)}')
    print(f'\tRecall={np.round(recall*100,2)}')
    print(f'\tFscore={np.round(fscore*100,2)}')
    print(f'\tEER={np.round(EER_*100,2)}\n')
    
    opFile = '/'.join(pred_fName.split('/')[:-1]) + '/Evaluation_Results.txt'
    with open(opFile, 'w+', encoding='utf8') as fid_:
        fid_.write('Results:\n')
        fid_.write('----------------------------------------\n')
        fid_.write(f'\tPrecision={np.round(precision*100,2)}\n')
        fid_.write(f'\tRecall={np.round(recall*100,2)}\n')
        fid_.write(f'\tFscore={np.round(fscore*100,2)}\n')
        fid_.write(f'\tEER={np.round(EER_*100,2)}\n\n')
    
    return FPR_, TPR_, EER_, EER_thresh_




def __init__():
    '''
    This function initiates the evaluation script for 
    the I-MSV challenge.

    Returns
    -------
    args : namespace
        This namespace contains the environment variables as attributes to 
        run the experiments.

    '''
    # Setting up the argparse object
    parser = argparse.ArgumentParser(
        prog='evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluation script for I-MSV challenge 2022',
        epilog='',
        )
    # Adding a switch for the path to a csv file containing the meta information of the dataset
    # Adding switch to indicate the type of data
    parser.add_argument(
        '--groundtruth',
        type=str,
        help='Path to the groundtruth csv file',
        required=True,
        )
    parser.add_argument(
        '--prediction',
        type=str,
        help='Path to predictions csv file',
        required=True,
        )
    args = parser.parse_args()
    
    return args



if __name__ == '__main__':
    args = __init__()
    
    if not os.path.exists(args.prediction):
        generate_random_test_predictions(args.groundtruth, args.prediction)
        
    compute_performance(args.groundtruth, args.prediction)