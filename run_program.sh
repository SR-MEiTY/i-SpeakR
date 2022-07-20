#!/bin/bash

# Run the main programs. For help, run the following command:
#       python main.py -h
#python main.py --data_info specify --data_path /home/mrinmoy/Projects/SR_MEiTY/IITG-MV_database/IITG_MV_Phase-I_Speaker_Recognition_Database/100_speaker_Database_Office_Environment/ --output_path ../


python data_preparation.py
python feature_computation.py
python train_test_model.py --model_type GMM-UBM
