#!/bin/bash

# Run the main programs. For help, run the following command:
#       python main.py -h
#python main.py --data_info specify --data_path /home/mrinmoy/Projects/SR_MEiTY/IITG-MV_database/IITG_MV_Phase-I_Speaker_Recognition_Database/100_speaker_Database_Office_Environment/ --output_path ../

#python data_preparation.py --dev_path <> --dev_key <> --enr_path <> --enr_key <> --test_path <> --test_key <>
#python feature_computation.py
###python train_test_model.py --model_type GMM-UBM
#python train_test_model.py --model_type I-Vector


python data_preparation.py --dev_path /home/mrinmoy/Documents/Speaker_Recognition_MeitY_Project/Project_Related/Toolkit/Test_Dataset/DEV/ --dev_key /home/mrinmoy/Documents/Speaker_Recognition_MeitY_Project/Project_Related/Toolkit/Test_Dataset/DEV.csv --enr_path /home/mrinmoy/Documents/Speaker_Recognition_MeitY_Project/Project_Related/Toolkit/Test_Dataset/ENR/ --enr_key /home/mrinmoy/Documents/Speaker_Recognition_MeitY_Project/Project_Related/Toolkit/Test_Dataset/ENR.csv --test_path /home/mrinmoy/Documents/Speaker_Recognition_MeitY_Project/Project_Related/Toolkit/Test_Dataset/TEST/ --test_key /home/mrinmoy/Documents/Speaker_Recognition_MeitY_Project/Project_Related/Toolkit/Test_Dataset/TEST.csv
python feature_computation.py
python train_test_model.py

