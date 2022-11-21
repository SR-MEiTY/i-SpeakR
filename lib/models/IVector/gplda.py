#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:12:11 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
@code source: Jagabandhu Mishra, Ph.D. Scholar, Dept. of EE, IIT Dharwad (Matlab version)
"""

import numpy as np
import scipy
import time
import sys


def LDA_computation(ivector_per_speaker, ivectors_combined, ivec_dim, num_eigen_vectors):
    # % w is a cell consisting of class number of matrixes, and each matrix consists
    # % of ivectors of [dim x no. of vectors ]
    # ivec_dim: ivector dimension
    # num_eigen_vectors: LDA dimensions
    
    # projectionMatrix_ = np.eye(ivec_dim)
    Sw_ = np.zeros((ivec_dim,ivec_dim))
    Sb_ = np.zeros((ivec_dim,ivec_dim))
    wbar_ = np.mean(ivectors_combined, axis=1)
    for speaker_id_ in ivector_per_speaker.keys():
        ws_ = ivector_per_speaker[speaker_id_]
        wsbar_ = np.mean(ws_, axis=1)
        Sb_ = np.add(Sb_, np.subtract(wsbar_, wbar_) @ np.subtract(wsbar_, wbar_).T)
        Sw_ = np.add(Sw_, np.cov(ws_, bias=True))
    
    eig_vals_, A_ = scipy.linalg.eig(Sb_, Sw_)
    largest_eig_val_idx_ = np.argsort(eig_vals_)[-num_eigen_vectors-1:-1]
    A_ = A_[:, largest_eig_val_idx_]
    A_ = np.divide(A_, np.repeat(np.array(np.linalg.norm(A_), ndmin=2), np.shape(A_)[0], axis=0)).T
    print(f'A_ = {np.shape(A_)}')
    
    return A_



def WCCN_computation(ivector_per_speaker, ivec_dim):
    # % w is a cell consisting of class number of matrixes, and each matrix consists
    # % of ivectors of [dim x no. of vectors ]  
    
    # projectionMatrix_ = np.eye(ivec_dim)
    alpha_ = 0.9
    
    W_ = np.zeros((ivec_dim, ivec_dim))
    for speaker_id_ in ivector_per_speaker.keys():
        ws_ = ivector_per_speaker[speaker_id_]
        W_ += np.cov(ws_, bias=True)
    
    W_ /= len(ivector_per_speaker)
    W_ = (1 - alpha_)*W_ + alpha_*np.eye(ivec_dim)
    B_ = np.linalg.cholesky(np.linalg.pinv(W_))

    print(f'B_={np.shape(B_)}')

    return B_




def GPLDA_computation(ivector_per_speaker, lda_dim=20, perform_LDA=False, perform_WCCN=False, num_iter=50):
    # % ivectorPerSpeaker is a dict consisting speaker wise i-vectors, 
    # % and each matrix consists of ivectors of [dim x no. of vectors ]
    # % if performLDA is false, pass lda_dim as some dommy value, it will never
    # % be used, else provide appropriate dimension
    
    w_ = ivector_per_speaker
    ivec_dim_ = np.shape(w_[list(w_.keys())[0]])[0]
    
    utterance_per_speaker_ = {}
    ivectors_train_ = np.empty([], dtype=np.float32)
    for speaker_id_ in w_.keys():
        utterance_per_speaker_[speaker_id_] = {}
        if np.size(ivectors_train_)<=1:
            utterance_per_speaker_[speaker_id_]['start'] = 0
            ivectors_train_ = w_[speaker_id_]
            utterance_per_speaker_[speaker_id_]['end'] = np.shape(ivectors_train_)[1]
        else:
            utterance_per_speaker_[speaker_id_]['start'] = np.shape(ivectors_train_)[1]
            ivectors_train_ = np.append(ivectors_train_, w_[speaker_id_], axis=1)
            utterance_per_speaker_[speaker_id_]['end'] = np.shape(ivectors_train_)[1]
    
    
    '''~~~~~~~~~~~~~~~~~~ LDA Computation ~~~~~~~~~~~~~~~~~ '''
    projectionMatrix_ = np.eye(ivec_dim_)
    if perform_LDA:
        startTime = time.process_time()
        A_ = LDA_computation(w_, ivectors_train_, ivec_dim_, lda_dim)
        print(f'A_={np.shape(A_)}')
        print(f'ivectors_train_={np.shape(ivectors_train_)}')
        ivectors_train_ = A_ @ ivectors_train_
        print(f'ivectors_train_={np.shape(ivectors_train_)}')
        for speaker_id_ in w_.keys():
            w_[speaker_id_] = ivectors_train_[:, utterance_per_speaker_[speaker_id_]['start']:utterance_per_speaker_[speaker_id_]['end']]
        projectionMatrix_ = A_ @ projectionMatrix_
        print(f"LDA projection matrix calculated {np.round(time.process_time()-startTime,2)} seconds). projectionMatrix_={np.shape(projectionMatrix_)}")
    
    
    '''~~~~~~~~~~~~~~~~~~ WCCN computation ~~~~~~~~~~~~~~~~~ '''
    if perform_WCCN:
        startTime = time.process_time()
        B_ = WCCN_computation(w_, np.min([ivec_dim_, lda_dim]))
        projectionMatrix_ = (B_ @ projectionMatrix_)
        print(f"WCCN projection matrix calculated {np.round(time.process_time()-startTime,2)} seconds). projectionMatrix_={np.shape(projectionMatrix_)}")
    
    
    ivectors_ = {}
    for speaker_id_ in w_.keys():
        ivectors_[speaker_id_] = projectionMatrix_.T @ w_[speaker_id_]
    num_eigen_voices_ = len(ivectors_) # check what is eigen voice may be 2


    K_ = len(ivectors_) # number of speakers
    D_ = ivec_dim_ # ivector dimension
    ivectors_all_ = np.empty([], dtype=np.float32)
    utterance_per_speaker_ = {}
    for speaker_id_ in ivectors_.keys():
        utterance_per_speaker_[speaker_id_] = {}
        if np.size(ivectors_all_)<=1:
            utterance_per_speaker_[speaker_id_]['start'] = 0
            ivectors_all_ = ivectors_[speaker_id_]
            utterance_per_speaker_[speaker_id_]['end'] = np.shape(ivectors_all_)[1]
        else:
            utterance_per_speaker_[speaker_id_]['start'] = np.shape(ivectors_all_)[1]
            ivectors_all_ = np.append(ivectors_all_, ivectors_[speaker_id_], axis=1)
            utterance_per_speaker_[speaker_id_]['end'] = np.shape(ivectors_all_)[1]
    N_ = np.shape(ivectors_all_)[1] # Number of ivectors
    mu_ = np.mean(ivectors_all_, axis=1) # Mean ivectors
    mu_repeat_ = np.repeat(np.array(mu_, ndmin=2).T, np.shape(ivectors_all_)[1], axis=1)
    ivectors_all_ = np.subtract(ivectors_all_, mu_repeat_)

    
    '''
    Whitening ZCA (Ensures the covariance matrix to Identity)
    '''
    whiteningType = 'ZCA'
    eps_ = sys.float_info.epsilon
    
    if whiteningType=='ZCA':
        S_ = np.cov(ivectors_all_)
        _, sD_, sV_ = np.linalg.svd(S_)
        W_ = np.repeat(np.array((np.sqrt(sD_) + eps_)**-1, ndmin=2).T, ivectors_all_.shape[0], axis=1) @ sV_.T
        ivectors_all_ = W_ @ ivectors_all_
        
    elif whiteningType=='PCA':
        S_ = np.cov(ivectors_all_)
        sD_, sV_ = np.linalg.eig(S_)
        sD__ = np.repeat(np.array((np.sqrt(sD_) + eps_)**-1, ndmin=2).T, ivectors_all_.shape[0], axis=1)
        W_ = sD__ @ sV_.T
        ivectors_all_ = W_ @ ivectors_all_

    else:
        W_ = np.eye(ivec_dim_)
        
    print(f'Whitening: ivectors_all_={np.shape(ivectors_all_)} W_={np.shape(W_)}')


    ivectors_all_ = np.divide(ivectors_all_, np.repeat(np.array(np.linalg.norm(ivectors_all_), ndmin=2), np.shape(ivectors_all_)[0], axis=0))
    S_ = ivectors_all_ @ ivectors_all_.T
    ivectors_ = {}
    for speaker_id_ in w_.keys():
        i_ = utterance_per_speaker_[speaker_id_]['start']
        j_ = utterance_per_speaker_[speaker_id_]['end']
        ivectors_[speaker_id_] = ivectors_all_[:, i_:j_]

    utter_lengths_ = {key:(utterance_per_speaker_[key]['end']-utterance_per_speaker_[key]['start']) for key in utterance_per_speaker_.keys()}
    uniq_lengths_ = np.sort(np.unique([val for val in utter_lengths_.values()]))

    speaker_idx_ = 0
    f_ = np.zeros((D_,K_))
    ivectors_sorted_ = {}
    for uniq_len_ in uniq_lengths_:
        idx_ = []
        for speaker_id_ in utter_lengths_.keys():
            if utter_lengths_[speaker_id_]==uniq_len_:
                idx_.append(speaker_id_)
        temp_ = {}
        count_ = 0
        for speaker_idx_within_unique_length_  in idx_:
            rho_ = ivectors_[speaker_idx_within_unique_length_]
            temp_[count_] = rho_
            f_[:,speaker_idx_] = np.sum(rho_, axis=1)
            speaker_idx_ += 1
        ivectors_sorted_[uniq_len_] = temp_
    
    print(f'Length wise sorting speakers: uniq lens={ivectors_sorted_.keys()}')
    
    
    '''
    GPLDA Training
    '''
    V_ = np.random.normal(loc=0.0, scale=1.0, size=(D_, num_eigen_voices_))
    Lambda_ = np.linalg.pinv(np.divide(S_, N_))     
    min_divergence_ = True
    
    for iter_i_ in range(num_iter):
        # print(f'G-PLDA: iter:{iter_i_}')
        # EXPECTATION
        gamma_ = np.zeros((num_eigen_voices_, num_eigen_voices_))
        EyTotal_ = np.zeros((num_eigen_voices_, K_))
        R_ = np.zeros((num_eigen_voices_, num_eigen_voices_))
        
        idx_ = 0
        for uniq_len_ in uniq_lengths_:
            ivector_len_ = uniq_len_
            
            # Isolate i-vectors of the same given length
            iv_ = ivectors_sorted_[uniq_len_]
            
            # Calculate M
            M_ = np.linalg.pinv(ivector_len_*(V_.T @ (Lambda_ @ V_)) + np.eye(num_eigen_voices_)) # Equation (A.7) in [13]
            
            # Loop over each speaker for the current i-vector length
            for ivec_len_ in iv_.keys():
                # First moment of latent variable for V
                Ey_ = M_ @ V_.T @ Lambda_ @ f_[:, idx_] # Equation (A.8) in [13]
                
                # Calculate second moment.
                Eyy_ = Ey_ @ Ey_.T
                
                # Update Ryy 
                R_ += ivector_len_*(M_ + Eyy_) # Equation (A.13) in [13]
                
                # Append EyTotal
                EyTotal_[:, idx_] = Ey_

                idx_ += 1
                
                # If using minimum divergence, update gamma.
                if min_divergence_:
                    gamma_ += (M_ + Eyy_) # Equation (A.18) in [13]

        # Calculate T
        TT_ = EyTotal_ @ f_.T # Equation (A.12) in [13]
        
        # MAXIMIZATION
        V_ = TT_.T @ np.linalg.pinv(R_) # Equation (A.16) in [13]
        Lambda_ = np.linalg.pinv(np.divide((S_ - V_ @ TT_), N_)) # Equation (A.17) in [13]
    
        # MINIMUM DIVERGENCE
        if min_divergence_:
            gamma_ = np.divide(gamma_, K_) # Equation (A.18) in [13]
            V_ = V_ @ np.linalg.cholesky(gamma_ @ gamma_.T) # Equation (A.22) in [13]. Originally cholesky decomposition of gamma_ is performed. But, this operation requires a positive definite matrix and gamma_ is not positive definite always. Hence, (gamma_ @ gamma_.T) is performed  

    gpldaModel_ = {
        'mu': mu_,
        'WhiteningMatrix': W_,
        'EigenVoices': V_,
        'Sigma': np.linalg.pinv(Lambda_),
        }
    
    return gpldaModel_, projectionMatrix_




def gpldaScore(gpldaModel, w1, wt):
    # % PLDA scoring defined in:
    # % D.Garcia-Romero and C. Epsy-Wilson, "Analysis of I-vector Length
    # % Normalization in Speaker Recognition Systems." Interspeech, 2011, pp.
    # % 249-252.
    # %
    # % Rajan, Padmanabhan, Anton Afanasyev, Ville Hautamaki, and Tomi Kinnunen.
    # % "From Single to Multiple Enrollment i-Vectors: Practical PLDA Scoring
    # % Variants for Speaker Verification." Digital Signal Processing 31 (2014):
    # % 93-101.
    
    # % IO
    
    # % gpldaModel: is a structure cosists of mu, whitening matrix,Eigen voices
    # % and Sigma
    
    # % output: log like lihood score (Eq-4), ratio of both vector from same to
    # % different
    
    # %
    
    # Center the data
    w1 = w1 - gpldaModel['mu']
    wt = wt - gpldaModel['mu']
    
    # Whiten the data
    w1 = gpldaModel['WhiteningMatrix'] @ w1
    wt = gpldaModel['WhiteningMatrix'] @ wt
    
    # Length-normalize the data
    w1 = np.divide(w1, np.repeat(np.array(np.linalg.norm(w1), ndmin=2), np.shape(w1)[0], axis=0)).T
    wt = np.divide(wt, np.repeat(np.array(np.linalg.norm(wt), ndmin=2), np.shape(wt)[0], axis=0)).T
    
    # Score the similarity of the i-vectors based on the log-likelihood.
    VVt_ = gpldaModel['EigenVoices'] @ gpldaModel['EigenVoices'].T
    SVVt_ = gpldaModel['Sigma'] + VVt_
    
    term1_ = np.linalg.pinv(np.append(np.append(SVVt_, VVt_, axis=1), np.append(VVt_, SVVt_, axis=1), axis=0))
    term2_ = np.linalg.pinv(SVVt_)
    
    w1wt_ = np.append(w1, wt, axis=0)
    score_ = -w1wt_.T @ term1_ @ w1wt_ + w1.T @ term2_ @ w1 + wt.T @ term2_ @ wt # Modified by jagabandhu 16/04 ref. paper 2 the Eq 5 and 6, in eq 5 to 6, there is a sign mismatch, has been corrected here   
    
    return score_