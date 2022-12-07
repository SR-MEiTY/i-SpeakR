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
    
    # projection_matrix_ = np.eye(ivec_dim)
    Sw_ = np.zeros((ivec_dim,ivec_dim))
    Sb_ = np.zeros((ivec_dim,ivec_dim))
    wbar_ = np.mean(ivectors_combined, axis=1)
    for speaker_id_ in ivector_per_speaker.keys():
        ws_ = ivector_per_speaker[speaker_id_]
        wsbar_ = np.mean(ws_, axis=1)
        Sb_ = np.add(Sb_, np.subtract(wsbar_, wbar_) @ np.subtract(wsbar_, wbar_).T)
        Sw_ = np.add(Sw_, np.cov(ws_, bias=True))
    
    eig_vals_, A_ = scipy.linalg.eig(Sb_, Sw_)
    # largest_eig_val_idx_ = np.argsort(eig_vals_)[-num_eigen_vectors-1:-1]
    largest_eig_val_idx_ = np.argsort(eig_vals_)[-num_eigen_vectors-1:-1] # Updated on 7 Dec 22. Previously largets eigenvector was not being selected
    A_ = A_[:, largest_eig_val_idx_]
    A_ = np.divide(A_, np.repeat(np.array(np.linalg.norm(A_, axis=0), ndmin=2), np.shape(A_)[0], axis=0)).T
    print(f'A_ = {np.shape(A_)}')
    
    return A_



def WCCN_computation(ivector_per_speaker, projection_dim):
    # % w is a cell consisting of class number of matrixes, and each matrix consists
    # % of ivectors of [dim x no. of vectors ]  
    
    # projection_matrix_ = np.eye(ivec_dim)
    alpha_ = 0.9
    
    W_ = np.zeros((projection_dim, projection_dim))
    for speaker_id_ in ivector_per_speaker.keys():
        ws_ = ivector_per_speaker[speaker_id_]
        W_ = np.add(W_, np.cov(ws_, bias=True))
    
    W_ = np.divide(W_, len(ivector_per_speaker))
    W_ = (1 - alpha_)*W_ + alpha_*np.eye(projection_dim)
    B_ = np.linalg.cholesky(np.linalg.pinv(W_))

    print(f'B_={np.shape(B_)}')

    return B_




def GPLDA_computation(ivector_per_speaker, num_eigen_vectors=20, perform_LDA=False, perform_WCCN=False, num_iter=50):
    # % ivectorPerSpeaker is a dict consisting speaker wise i-vectors, 
    # % and each matrix consists of ivectors of [dim x no. of vectors ]
    # % if performLDA is false, pass lda_dim as some dommy value, it will never
    # % be used, else provide appropriate dimension
    
    w_ = ivector_per_speaker.copy()
    ivec_dim_ = np.shape(w_[list(w_.keys())[0]])[0]
    print(f'ivec_dim_={ivec_dim_}')
    
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
    projection_matrix_ = np.eye(ivec_dim_)
    print(f'projection_matrix_={np.shape(projection_matrix_)}')
    if perform_LDA:
        startTime = time.process_time()
        A_ = LDA_computation(w_, ivectors_train_, ivec_dim_, num_eigen_vectors)
        ivectors_train_ = A_ @ ivectors_train_
        for speaker_id_ in w_.keys():
            w_[speaker_id_] = ivectors_train_[:, utterance_per_speaker_[speaker_id_]['start']:utterance_per_speaker_[speaker_id_]['end']].copy()
        projection_matrix_ = A_ @ projection_matrix_
        print(f"LDA projection matrix calculated {np.round(time.process_time()-startTime,2)} seconds). projection_matrix_={np.shape(projection_matrix_)}")

    print(f'A_={np.shape(A_)}')
    print(f'projection_matrix_={np.shape(projection_matrix_)}')
    
    
    '''~~~~~~~~~~~~~~~~~~ WCCN computation ~~~~~~~~~~~~~~~~~ '''
    if perform_WCCN:
        startTime = time.process_time()
        projection_dim_ = projection_matrix_.shape[0]
        B_ = WCCN_computation(w_, projection_dim_)
        projection_matrix_ = B_ @ projection_matrix_
        print(f"WCCN projection matrix calculated {np.round(time.process_time()-startTime,2)} seconds). projection_matrix_={np.shape(projection_matrix_)}")

    print(f'B_={np.shape(B_)}')
    print(f'projection_matrix_={np.shape(projection_matrix_)}')
    
    
    # Applying the projection matrix to the train set
    ivectors_ = {}
    for speaker_id_ in w_.keys():
        ivectors_[speaker_id_] = projection_matrix_ @ ivector_per_speaker[speaker_id_]
        print(f'Projected i-vectors: {speaker_id_} {ivectors_[speaker_id_].shape}')


    num_eigen_voices_ = num_eigen_vectors # len(ivectors_) # check what is eigen voice may be 2
    K_ = len(ivectors_) # number of speakers
    D_ = ivectors_[next(iter(ivectors_))].shape[0] # ivec_dim_ # ivector dimension

    print(f'num_eigen_voices_={num_eigen_voices_}')
    print(f'number of speakers K_={K_}')
    print(f'ivector dimension D_={D_}')
    
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

    print(f'Total ivector matrix={np.shape(ivectors_all_)}')
    print(f'Total Number of ivectors N_={N_}')
    print(f'Total ivectors Mean mu_={np.shape(mu_)} mu_repeat_={np.shape(mu_repeat_)}')

    
    '''
    Whitening ZCA (Ensures the covariance matrix to Identity)
    '''
    whiteningType = 'ZCA'
    eps_ = sys.float_info.epsilon
    
    if whiteningType=='ZCA':
        S_ = np.cov(ivectors_all_)
        _, sD_, sV_ = np.linalg.svd(S_)
        print(f'sD_={np.shape(sD_)}')
        print(f'sV_={np.shape(sV_)}')
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
    
    ivectors_norm_ = np.repeat(np.array(np.linalg.norm(ivectors_all_, axis=0), ndmin=2), np.shape(ivectors_all_)[0], axis=0)+1e-10
    print(f'ivectors_norm_={np.shape(ivectors_norm_)}')
    ivectors_all_ = np.divide(ivectors_all_, ivectors_norm_)
    print(f'ivectors_all_={np.shape(ivectors_all_)}')


    S_ = ivectors_all_ @ ivectors_all_.T # global second-order moment
    print(f'S_={np.shape(S_)}')
    
    ivectors_ = {}
    for speaker_id_ in w_.keys():
        i_ = utterance_per_speaker_[speaker_id_]['start']
        j_ = utterance_per_speaker_[speaker_id_]['end']
        ivectors_[speaker_id_] = ivectors_all_[:, i_:j_].copy()

    utter_lengths_ = {key:(utterance_per_speaker_[key]['end']-utterance_per_speaker_[key]['start']) for key in utterance_per_speaker_.keys()}
    uniq_lengths_ = np.sort(np.unique([val for val in utter_lengths_.values()]))
    
    print(f'utter_lengths_: {utter_lengths_}')
    print(f'uniq_lengths_: {uniq_lengths_}')


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
            # print(f'rho_={np.shape(rho_)}')
            # print(f'rho_ sum={np.shape(np.sum(rho_, axis=1))}')
            f_[:,speaker_idx_] = np.sum(rho_, axis=1)
            speaker_idx_ += 1
            count_ += 1
        ivectors_sorted_[uniq_len_] = temp_.copy()

    print(f'f_={np.shape(f_)} determinant={np.linalg.det(f_@f_.T)}')    
    print(f'Length wise sorting speakers {speaker_idx_} sorted={len(ivectors_sorted_)} actual={len(ivectors_)}')
    count = 0
    for key in ivectors_sorted_.keys():
        count += len(ivectors_sorted_[key])
    print(f'count={count}')

    
    
    '''
    GPLDA Training
    '''
    V_ = np.random.normal(loc=0.0, scale=1.0, size=(D_, num_eigen_voices_)) # eigenvoices matrix
    Lambda_ = np.linalg.pinv(np.divide(S_, N_)) #  inverse noise variance term
    min_divergence_ = True
    
    print(f'eigenvoices matrix V_={np.shape(V_)} det={np.linalg.det(V_)}')
    print(f'global second-order moment S_={np.shape(S_)} det={np.linalg.det(S_)}')
    print(f'S/N={np.shape(np.divide(S_,N_))}  det={np.linalg.det(np.divide(S_,N_))}')
    print(f'inverse noise variance term Lambda_={np.shape(Lambda_)} det={np.linalg.det(Lambda_)}')

    
    for iter_i_ in range(num_iter): # num_iter
        # print(f'G-PLDA: iter:{iter_i_}')
        # EXPECTATION
        gamma_ = np.zeros((num_eigen_voices_, num_eigen_voices_))
        EyTotal_ = np.zeros((num_eigen_voices_, K_))
        R_ = np.zeros((num_eigen_voices_, num_eigen_voices_))
        
        # print(f'gamma_={np.shape(gamma_)}')
        # print(f'EyTotal_={np.shape(EyTotal_)}')
        # print(f'R_={np.shape(R_)}')
                
        idx_ = 0
        for uniq_len_ in uniq_lengths_:
            ivector_len_ = uniq_len_
            # print(f'ivector_len_={ivector_len_}')
            
            # Isolate i-vectors of the same given length
            iv_ = ivectors_sorted_[uniq_len_]
            # print(f'iv_={iv_}')
            
            # Calculate M
            M_ = np.linalg.pinv(ivector_len_*(V_.T @ (Lambda_ @ V_)) + np.eye(num_eigen_voices_)) # Equation (A.7) in [13]
            # print(f'M_={np.shape(M_)} det={np.linalg.det(M_)}')
            
            # Loop over each speaker for the current i-vector length
            for spk_count_ in iv_.keys():
                # First moment of latent variable for V
                # print(f'{idx_} f_={np.sum(f_[:,idx_])}')
                Ey_ = M_ @ V_.T @ Lambda_ @ f_[:, idx_] # Equation (A.8) in [13]
                # print(f'Ey_={np.shape(Ey_)}')
                # print(f'Ey_={Ey_}')
                
                # Calculate second moment.
                Eyy_ = Ey_ @ Ey_.T
                # print(f'Eyy_={np.shape(Eyy_)}')
                
                # Update Ryy 
                R_ += ivector_len_*(M_ + Eyy_) # Equation (A.13) in [13]
                # print(f'R_={np.shape(R_)}')
                
                # Append EyTotal
                EyTotal_[:, idx_] = Ey_
                # print(f'EyTotal_={np.shape(EyTotal_)}')

                idx_ += 1
                
                # If using minimum divergence, update gamma.
                if min_divergence_:
                    gamma_ += (M_ + Eyy_) # Equation (A.18) in [13]
                    # print(f'gamma_={np.shape(gamma_)}')

        # print(f'gamma_={np.shape(gamma_)}')

        # Calculate T
        TT_ = EyTotal_ @ f_.T # Equation (A.12) in [13]
        # print(f'TT_={np.shape(TT_)}')
        
        # MAXIMIZATION
        V_ = TT_.T @ np.linalg.pinv(R_) # Equation (A.16) in [13]
        # print(f'V_={np.shape(V_)}')
        Lambda_ = np.linalg.pinv(np.divide((S_ - V_ @ TT_), N_)) # Equation (A.17) in [13]
        # print(f'Lambda_={np.shape(Lambda_)}')

        print(f'eigenvoices matrix V_={np.shape(V_)} det={np.linalg.det(V_)}')
        print(f'inverse noise variance term Lambda_={np.shape(Lambda_)} det={np.linalg.det(Lambda_)}')
    
        # MINIMUM DIVERGENCE
        if min_divergence_:
            print(f'gamma_ det={np.linalg.det(gamma_)} {np.linalg.det(gamma_@gamma_.T)}')
            print(f'EigenVoices={np.linalg.det(V_)}')
            if np.linalg.det(gamma_)<=0:
                continue
            gamma_ = np.divide(gamma_, K_) # Equation (A.18) in [13]
            V_ = V_ @ np.linalg.cholesky(gamma_) # Equation (A.22) in [13]. 
            # print(f'V_={np.shape(V_)}')
            
            # Originally cholesky decomposition of gamma_ is performed. But, this operation requires a positive definite matrix and gamma_ is not positive definite always. Hence, (gamma_ @ gamma_.T) is performed  
            # V_ = V_ @ np.linalg.cholesky(gamma_ @ gamma_.T) # Equation (A.22) in [13]. 
        
    
    # sys.exit(0)

    
    print(f'mu_={np.shape(mu_)}')
    print(f'WhiteningMatrix={np.shape(W_)}')
    print(f'EigenVoices={np.shape(V_)}')
    print(f'Sigma={np.shape(np.linalg.pinv(Lambda_))}')


    gpldaModel_ = {
        'mean': mu_,
        'WhiteningMatrix': W_,
        'EigenVoices': V_,
        'Sigma': np.linalg.pinv(Lambda_),
        }
    
    return gpldaModel_, projection_matrix_




def compute_gplda_score(gplda_model, w1, wt):
    '''
    Compute the G-PLDA score between enrollment ivector and test ivector

    Parameters
    ----------
    gplda_model : dict
        Dictionary object containing the G-PLDA model.
    w1 : ndarray
        Enrollment speaker i-vector.
    wt : ndarray
        Test utterrance i-vector.

    Returns
    -------
    score_ : float
        G-PLDA score.

    '''
    # Center the data
    w1 = np.subtract(w1, gplda_model['mean'])
    wt = np.subtract(wt, gplda_model['mean'])
    # print(f'w1={np.shape(w1)}')
    # print(f'wt={np.shape(wt)}')
    
    # Whiten the data
    w1 = gplda_model['WhiteningMatrix'] @ w1
    wt = gplda_model['WhiteningMatrix'] @ wt
    # print(f'w1={np.shape(w1)}')
    # print(f'wt={np.shape(wt)}')
    
    # Length-normalize the data
    # w1 = np.divide(w1, np.repeat(np.array(np.linalg.norm(w1), ndmin=2), np.shape(w1)[0], axis=0)).T
    w1 = np.divide(w1, np.linalg.norm(w1))
    # wt = np.divide(wt, np.repeat(np.array(np.linalg.norm(wt), ndmin=2), np.shape(wt)[0], axis=0)).T
    wt = np.divide(wt, np.linalg.norm(wt))
    # print(f'w1={np.shape(w1)}')
    # print(f'wt={np.shape(wt)}')
    
    # Score the similarity of the i-vectors based on the log-likelihood.
    VVt_ = gplda_model['EigenVoices'] @ gplda_model['EigenVoices'].T
    SVVt_ = gplda_model['Sigma'] + VVt_
    # print(f'VVt_={np.shape(VVt_)}')
    # print(f'SVVt_={np.shape(SVVt_)}')
    
    x_ = np.append(SVVt_, VVt_, axis=1)
    # print(f'x_={np.shape(x_)}')
    y_ = np.append(VVt_, SVVt_, axis=1)
    # print(f'y_={np.shape(y_)}')
    z_ = np.append(x_, y_, axis=0)
    # print(f'z_={np.shape(z_)}')
    term1_ = np.linalg.pinv(z_)
    # print(f'term1_={np.shape(term1_)}')
    term2_ = np.linalg.pinv(SVVt_)
    # print(f'term2_={np.shape(term2_)}')
    
    w1wt_ = np.append(w1, wt, axis=0)
    # print(f'w1wt_={np.shape(w1wt_)}')
    score_ = -w1wt_.T @ term1_ @ w1wt_ + w1.T @ term2_ @ w1 + wt.T @ term2_ @ wt # Modified by jagabandhu 16/04 ref. paper 2 the Eq 5 and 6, in eq 5 to 6, there is a sign mismatch, has been corrected here   
    # print(f'score_={np.shape(score_)} {score_}')
    
    return score_
