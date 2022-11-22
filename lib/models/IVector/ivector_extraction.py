#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:31:50 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
@matlab_code_source: https://in.mathworks.com/help/audio/ug/speaker-verification-using-ivectors.html#mw_rtc_SpeakerVerificationUsingIVectorsExample_D8A31795
"""

from sklearn.mixture import GaussianMixture
import numpy as np
from lib.models.GMM_UBM.speaker_adaptation import SpeakerAdaptation
import pickle
import os
# from sklearn.preprocessing import StandardScaler
# from lib.metrics.performance_metrics import PerformanceMetrics
import psutil
# import sys
import time
from lib.models.IVector.gplda import GPLDA_computation, compute_gplda_score
import csv
from lib.metrics.performance_metrics import PerformanceMetrics



class IVector:
    NCOMP = 0
    UBM_DIR = ''
    MODEL_DIR = ''
    BACKGROUND_MODEL = None
    FEATURE_SCALING = 0
    SCALER = None
    N_BATCHES = 0
    
    
    def __init__(self, ubm_dir, model_dir, opDir, num_mixtures=128, ivec_dim=100, feat_scaling=0, tv_iteration=10, mem_limit=5000, lda=True, wccn=True):
        self.UBM_DIR = ubm_dir
        self.MEM_LIMIT = mem_limit
        self.MODEL_DIR = model_dir
        self.OPDIR = opDir
        self.NCOMP = num_mixtures
        self.IVEC_DIM = ivec_dim
        self.FEATURE_SCALING = feat_scaling
        self.N_BATCHES = 1
        self.TV_ITERATION = tv_iteration
        self.LDA = lda
        self.WCCN = wccn
    
    
    def train_ubm(self, X, ram_mem_req, cov_type='diag', ubm_dir='', max_iterations=100, num_init=1, verbosity=1):
        '''
        Training the GMM Universal Background Model.

        Parameters
        ----------
        X_combined_ : array
            All feature arrays.
        cov_type : str, optional
            Type of covariance to be used to train the GMM. The default is 'diag'.
        max_iterations : int, optional
            Maximum number of iterations to train the GMM. The default is 100.
        num_init : int, optional
            Number of random initializations of the GMM model. The default is 3.
        verbosity : int, optional
            Flag to indicate whether to print GMM training outputs. The default is 1.

        Returns
        -------
        None.

        '''

        ram_mem_avail_ = psutil.virtual_memory().available >> 20 # in MB; >> 30 in GB
        print(f'Available RAM: {ram_mem_avail_} MB')
        print(f'RAM required: {ram_mem_req} MB')

        if (ram_mem_req>ram_mem_avail_) or (ram_mem_req>self.MEM_LIMIT):
            self.N_BATCHES = int(ram_mem_req/self.MEM_LIMIT) # int(np.ceil(ram_mem_req_/(0.1*ram_mem_avail_)))
            '''
            Batch-wise training GMM-UBM
            '''
            print(f'Training GMM-UBM in a batch-wise manner. # Batches={self.N_BATCHES}')
            
            speaker_ids_ = [spId_ for spId_ in X.keys()]
            np.random.shuffle(speaker_ids_)
            
            for batch_i_ in range(self.N_BATCHES):
                if speaker_ids_==[]:
                    break
                X_batch_ = np.empty([])
                data_size_ = 0
                while data_size_<self.MEM_LIMIT:
                    spId_ = speaker_ids_.pop()
                    for split_id_ in X[spId_].keys():
                        if np.size(X_batch_)<=1:
                            X_batch_ = X[spId_][split_id_]
                        else:
                            X_batch_ = np.append(X_batch_, X[spId_][split_id_], axis=0)
                    data_size_ = (np.size(X_batch_)*8 + np.shape(X_batch_)[0]*self.NCOMP*8) >> 20
                print(f'Batch={batch_i_} X_batch={np.shape(X_batch_)} data_size={data_size_}')
                
                if batch_i_==0:
                    training_success_ = False
                    reg_covar_ = 1e-6
                    self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbosity, reg_covar=reg_covar_)
                    self.BACKGROUND_MODEL.fit(X_batch_)

                    print(f'Batch: {batch_i_+1} model trained')
                else:
                    adapt_ = None
                    del adapt_
                    adapt_ = SpeakerAdaptation().adapt_ubm(X_batch_.T, self.BACKGROUND_MODEL, use_adapt_w_cov=False)
                    self.BACKGROUND_MODEL.means_ = adapt_['means']
                    self.BACKGROUND_MODEL.weights_ = adapt_['weights']
                    self.BACKGROUND_MODEL.covariances_ = adapt_['covariances']
                    self.BACKGROUND_MODEL.precisions_ = adapt_['precisions']
                    self.BACKGROUND_MODEL.precisions_cholesky_ = adapt_['precisions_cholesky']
                    print(f'Batch: {batch_i_+1} model updated')
        
        else:
            X_combined_ = np.empty([], dtype=np.float32)
            speaker_count_ = 0
            for speaker_id_ in X.keys():
                split_count_ = 0
                for split_id_ in X[speaker_id_].keys():
                    if np.size(X_combined_)<=1:
                        X_combined_ = np.array(X[speaker_id_][split_id_], dtype=np.float32)
                    else:
                        X_combined_ = np.append(X_combined_, np.array(X[speaker_id_][split_id_], dtype=np.float32), axis=0)
                    split_count_ += 1
                    # print(f'Splits per speaker: ({split_count_}/{len(X[speaker_id_].keys())})', end='\r', flush=True)
                # print('')
                speaker_count_ += 1
                print(f'Speaker-wise data combination: ({speaker_count_}/{len(X.keys())})', end='\r', flush=True)
            print('')
            print(f'Development data shape={np.shape(X_combined_)} data_type={X_combined_.dtype}')


            training_success_ = False
            reg_covar_ = 1e-6
            while not training_success_:
                try:
                    self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbosity, reg_covar=reg_covar_)
                    self.BACKGROUND_MODEL.fit(X_combined_)
                    training_success_ = True
                except:
                    reg_covar_ = np.max([reg_covar_*10, 1e-1])
                    print(f'Singleton component error. Reducing reg_covar to {np.round(reg_covar_,6)}')
        
        ubm_fName = ubm_dir + '/ubm.pkl'
        with open(ubm_fName, 'wb') as f_:
            pickle.dump({'model':self.BACKGROUND_MODEL, 'scaler':self.SCALER}, f_, pickle.HIGHEST_PROTOCOL)
        
        return


    
    
    def Baum_Welch_Statistics(self, X):
        '''
        % Collect sufficient stats in baum-welch fashion
        %  [N, F] = collect_suf_stats(FRAMES, M, V, W) returns the vectors N and
        %  F of zero- and first- order statistics, respectively, where FRAMES is a 
        %  dim x length matrix of features, M is dim x gaussians matrix of GMM means
        %  V is a dim x gaussians matrix of GMM variances, W is a vector of GMM weights.

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        gmm : object
            GMM-UBM model.

        Returns
        -------
        N_ : ndarray
            Zeroeth-order statistics for all utterance utterance
        F_ : ndarray
            First-order statistics for all utterance utterance

        '''
        X_utterance_ = {}
        for speaker_id_ in X.keys():
            for split_id_ in X[speaker_id_].keys():
                X_utterance_[split_id_] = np.array(X[speaker_id_][split_id_], dtype=np.float32)

        if not self.BACKGROUND_MODEL:
            ubm_fName_ = self.MODEL_DIR + '/ubm.pkl'
            if not os.path.exists(ubm_fName_):
                print('Background model does not exist')
                return
            try:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)['model']
                with open(ubm_fName_, 'rb') as f_:
                    self.SCALER = pickle.load(f_)['scaler']
            except:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)
        
        N_ = {}
        F_ = {}
        # ncomp_ = self.BACKGROUND_MODEL.means_.shape[0]
        # nfeat_ = self.BACKGROUND_MODEL.means_.shape[1]
        split_count_ = 0
        for split_id_ in X_utterance_.keys():
            # compute the GMM posteriors for the given data
            gammas_ = self.BACKGROUND_MODEL.predict_proba(X_utterance_[split_id_]) # n_samples, n_components
            
            # zero order stats for each Gaussian are just sum of the posteriors (soft counts)
            Nc_ = np.sum(gammas_, axis=0) # n_components
            # N_[split_id_] = np.tile(np.multiply(Nc_, np.eye(ncomp_)), [nfeat_, nfeat_])
            N_[split_id_] = Nc_ # To save memory
            
            # first order stats is just a (posterior) weighted sum
            Fc_ = np.matmul(X_utterance_[split_id_].T, gammas_) # n_dim, n_components
            Fc_ -= np.multiply(np.repeat(np.array(Nc_, ndmin=2), self.BACKGROUND_MODEL.means_.shape[1], axis=0), self.BACKGROUND_MODEL.means_.T)
            Fc_ = np.array(Fc_.flatten(), ndmin=2).T # n_dim*n_component, 1
            F_[split_id_] = Fc_
            split_count_ += 1
            
        return N_, F_




    def cosine_kernal_scoring_on_spkr_factors(self, trn_y, tst_y):
        M_ =  trn_y
        F_ =  tst_y

        # Normalization        
        M_ = np.divide(M_, np.repeat(np.array(np.sqrt(np.sum(np.power(M_), axis=1))), ndmin=2).T, np.shape(M_)[1], axis=1)
        F_ = np.divide(F_, np.repeat(np.array(np.sqrt(np.sum(np.power(F_), axis=1))), ndmin=2).T, np.shape(F_)[1], axis=1)
        
        scores = np.inner(M_, F_)
        return scores



    def train_tv_matrix(self, X, tv_fName):
        '''
        Training the Total Variability matrix

        Parameters
        ----------
        X : dict
            Dictionary containing the speaker-wise data.
        tv_fName : str
            Filename to store the trained TV matrix.

        Returns
        -------
        None.

        '''
        
        if not self.BACKGROUND_MODEL:
            ubm_fName_ = self.UBM_DIR + '/ubm.pkl'
            if not os.path.exists(ubm_fName_):
                print('Background model does not exist')
                return
            try:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)['model']
                with open(ubm_fName_, 'rb') as f_:
                    self.SCALER = pickle.load(f_)['scaler']
            except:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)
        print('Background model loaded')
        
        Mu_ = self.BACKGROUND_MODEL.means_ # UBM Mean super-vector
        Sigma_ = np.empty([], dtype=np.float32)
        if len(np.shape(self.BACKGROUND_MODEL.covariances_))==3:
            Sigma_ = np.zeros((np.shape(self.BACKGROUND_MODEL.covariances_)[0], np.shape(self.BACKGROUND_MODEL.covariances_)[1]))
            for comp_i_ in range(np.shape(self.BACKGROUND_MODEL.covariances_)[0]):
                Sigma_[comp_i_,:] = np.diag(self.BACKGROUND_MODEL.covariances_[comp_i_,:,:])
        else:
            Sigma_ = self.BACKGROUND_MODEL.covariances_
        Sigma_ = Sigma_.flatten() # UBM Variance super-vector
        
        # spk_ids_ = X.keys()
        
        suf_stats_fName_ = self.MODEL_DIR+'/Sufficient_Stats.pkl'
        if not os.path.exists(suf_stats_fName_):
            N_, F_ = self.Baum_Welch_Statistics(X)
            with open(suf_stats_fName_, 'wb') as f_:
                pickle.dump({'0th_order':N_, '1st_order': F_}, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(suf_stats_fName_, 'rb') as f_:
                stats_ = pickle.load(f_)
                N_ = stats_['0th_order']
                F_ = stats_['1st_order']
        
        # print(f'0th order stats: {len(N_)}')
        # print(f'1st order stats: {len(F_)}')
        
        print('Initializing T matrix (randomly)')
        # print(f'F_={np.shape(F_)} Sigma_={np.shape(Sigma_)}')
        T_mat_ = np.random.normal(loc=0.0, scale=0.0001, size=(np.shape(Sigma_)[0], self.IVEC_DIM))
        T_mat_ /= np.linalg.norm(T_mat_) # (19968 x 100)
        # print(f'T_mat_ = {T_mat_.shape}')
        
        # we don't use the second order stats - set 'em to empty matrix
        # S_ = []

        # n_speakers_ = len(spk_ids_)
        # n_sessions_ = np.shape(spk_ids_)[0]
        n_feat_ = Mu_.shape[1]
        n_comp_ = Mu_.shape[0]
        
        I_ = np.eye(self.IVEC_DIM)
        # print(f'I_={I_.shape}')
        Ey_ = {} # numSpeakers_
        Eyy_ = {} # numSpeakers_
        Linv_ = {} # numSpeakers_
                
        for iter_i_ in range(self.TV_ITERATION):
            startTime = time.process_time()

            # 1. Calculate the posterior distribution of the hidden variable
            TtimesInverseSSdiag_ = np.divide(T_mat_, np.repeat(np.array(Sigma_, ndmin=2).T, self.IVEC_DIM, axis=1)+1e-10).T # (19968 x 100)
            # print(f'TtimesInverseSSdiag_={np.shape(TtimesInverseSSdiag_)}')
            
            for s_ in N_.keys():
                # print(f'N_[s_]={N_[s_].shape}')
                Nc_ = np.repeat(np.array(np.tile(N_[s_], [n_feat_]), ndmin=2), self.IVEC_DIM, axis=0)
                # print(f'I_={I_.shape} TtimesInverseSSdiag_={TtimesInverseSSdiag_.shape}, Nc_={Nc_.shape} T_mat_={T_mat_.shape}')
                L_ = I_ + (np.multiply(TtimesInverseSSdiag_, Nc_) @ T_mat_)
                # print(f'L_={L_.shape}')
                Linv_[s_] = np.linalg.pinv(L_)
                # print(f'Linv_={Linv_[s_].shape}')
                Ey_[s_] = Linv_[s_] @ TtimesInverseSSdiag_ @ F_[s_]
                # print(f'Ey_={Ey_[s_].shape}')
                Eyy_[s_] = Linv_[s_] + Ey_[s_] @ Ey_[s_].T
                # print(f'Eyy_={Eyy_[s_].shape}')

            # 2. Accumlate statistics across the speakers
            Eymat_ = []
            for s_ in Ey_.keys():
                Eymat_.append(Ey_[s_].flatten())
            # print(f'Eymat_={np.shape(Eymat_)}')
            FFmat_ = []
            for s_ in F_.keys():
                FFmat_.append(F_[s_].flatten())
            # print(f'FFmat_={np.shape(FFmat_)}')
            Kt_ = np.array(FFmat_, ndmin=2).T @ np.array(Eymat_, ndmin=2) # (19968, 100)
            # print(f'Kt_={np.shape(Kt_)}')
            
            K_ = {}
            for dim_i_ in range(self.IVEC_DIM):
                K_[dim_i_] = np.reshape(Kt_[:, dim_i_], newshape=(n_feat_, n_comp_)) # (39, 512)
                # print(f'{s_} {K[s_].shape}')

            newT_ = {}
            for c_ in range(n_comp_):
                AcLocal_ = np.zeros((self.IVEC_DIM, self.IVEC_DIM)) # (100, 100)
                for s_ in Eyy_.keys():
                    Nc_ = np.repeat(np.array(N_[s_], ndmin=2), self.IVEC_DIM, axis=0)
                    # print(f'N_[s_] {N_[s_].shape} Nc_={Nc_.shape}')
                    # print(f'Eyy_[s_] {Eyy_[s_].shape}')
                    AcLocal_ += Nc_[:, c_] @ Eyy_[s_]
                    # print(f'AcLocal_ {AcLocal_.shape}')

                # 3. Update the Total Variability Space
                Kc_ = np.empty([])
                for dim_i_ in range(self.IVEC_DIM):
                    K_temp_ = np.array(K_[dim_i_][:,c_], ndmin=2)
                    if np.size(Kc_)<=1:
                        Kc_ = K_temp_
                    else:
                        Kc_ = np.append(Kc_, K_temp_, axis=0)
                # print(f'Kc_: {Kc_.shape}')
                newT_[c_] = (np.linalg.pinv(AcLocal_) @ Kc_).T # (39, 100)
                # print(f'newT_[c_]: {newT_[c_].shape}')
            
            for dim_i_ in range(self.IVEC_DIM):
                T_col_ = []
                for c_ in range(n_comp_):
                    T_col_.extend(newT_[c_][:, dim_i_].flatten())
                # print(f'{dim_i_} T_col_={np.shape(T_col_)}')
                T_mat_[:, dim_i_] = np.array(T_col_)
            # print(f'T_mat_: {np.shape(T_mat_)}')

            print(f"Training Total Variability Space: {iter_i_} / {self.TV_ITERATION} complete ({np.round(time.process_time()-startTime,2)} seconds).")

        with open(tv_fName, 'wb') as f_:
            pickle.dump({'tv_mat':T_mat_}, f_, pickle.HIGHEST_PROTOCOL)

        return



    def compute_ivector(self, X, TV):
        if not self.BACKGROUND_MODEL:
            ubm_fName_ = self.UBM_DIR + '/ubm.pkl'
            if not os.path.exists(ubm_fName_):
                print('Background model does not exist')
                return
            try:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)['model']
                with open(ubm_fName_, 'rb') as f_:
                    self.SCALER = pickle.load(f_)['scaler']
            except:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)
        # print('Background model loaded')

        Mu_ = self.BACKGROUND_MODEL.means_ # UBM Mean super-vector (n_comp_, n_feat_)
        Sigma_ = np.empty([], dtype=np.float32)
        if len(np.shape(self.BACKGROUND_MODEL.covariances_))==3:
            Sigma_ = np.zeros((np.shape(self.BACKGROUND_MODEL.covariances_)[0], np.shape(self.BACKGROUND_MODEL.covariances_)[1]))
            for comp_i_ in range(np.shape(self.BACKGROUND_MODEL.covariances_)[0]):
                Sigma_[comp_i_,:] = np.diag(self.BACKGROUND_MODEL.covariances_[comp_i_,:,:])
        else:
            Sigma_ = self.BACKGROUND_MODEL.covariances_
        Sigma_ = Sigma_.flatten() # UBM Variance super-vector

        lld_ = self.BACKGROUND_MODEL.predict_proba(X) # (1280, 512)
        # Compute a posteriori normalized probability
        amax = np.max(lld_, axis=0) # (512,)
        amax_arr = np.repeat(np.array(amax, ndmin=2), np.shape(lld_)[0], axis=0) # (1280, 512)
        lld_sum_ = amax + np.log(np.sum(np.exp(np.subtract(lld_,amax_arr)), axis=0)) # (512,)
        gamma_ = np.exp(np.subtract(lld_, np.repeat(np.array(lld_sum_, ndmin=2), lld_.shape[0], axis=0))) # (512, 1280)
                
        # Compute Baum-Welch statistics
        n_ = np.sum(gamma_, axis=0) # (1280,)                
        f_ = X.T @ gamma_ - np.multiply(np.repeat(np.array(n_, ndmin=2), Mu_.shape[1], axis=0), Mu_.T) # ((39, 512))
        
        TS_ = np.divide(TV, np.repeat(np.array(Sigma_, ndmin=2).T, TV.shape[1], axis=1)) # (19968, 100)
        TSi_ = TS_.T # (19968, 100)
        I_ = np.eye(self.IVEC_DIM) # (100, 100)
        n_feat_ = Mu_.shape[1] # 39
        TS_temp_ = np.multiply(TS_, np.repeat(np.array(np.tile(n_, [n_feat_]), ndmin=2).T, self.IVEC_DIM, axis=1)) # (19968, 100)
        w_ = np.linalg.pinv(I_ + TS_temp_.T @ TV) @ TSi_ @ f_.flatten()
        # print(f'w_={np.shape(w_)}')
        
        return w_
        


    def extract_ivector(self, X, tv_fName, opDir):
        '''
        Extracting the ivectors for the given data

        Parameters
        ----------
        X : dict
            Dictionary object containing speaker-wise data.
        tv_fName : str
            Filename of the trained TV matrix.
        opDir : TYPE
            Directory to store the extracted i-vectors.

        Returns
        -------
        None.

        '''
        if not os.path.exists(opDir):
            os.makedirs(opDir)
                
        with open(tv_fName, 'rb') as f_:
            TV_ = pickle.load(f_)
            TV_ = TV_['tv_mat']

        for speaker_id_ in X.keys():
            ivec_speaker_path_ = opDir + '/' + speaker_id_ + '/' 
            if not os.path.exists(ivec_speaker_path_):
                os.makedirs(ivec_speaker_path_)
            ivec_fName_ = ivec_speaker_path_ + '/' + speaker_id_ + '.pkl'
            if os.path.exists(ivec_fName_):
                print(f'I-vector already computed for {speaker_id_}')
                continue
            
            I_vectors_ = np.zeros((len(X[speaker_id_]), self.IVEC_DIM))
            utter_count_ = 0
            for split_id_ in X[speaker_id_].keys():
                X_ = X[speaker_id_][split_id_] # (1280, 39)
                w_ = self.compute_ivector(X_, TV_)                
                I_vectors_[utter_count_,:] = w_
                utter_count_ += 1
            
            # print(f'ivec_fName_={ivec_fName_}')
            with open(ivec_fName_, 'wb') as f_:
                pickle.dump(I_vectors_, f_, pickle.HIGHEST_PROTOCOL)
        
        return
    
    
    
    def plda_training(self, X, model_dir):
        '''
        Training PLDA model.

        Parameters
        ----------
        X : dict
            Dictionary containing speaker-wise data.
        model_dir : str
            Directory to store the PLDA model.

        Returns
        -------
        None.

        '''
        ivector_per_speaker_ = {}
        for speaker_id_ in X.keys():
            ivec_fName_ = model_dir + '/' + speaker_id_ + '/' + speaker_id_ + '.pkl'
            with open(ivec_fName_, 'rb') as f_:
                I_vectors_ = pickle.load(f_)
                ivector_per_speaker_[speaker_id_] = I_vectors_.T
                print(f'speaker={speaker_id_} ivector={np.shape(ivector_per_speaker_[speaker_id_])}')
        
        gpldaModel_, projectionMatrix_ = GPLDA_computation(ivector_per_speaker_, lda_dim=20, perform_LDA=self.LDA, perform_WCCN=self.WCCN, num_iter=50)
        
        return gpldaModel_, projectionMatrix_



    def perform_testing(self, enr_ivec_dir, tv_fName, classifier=None, opDir='', feat_info=None,  dim=None, test_key=None, duration=None):
        '''
        Test i-vector based speaker recognition system

        Parameters
        ----------
        enr_ivec_dir : str
            Enrollment i-vector directory.
        tv_fName : str
            Trained Total Variability matrix file.
        classifier : dict, optional
            G-PLDA model and projection matrix. The default is None.
        opDir : str, optional
            Path to dump output files. The default is ''.
        feat_info : dict, optional
            Dictionary containing speaker-wise feature info. The default is None.
        dim : int, optional
            Dimensions of input features. The default is None.
        test_key : str, optional
            Path to test key csv file. The default is None.
        duration : float, optional
            Test utterance chop duation. The default is None.

        Returns
        -------
        scores_ : dict
            Scores computed for all test utterances.

        '''
        if duration:
            score_fName_ = opDir + '/Test_Scores_' + str(duration) + 's.pkl'
        else:
            score_fName_ = opDir + '/Test_Scores.pkl'
    
        score_fName_ = opDir + '/Test_Scores.pkl'
        print(f'scores={os.path.exists(score_fName_)}')
        if not os.path.exists(score_fName_):    
            scores_ = {}
            true_lab_ = []
            pred_lab_ = []
            
            '''
            Loading Total Variability matrix
            '''
            with open(tv_fName, 'rb') as f_:
                TV_ = pickle.load(f_)
                TV_ = TV_['tv_mat']
            
            ''' Loading the speaker models '''
            enrolled_speakers_ = next(os.walk(enr_ivec_dir))[1]
            enr_ivectors_ = {}
            for speaker_id_ in enrolled_speakers_:
                ivec_fName_ = enr_ivec_dir + '/' + speaker_id_ + '/' + speaker_id_ + '.pkl'
                with open(ivec_fName_, 'rb') as f_:
                    I_vectors_ = pickle.load(f_)
                    enr_ivectors_[speaker_id_] = I_vectors_.T
                                
            confusion_matrix_ = np.zeros((len(enrolled_speakers_), len(enrolled_speakers_)))
            match_count_ = np.zeros(len(enrolled_speakers_))
            
            '''
            Testing every test utterance one by one 
            '''
            total_splits_ = 0
            
            speaker_id_list_ = []
            for split_id_ in feat_info.keys():
                if not split_id_.split('_')[-2]=='x':
                    split_dur_ = int(split_id_.split('_')[-2])
                else:
                    split_dur_ = duration
                if duration:
                    if not split_dur_==duration:
                        continue
                speaker_id_list_.append(feat_info[split_id_]['speaker_id'])
                total_splits_ += 1
                
            output_fName_ = opDir + '/' + test_key.split('/')[-1].split('.')[0] + '_predictions.csv'
            with open(output_fName_, 'a+', encoding='utf8') as fid_:
                writer_ = csv.writer(fid_)
                writer_.writerow([
                    'utterance_id',
                    'speaker_id', 
                    'score', 
                    ])
                
            split_count_ = 0
            with open(test_key, 'r' ) as test_meta_info_:
                reader_ = csv.DictReader(test_meta_info_)
                for row_ in reader_:
                    split_id_ = row_['split_id']
                    # print(row_['cohorts'])
                    cohort_speakers_ = row_['cohorts'].split('|')
                    # print(f'cohort_speakers_={cohort_speakers_}')
                            
                    split_count_ += 1    
                    speaker_id_ = feat_info[split_id_]['speaker_id']
                    test_ivec_opDir_ = opDir + '/ivectors/' + speaker_id_ + '/'
                    if not os.path.exists(test_ivec_opDir_):
                        os.makedirs(test_ivec_opDir_)
                        
                    test_ivec_fName_ = test_ivec_opDir_ + '/' + split_id_ + '.pkl'
                    test_ivec_ = np.empty([])
                    if not os.path.exists(test_ivec_fName_):
                        feature_path_ = feat_info[split_id_]['file_path']
                        fv_ = None
                        del fv_
                        fv_ = np.load(feature_path_, allow_pickle=True)
                        # The feature vectors must be stored as individual rows in the 2D array
                        if dim:
                            if np.shape(fv_)[0]==dim:
                                fv_ = fv_.T
                        elif np.shape(fv_)[1]>np.shape(fv_)[0]:
                            fv_ = fv_.T
        
                        ''' Feature Scaling '''
                        if self.FEATURE_SCALING>0:
                            fv_ = self.SCALER.transform(fv_)
                            fv_ = fv_.astype(np.float32)
                        
                        test_ivec_ = self.compute_ivector(fv_, TV_)
                        with open(test_ivec_fName_, 'wb') as ivec_fid_:
                            pickle.dump(test_ivec_, ivec_fid_, pickle.HIGHEST_PROTOCOL)
                        # print(f'Test {split_id_} ivector saved')
                    else:
                        with open(test_ivec_fName_, 'rb') as ivec_fid_:
                            test_ivec_ = pickle.load(ivec_fid_)
                        # print(f'Test {split_id_} ivector loaded')

                    if (self.LDA) or (self.WCCN):
                        test_ivec_ = classifier['projection_matrix'] @ test_ivec_

                    gplda_scores_ = {}
                    for index_i_ in enr_ivectors_.keys():
                        if index_i_ in cohort_speakers_:
                            spk_scores_ = []
                            for utter_i_ in range(enr_ivectors_[index_i_].shape[1]):
                                enr_ivec_ = enr_ivectors_[index_i_][:,utter_i_]
                                if (self.LDA) or (self.WCCN):
                                    enr_ivec_ = classifier['projection_matrix'] @ enr_ivec_
                                spk_scores_.append(compute_gplda_score(classifier['gplda_model'], enr_ivec_, test_ivec_))
                            gplda_scores_[index_i_] = np.min(spk_scores_)
                    # print(gplda_scores_)
                    
                    scores_[split_id_] = {
                        'speaker_id':speaker_id_, 
                        'gplda_scores': gplda_scores_, 
                        'enrolled_speakers': enrolled_speakers_,
                        }

                    for cohort_id_ in gplda_scores_.keys():
                        output_fName_ = opDir+'/'+test_key.split('/')[-1].split('.')[0]+'_predictions.csv'
                        with open(output_fName_, 'a+', encoding='utf8') as fid_:
                            writer_ = csv.writer(fid_)
                            writer_.writerow([
                                row_['utterance_id'],
                                cohort_id_, 
                                gplda_scores_[cohort_id_], 
                                ])

                    '''
                    Displaying progress
                    '''
                    print(f'\t{split_id_}', end='\t', flush=True)
                    print(f'splits=({split_count_}/{total_splits_})', end='\t', flush=True)
                    print(f'true=({speaker_id_})', end='\t', flush=True)
                    print(f'cohort scores={gplda_scores_}', end='\n', flush=True)
                    
            '''
            Saving the scores for the selected duration
            '''
            with open(score_fName_, 'wb') as f_:
                pickle.dump(scores_, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(score_fName_, 'rb') as f_:
                scores_ = pickle.load(f_)
        
        return scores_



    def evaluate_performance(self, res, opDir, duration):
        '''
        Compute performance metrics.

        Parameters
        ----------
        res : dict
            Dictionary containing the sub-utterance wise scores.
        opDir : str
            Output path.
        duration : int
            Duration of the testing utterance.

        Returns
        -------
        metrics_ : dict
            Disctionary containg the various evaluation metrics:
                accuracy, precision, recall, f1-score, eer

        '''
        groundtruth_scores_ = np.empty([])
        predicted_scores_ = np.empty([])
        for split_id_ in res.keys():
            true_speaker_id_ = res[split_id_]['speaker_id']
            if true_speaker_id_=='0000':
                continue
            cohort_scores_ = res[split_id_]['gplda_scores']
            
            gt_score_ = []
            ptd_scores_ = []
            for cohort_id_ in cohort_scores_.keys():
                if cohort_id_==true_speaker_id_:
                    gt_score_.append(1)
                else:
                    gt_score_.append(0)
                ptd_scores_.append(cohort_scores_[cohort_id_])
                
            if np.size(groundtruth_scores_)<=1:
                groundtruth_scores_ = np.array(gt_score_)
                predicted_scores_ = np.array(ptd_scores_)
            else:
                groundtruth_scores_ = np.append(groundtruth_scores_, np.array(gt_score_), axis=0)
                predicted_scores_ = np.append(predicted_scores_, np.array(ptd_scores_), axis=0)
                
        if not np.sum(groundtruth_scores_)==0:            
            if not np.size(groundtruth_scores_)<=1:
                FPR_, TPR_, EER_, EER_thresh_ = PerformanceMetrics().compute_eer(groundtruth_scores_, predicted_scores_)
            else:
                FPR_ = np.empty([])
                TPR_ = np.empty([])
                EER_ = -1
                EER_thresh_ = np.empty([])                
        else:
            FPR_ = np.empty([])
            TPR_ = np.empty([])
            EER_ = -1
            EER_thresh_ = np.empty([])
                
        metrics_ = {
            'fpr': FPR_,
            'tpr': TPR_,
            'eer': EER_,
            'eer_threshold': EER_thresh_,
            }
            
        print(f'\n\nUtterance duration: {duration}s:\n__________________________________________')
        print(f"\tEER: {np.round(np.mean(metrics_['eer'])*100,2)}")

        with open(opDir+'/Performance.txt', 'a+') as f_:
            f_.write(f'Utterance duration: {duration}s:\n__________________________________________\n')
            f_.write(f"\tEER: {np.round(np.mean(metrics_['eer'])*100,2)}\n\n")
        
        return metrics_