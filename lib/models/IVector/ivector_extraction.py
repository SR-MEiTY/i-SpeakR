#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 18:31:50 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
@collaborator: Dr. Gayadhar Pradhan, Assoc. Prof., NIT Patna
"""

from sklearn.mixture import GaussianMixture
import numpy as np
from lib.models.GMM_UBM.speaker_adaptation import SpeakerAdaptation
# from lib.models.IVector import sidekit_util
# from lib.models.IVector.statserver import StatServer
# import sidekit
import pickle
import os
from sklearn.preprocessing import StandardScaler
from lib.metrics.performance_metrics import PerformanceMetrics
import psutil
import sys
import time


class IVector:
    NCOMP = 0
    UBM_DIR = ''
    MODEL_DIR = ''
    BACKGROUND_MODEL = None
    FEATURE_SCALING = 0
    SCALER = None
    N_BATCHES = 0
    
    
    def __init__(self, ubm_dir, model_dir, opDir, num_mixtures=128, feat_scaling=0, num_iter=10):
        self.UBM_DIR = ubm_dir
        self.MODEL_DIR = model_dir
        self.OPDIR = opDir
        self.NCOMP = num_mixtures
        self.FEATURE_SCALING = feat_scaling
        self.N_BATCHES = 1
        self.N_ITER = num_iter
    
    
    def train_ubm(self, X, ram_mem_req, cov_type='diag', max_iterations=100, num_init=1, verbosity=1):
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
        
        ubm_fName = self.MODEL_DIR + '/ubm.pkl'
        with open(ubm_fName, 'wb') as f:
            pickle.dump({'model':self.BACKGROUND_MODEL, 'scaler':self.SCALER}, f, pickle.HIGHEST_PROTOCOL)
        
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
        N : ndarray
            Zeroeth-order statistics
        F : ndarray
            First-order statistics

        '''

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
            speaker_count_ += 1
            print(f'Speaker-wise data combination: ({speaker_count_}/{len(X.keys())})', end='\r', flush=True)
        print('')
        print(f'Development data shape={np.shape(X_combined_)} data_type={X_combined_.dtype}')
        

        ''' Feature Scaling '''
        if self.FEATURE_SCALING>0:
            X_combined_ = np.empty([], dtype=np.float32)
            for speaker_id_ in X.keys():
                for split_id_ in X[speaker_id_].keys():
                    if np.size(X_combined_)<=1:
                        X_combined_ = np.array(X[speaker_id_][split_id_], dtype=np.float32)
                    else:
                        X_combined_ = np.append(X_combined_, np.array(X[speaker_id_][split_id_], dtype=np.float32), axis=0)
    
            if self.FEATURE_SCALING==1:
                self.SCALER = StandardScaler(with_mean=True, with_std=False).fit(X_combined_)
                X_combined_ = X_combined_.astype(np.float32)
            elif self.FEATURE_SCALING==2:
                self.SCALER = StandardScaler(with_mean=True, with_std=True).fit(X_combined_)
                X_combined_ = X_combined_.astype(np.float32)


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
        
        # Nc_ = {}
        # Fc_ = {}
        # for speaker_id_ in X.keys():
        #     for split_id_ in X[speaker_id_].keys():
        #         fv_ = None
        #         del fv_
                
        #         # Load features
        #         fv_ = X[speaker_id_][split_id_]
                
        #         # Compute a posteriori log-likelihood
        #         log_lkdh_ = self.BACKGROUND_MODEL.predict_proba(fv_)
                
        #         # Compute a posteriori normalized probability
        #         amax_ = np.max(log_lkdh_, axis=0)
        #         norm_log_lkhd_ = np.subtract(log_lkdh_, np.repeat(np.array(amax_, ndmin=2), np.shape(log_lkdh_)[0], axis=0))
        #         log_lkhd_sum_ = amax_ + np.log(np.sum(np.exp(norm_log_lkhd_), axis=0))
        #         gamma_ = np.exp(np.subtract(log_lkdh_, np.repeat(np.array(log_lkhd_sum_, ndmin=2), np.shape(log_lkdh_)[0], axis=0)))
                
        #         # Compute Baum-Welch statistics
        #         n_ = np.sum(gamma_, axis=0) # zeroeth-order
        #         f_ = np.multiply(fv_.T, gamma_) # first-order
                
        #         Nc_[split_id_] = np.expand_dims(np.aray(n_, ndmin=2), axis=0)
        #         Fc_[split_id_] = np.expand_dims(f_, axis=1)

        # compute the GMM posteriors for the given data
        gammas = self.BACKGROUND_MODEL.predict_proba(X_combined_) # n_samples, n_components
        
        # zero order stats for each Gaussian are just sum of the posteriors (soft counts)
        N = np.sum(gammas, axis=0) # n_components
        
        # first order stats is just a (posterior) weighted sum
        F = np.matmul(X_combined_.T, gammas) # n_dim, n_components
        F = np.array(F.flatten(), ndmin=2).T # n_dim*n_component, 1

        return N, F




    def cosine_kernal_scoring_on_spkr_factors(self, trn_y, tst_y):
        M_ =  trn_y
        F_ =  tst_y

        # Normalization        
        M_ = np.divide(M_, np.repeat(np.array(np.sqrt(np.sum(np.power(M_), axis=1))), ndmin=2).T, np.shape(M_)[1], axis=1)
        F_ = np.divide(F_, np.repeat(np.array(np.sqrt(np.sum(np.power(F_), axis=1))), ndmin=2).T, np.shape(F_)[1], axis=1)
        
        scores = np.inner(M_, F_)
        return scores



    def estimate_y_and_v(self, F, N, S, m, E, d, v, u, z, y, x, spk_ids, accumulator=False):
        '''
        % ESTIMATE_Y_AND_V estimates speaker factors and eigenvoices for
        % joint factor analysis model 
        %
        %
        % [y v]=estimate_y_and_v(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
        %
        % provides new estimates of channel factors, x, and 'eigenchannels', u,
        % given zeroth and first order sufficient statistics (N. F), current
        % hyper-parameters of joint factor analysis  model (m, E, d, u, v) and
        % current estimates of speaker and channel factors (x, y, z)
        %
        % F - matrix of first order statistics (not centered). The rows correspond
        %     to training segments. Number of columns is given by the supervector
        %     dimensionality. The first n collums correspond to the n dimensions
        %     of the first Gaussian component, the second n collums to second 
        %     component, and so on.
        % N - matrix of zero order statistics (occupation counts of Gaussian
        %     components). The rows correspond to training segments. The collums
        %     correspond to Gaussian components.
        % S - NOT USED by this function; reserved for second order statistics
        % m - speaker and channel independent mean supervector (e.g. concatenated
        %     UBM mean vectors)
        % E - speaker and channel independent variance supervector (e.g. concatenated
        %     UBM variance vectors)
        % d - Row vector that is the diagonal from the diagonal matrix describing the
        %     remaining speaker variability (not described by eigenvoices). Number of
        %     columns is given by the supervector dimensionality.
        % v - The rows of matrix v are 'eigenvoices'. (The number of rows must be the
        %     same as the number of columns of matrix y). Number of columns is given
        %     by the supervector dimensionality.
        % u - The rows of matrix u are 'eigenchannels'. (The number of rows must be
        %     the same as the number of columns of matrix x) Number of columns is
        %     given by the supervector dimensionality.
        % y - NOT USED by this function; used by other JFA function as
        %     matrix of speaker factors corresponding to eigenvoices. The rows
        %     correspond to speakers (values in vector spk_ids are the indices of the
        %     rows, therfore the number of the rows must be (at least) the highest
        %     value in spk_ids). The columns correspond to eigenvoices (The number
        %     of columns must the same as the number of rows of matrix v).
        % z - matrix of speaker factors corresponding to matrix d. The rows
        %     correspond to speakers (values in vector spk_ids are the indices of the
        %     rows, therfore the number of the rows must be (at least) the highest
        %     value in spk_ids). Number of columns is given by the supervector 
        %     dimensionality.
        % x - matrix of channel factors. The rows correspond to training
        %     segments. The columns correspond to eigenchannels (The number of columns 
        %     must be the same as the number of rows of matrix u)
        % spk_ids - column vector with rows corresponding to training segments and
        %     integer values identifying a speaker. Rows having same values identifies
        %     segments spoken by same speakers. The values are indices of rows in
        %     y and z matrices containing corresponding speaker factors.
        %
        %
        % y=estimate_y_and_v(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
        %
        % only the speaker factors are estimated
        %
        %
        % [y A C]=estimate_y_and_v(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
        %
        % estimates speaker factors and acumulators A and C. A is cell array of MxM
        % matrices, where M is number of eigenvoices. Number of elements in the
        % cell array is given by number of Gaussian components. C is of the same size
        % at the matrix v.
        %
        %
        % v=estimate_y_and_v(A, C)
        %
        % updates eigenvoices from accumulators A and C. Using F and N statistics
        % corresponding to subsets of training segments, multiple sets of accumulators
        % can be collected (possibly in parallel) and summed before the update. Note
        % that segments of one speaker must not be split into different subsets.

        '''
        
        if accumulator:
            # update v from acumulators A and C
            y = self.update_v(F, N)
        else:            
            # this will just create a index map, so that we can copy the counts n-times (n=dimensionality)
            dim = int(np.ceil(np.shape(F)[1]/np.shape(N)[1]))
            index_map = np.reshape(np.repeat(np.array(list(range(np.shape(N)[1])), ndmin=2), dim, axis=0), (np.shape(F)[1], 1))
            y = np.zeros((len(spk_ids), np.shape(v)[0]))
            
            # if nargout > 1
            A = {}
            for c in range(np.shape(N)[1]):
                A[c] = np.zeros(np.shape(v)[1])
            C = np.zeros(np.shape(v)[0], np.shape(F)[1])
            # end
            
            vEvT = {}
            for c in range(np.shape(N)[1]):
                c_elements = list(range(c*dim, (c+1)*dim))
                vEvT[c] = np.multiply(v[:,c_elements], np.repeat(np.array(np.power(E[c_elements]*1.0,-1), ndmin=2), np.shape(v)[0], axis=0)) @ v[:,c_elements]

            for ii in np.unique(spk_ids):
                speakers_sessions = np.squeeze(np.where(spk_ids == ii))
                Fs = np.sum(F[speakers_sessions,:], axis=0)
                Nss = np.sum(N[speakers_sessions,:], axis=0)
                Ns = Nss[0, index_map]
                Fs = Fs -  np.multiply((m + np.multiply(z[ii,:], d)), Ns)
                for jj in speakers_sessions:
                    Fs = Fs - np.multiply((x[jj,:] @ u), N[jj, index_map])
            
                # L = eye(size(v,1)) + v * diag(Ns./E) * v'
                L = np.eye(np.shape(v)[0])
                for c in range(np.shape(N)[1]):
                    L = L + np.multiply(vEvT[c], Nss[c])
                    
                invL = np.linalg.inv(L)
                y[ii, :] = ((np.divide(Fs, E)) @ v) @ invL
                # if nargout > 1
                invL = invL + (y[ii,:] @ y[ii,:])
                for c in range (np.shape(N)[1]):
                    A[c] += invL @ Nss[c]
                C += y[ii,:] @ Fs
            
            # output new estimates of y and v
            v = self.update_v(A, C)
        
        return y, v
    


    def update_v(self, A, C):
        dim = int(np.shape(C)[1]/len(A))
        for c in range(len(A)):
            c_elements = list(range(c*dim, ((c+1)*dim)))
            C[:,c_elements] = np.linalg.inv(A[c]) @ C[:,c_elements]
        
        return C

       

    def write_list_to_txt(self, txt_file, tmp_list):
        txt_data = ",".join(tmp_list)
        with open(txt_file, "w") as f:
            f.write(txt_data)


        
    def train_t_matrix(self, X_Enr, ivec_dim=100):
        
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
        
        # enr_feat_dir_ = PARAMS['feat_dir'] + '/' + list(filter(None, PARAMS['dev_path'].split('/')))[-1] + '/'
        # leftlist, rightlist = sidekit_util.id_map_list(enr_feat_dir_)
        # self.write_list_to_txt(os.path.join(PARAMS['output_dir'], 'left_list.txt'), leftlist)
        # self.write_list_to_txt(os.path.join(PARAMS['output_dir'], 'right_list.txt'), rightlist)
    
        # tv_idmap = sidekit_util.get_id_map(leftlist, rightlist)
        
        # stat_file = PARAMS['output_dir'] + '/stat_ubm_tv_' + str(PARAMS['distrib_nb']) + '.h5'
        # self.train_tv_matrix(PARAMS, self.BACKGROUND_MODEL, stat_file)
        
        # tv_stat = StatServer.read_subset(stat_file, tv_idmap)
        # tv_mean, tv, _, __, tv_sigma = tv_stat.factor_analysis(
        #     rank_f=int(PARAMS['rank_TV']),
        #     rank_g=0,
        #     rank_h=None,
        #     re_estimate_residual=False,
        #     it_nb=(int(PARAMS['tv_iteration']), 0, 0),
        #     min_div=True,
        #     ubm=self.BACKGROUND_MODEL,
        #     batch_size=int(PARAMS['batch_size']),
        #     num_thread=int(PARAMS['nbThread']),
        #     save_partial=PARAMS['tv_path'],
        #     )
        
        m_ = self.BACKGROUND_MODEL.means_.flatten() # UBM Mean super-vector
        if len(np.shape(self.BACKGROUND_MODEL.covariances_))==3:
            E_ = np.zeros((np.shape(self.BACKGROUND_MODEL.covariances_)[0], np.shape(self.BACKGROUND_MODEL.covariances_)[1]))
            for comp_i_ in range(np.shape(self.BACKGROUND_MODEL.covariances_)[0]):
                E_[comp_i_,:] = np.diag(self.BACKGROUND_MODEL.covariances_[comp_i_,:,:])
        else:
            E_ = self.BACKGROUND_MODEL.covariances_
        E_ = E_.flatten() # UBM Variance super-vector
        
        spk_ids_ = X_Enr.keys()
        
        suf_stats_fName_ = self.MODEL_DIR+'/Sufficient_Stats.pkl'
        if not os.path.exists(suf_stats_fName_):
            N_, F_ = self.Baum_Welch_Statistics(X_Enr)
            with open(suf_stats_fName_, 'wb') as f_:
                pickle.dump({'zeroeth_order':N_, 'first_order': F_}, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(suf_stats_fName_, 'rb') as f_:
                N_ = pickle.load(f_)['zeroeth_order']
                F_ = pickle.load(f_)['first_order']
        
        print('Initializing T matrix (randomly)')
        T_ = (np.random.normal(loc=0.0, scale=1.0, size=(ivec_dim, np.shape(F_)[1])) @ E_) * 0.001
        # we don't use the second order stats - set 'em to empty matrix
        S_ = []
        
        # % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        n_speakers_ = len(spk_ids_)
        n_sessions_ = np.shape(spk_ids_)[0]
        
        # iteratively retrain v
        for ii in range(self.N_ITER):
            startTime = time.process_time()
            print(f'Starting iteration: {ii}')
            w_, T_ = self.estimate_y_and_v(F_, N_, S_, m_, E_, 0, T_, 0, np.zeros(n_speakers_), 0, np.zeros(n_sessions_), spk_ids_)
            print(f'Iteration: {ii} CPU time: {time.process_time()-startTime}')

        T_matrix_fName_ = self.MODEL_DIR+'/T_matrix.pkl'
        with open(T_matrix_fName_, 'wb') as f_:
            pickle.dump({'T_matrix':T_, 'speaker_factors_': w_}, f_, pickle.HIGHEST_PROTOCOL)
                
        return w_, T_

    
    
    # def train_tv_matrix(self, PARAMS, ubm, stat_path):
    #     enroll_stat = sidekit_util.adapt_stats_with_feat_dir(PARAMS['feat_dir'], ubm, int(PARAMS['nbThread']))
    #     enroll_stat.write(stat_path)
