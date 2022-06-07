#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:45:17 2022

@author: Jagabandhu Mishra
"""

import numpy as np
from scipy import linalg


class SpeakerAdaptation:
    
    def __init__(self):
        return
    
    def _compute_precision_cholesky(self, covariances, covariance_type):
        """
        Compute the Cholesky decomposition of the precisions.
    
        Parameters
        ----------
        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
    
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.
    
        Returns
        -------
        precisions_cholesky : array-like
            The cholesky decomposition of sample precisions of the current
            components. The shape depends of the covariance_type.
        """
        
        est_prec_errmsg_  =  (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar."
        )
    
        if covariance_type == "full":
            n_components, n_features, _  =  covariances.shape
            precisions_chol  =  np.empty((n_components, n_features, n_features))
            for k, covariance in enumerate(covariances):
                try:
                    cov_chol  =  linalg.cholesky(covariance, lower = True)
                except linalg.LinAlgError:
                    raise ValueError(est_prec_errmsg_)
                precisions_chol[k]  =  linalg.solve_triangular(
                    cov_chol, np.eye(n_features), lower = True
                ).T
        elif covariance_type == "tied":
            _, n_features  =  covariances.shape
            try:
                cov_chol  =  linalg.cholesky(covariances, lower = True)
            except linalg.LinAlgError:
                raise ValueError(est_prec_errmsg_)
            precisions_chol  =  linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower = True
            ).T
        else:
            if np.any(np.less_equal(covariances, 0.0)):
                raise ValueError(est_prec_errmsg_)
            precisions_chol  =  1.0 / np.sqrt(covariances)
        return precisions_chol
    
    
    #precisions_ = precisions_cholesky_**2
    def _compute_precisions(self, precisions_cholesky_, means_, covariance_type):
            # Attributes computation
            _, n_features  =  means_.shape
            if covariance_type == "full":
                precisions_  =  np.empty(precisions_cholesky_.shape)
                for k, prec_chol in enumerate(precisions_cholesky_):
                    precisions_[k]  =  np.dot(prec_chol, prec_chol.T)
            elif covariance_type == "tied":
                precisions_  =  np.dot(
                    precisions_cholesky_, precisions_cholesky_.T
                )
            else:
                precisions_  =  precisions_cholesky_**2
            
            return precisions_
    
        
    def adapt_ubm(self, X, gmm_ubm, use_adapt_w_cov):
        """
        Reference :  "Speaker Verification Using Adapted Gaussian Mixture Models" by Douglas A. Reynolds,
    
        Parameters
        ----------
        X : np array (dim X no. offrames)
            
        gmm_ubm : universal background model (UBM) class object of sklearn gmm 
            (trained using all the features of development data)
            
        use_adapt_w_cov : Binary (True/False) 
            Generally during score computation with respect to adapt model only 
            mean of adapt model is used and weight and covariance of ubm is used 
            according to the requirement flag can be used as True/False
    
        Returns
        -------
        adapt : dictionary
            Having mean, variance, weights and precisions (inverse of coveriance), 
            can later used to create a sklearn gmm model (helpful during testing)
            
         Usase:   
            adapt  =  ubm_adaptation(FV,gmm,use_adapt_w_cov)
            adapt_gmm  =  GaussianMixture(n_components=128, covariance_type='diag')
            adapt_gmm.means_ = adapt['means']
            adapt_gmm.weights_ = adapt['weights']
            adapt_gmm.covariances_ = adapt['covariances']
            adapt_gmm.precisions_ = adapt['precisions']
            adapt_gmm.precisions_cholesky_ = adapt['precisions_cholesky']
    
        """
        
        
        # The difference between 1.0 and the next smallest representable float larger 
        # than 1.0. For example, for 64-bit binary floats in the IEEE-754 standard, 
        # eps = 2**-52, approximately 2.22e-16.
        eps_ = np.finfo(float).eps
        adapt = {}
    
        mv_ = gmm_ubm.means_ #ubm mean
        dcov_ = gmm_ubm.covariances_ #ubm coveriance
        w_ = gmm_ubm.weights_ # ubm weights
        
        gamma_ = gmm_ubm.predict_proba(X.T) # responsibility factor (Eq 7)
    
        ni_ = np.sum(gamma_,axis = 0) # zeroth order statistics (Eq 8)
        ni_ = ((ni_+eps_)/(sum(ni_)+ni_.shape[0]*eps_))*sum(ni_) # to avoid divide by zero error
        
        ex_ = ((X @ gamma_)/ni_).T # 1st order statistics/ expected speaker specific mean (Eq 9)
        
        ex2_ = ((np.square(X) @ gamma_)/ni_).T #2nd order statistics/speaker specific second order moment (Eq 10)
    
        alpha_ = ni_/(ni_+16) # relevence factor (Eq 14)
    
        w_temp_ = ((alpha_*ni_)/X.shape[1])+(w_)*(1-alpha_) # (Eq 11)
        
        temp_alpha_ = np.tile(alpha_,(ex_.shape[1],1)).T   # alpha_i
        temp_alpha_c_ = np.tile((1-alpha_),(ex_.shape[1],1)).T # (1-alpha)
    
        if use_adapt_w_cov:
            # print(w_temp/sum(w_temp))
            adapt['weights'] = w_temp_/sum(w_temp_)   # updated weight (Eq 11)
            adapt['means'] = (ex_*temp_alpha_+mv_*temp_alpha_c_) #updated mean (Eq 12)
            adapt['covariances'] = (ex2_*temp_alpha_+(dcov_+np.square(mv_))*temp_alpha_c_)-np.square(adapt['means']) # updated covariance (Eq 13)
        else:
            adapt['means'] = (ex_*temp_alpha_+mv_*temp_alpha_c_)  #updated mean (Eq 12)
            adapt['weights'] = w_   # ubm weights are assigned to adapt model
            adapt['covariances'] = dcov_ # ubm covarinace is assigned to adapt model
        
        adapt['precisions_cholesky'] = self._compute_precision_cholesky(adapt['covariances'], gmm_ubm.covariance_type)
        adapt['precisions'] = self._compute_precisions(adapt['precisions_cholesky'], gmm_ubm.means_, gmm_ubm.covariance_type)
        
        return adapt
