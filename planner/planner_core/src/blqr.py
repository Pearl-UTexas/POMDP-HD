#!/usr/bin/env python2
# -*- coding: utf-8 -*-
########## Stabilization     

import numpy as np
from py_utils import *


class blqr:
    def __init__(self, nState, nInput, nOutput, Q, R, labda, Q_f):
        self.Q = Q
        self.R = R
        self.nState = nState
        self.nInput = nInput
        self.nOutput = nOutput 
        self.len_s = self.nState*(self.nState+1)/2
        self.id1 = np.triu_indices(self.nState)
        self.labda = labda
        self.Q_f = Q_f
        self.W = 0.5*np.eye(nState)

    def finiteLQR(self, tFinal, A, B, Q, R, F):
        nState = np.shape(A)[0]
        nInput = np.shape(B)[1]
        S = np.zeros((nState,nState,int(tFinal)))
        S[:,:,int(tFinal)-1] = F
        K = np.zeros((nInput,nState,int(tFinal)))

        # Backward Pass
        for t in range(int(tFinal)-2,-1,-1):
            S[:,:,t] = Q + A.T.dot(S[:,:,t+1].dot(A)) - A.T.dot(S[:,:,t+1].dot(B)).dot(np.linalg.inv(B.T.dot(S[:,:,t+1].dot(B)) + R)).dot(B.T.dot(S[:,:,t+1].dot(A)))

        # Forward Pass
        for t in range(int(tFinal)-1):
            K[:,:,t] = np.linalg.inv(B.T.dot(S[:,:,t+1].dot(B)) + R).dot(B.T.dot(S[:,:,t+1].dot(A)))

        return K, S


    def blqr(self, mu, sigma, mu_ref, sig_ref, u_ref, A, B, C, W, tFinal):
        # tFinal = 3.
        cov = symarray(np.zeros((self.nState, self.nState)))
        cov[self.id1] = sigma*1.
        gamma = A.dot(cov*(A.T))
        dg_dsig = A.dot(A.T)
        dcov_dsig = dg_dsig - dg_dsig.dot(C.T.dot(np.linalg.inv(C.dot(gamma.dot(C.T)) + self.W).dot(C.dot(gamma))))+ gamma.dot(C.T.dot(np.linalg.inv(C.dot(gamma.dot(C.T)) + self.W).dot(C.dot(dg_dsig.dot(C.T.dot(np.linalg.inv(C.dot(gamma.dot(C.T)) + self.W)).dot(C.dot(gamma))))))) - gamma.dot(C.T.dot(np.linalg.inv(C.dot(gamma.dot(C.T)) + self.W).dot(C.dot(dg_dsig))))

        d_s_m = np.zeros((self.nState,self.nState))

        # l_row = [0. for i in range(self.nState)]
        # l_row.extend([ds_dsig[0][0]])
        # A_ext[self.nState, :] = l_row

        # Extended Matrices
        A1 = np.concatenate((A, np.zeros((self.nState, self.nState))), axis=1)
        A2 = np.concatenate((d_s_m, dcov_dsig), axis=1)
        A_ext = np.concatenate((A1, A2), axis=0)

        B_ext = np.concatenate((B, np.zeros((self.nState, self.nInput))), axis=0)
        Q_ext = np.concatenate((np.concatenate((self.Q, np.zeros((self.nState, self.nState))), axis=1), np.zeros((self.nState, self.nState+self.nState))), axis=0)
        F = np.concatenate((np.concatenate((self.Q_f, np.zeros((self.nState, self.nState))), axis=1), np.zeros((self.nState, self.nState+self.nState))), axis=0)
        
        final_cov = symarray(np.zeros((self.nState, self.nState)))
        final_cov[self.id1] = self.labda[self.id1]
        F[self.nState:, self.nState:] = final_cov

        [K,S] = self.finiteLQR(tFinal, A_ext, B_ext, Q_ext, self.R, F)

        print "Error in LQR = ", mu-mu_ref

        # print "(sigma-sig_ref) =", sigma-sig_ref
        cov[self.id1] = (sigma-sig_ref)*1.  # we are just controlling covariance along the diagonal

        u = -K[:,:,1].dot(np.concatenate((mu-mu_ref, np.diag(cov)))) + u_ref
        return u
        
    
        
