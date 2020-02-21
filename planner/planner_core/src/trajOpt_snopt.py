#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:55:59 2017

@author: Ajinkya
"""

import time, sys

import numpy as np
from scipy.optimize import *
from optimize import snopta, SNOPT_options
from copy import deepcopy
from numpy import linalg as la

from py_utils import *

class trajectoryOptimization:
    def __init__ (self, nState, nSegments, nInput, nOutput, nModel, Q, R, Q_f, labda, sysDynamics):
        # Planning Parameters
        self.nState = nState
        self.nInput = nInput
        self.nOutput = nOutput
        self.nSegments = nSegments
        self.nModel = nModel
        self.len_s = self.nState*(self.nState+1)/2
        self.id1 = np.triu_indices(self.nState)
        self.goal = None
        self.delta = 0.1
        self.sysDynamics = sysDynamics
        self.t_res = 1.0

        # Cost Matrices
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.labda = labda

        # Logging
        self.do_verbose = True


    def objectiveFunction(self, X):
        s = X[(self.nState + self.len_s)*self.nSegments - self.len_s:(self.nState + self.len_s)*self.nSegments]
        J = 0.

        for i in range(self.nSegments):
            m = X[i*self.nState:(i*self.nState+self.nState)]
            u = X[i*self.nInput + (self.nState + self.len_s)*self.nSegments: (i+1)*self.nInput + (self.nState + self.len_s)*self.nSegments]
            J = J + (m-self.goal).dot(self.Q).dot(m-self.goal) + u.dot(self.R.dot(u))

        J = J + s.T.dot(self.labda).dot(s) + (m-self.goal).dot(self.Q_f).dot(m-self.goal)

        return J
    

    def constraints(self, X, X0):        
        u = X[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments]
        u = np.reshape(u,(self.nInput,self.nSegments),'F')
        
        m_new = np.zeros((self.nState,self.nSegments))
        s_new = np.zeros((self.len_s,self.nSegments))
        u_new = X[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments]

        mu_old = X[:self.nState]
        s_old = X[self.nState*self.nSegments:self.nState*self.nSegments + self.len_s]

        # cov_old = symarray(np.zeros((self.nState, self.nState)))
        # cov_old[self.id1] = s_old*1.
        
        U = np.zeros((self.nState, self.nState))
        U[self.id1] = deepcopy(s_old)
        L = U.T
        cov_old = deepcopy(np.dot(L, L.T.conj()))
        if not is_pos_def(cov_old):
            cov_old = deepcopy(nearestPD(cov_old))

        m_new[:,0] = deepcopy(mu_old)
        s_new[:,0] = deepcopy(s_old)

        mu = np.ndarray(self.nState)
        cov = np.ndarray((self.nState, self.nState))
        wts = np.ndarray(self.nModel)
        ds = 0

        for i in range(self.nSegments-1):
            for t in range(self.delta):
                # Incuding numerical integration for belief evolution
                for t_int in range(int(1/self.t_res)):
                    ds_bar = self.sysDynamics.beliefUpdatePlanning(mu_old, cov_old, u[:,i]*self.t_res, mu, cov, wts, ds)
                    mu_old = deepcopy(mu)
                    cov_old = deepcopy(cov)

            m_new[:,i+1] = deepcopy(mu)
            s_new[:,i+1] = deepcopy(cov[self.id1])

        m_new = np.reshape(m_new,(self.nState*self.nSegments),'F')
        s_new = np.reshape(s_new,(self.len_s*self.nSegments),'F')

        X_new = np.reshape(np.concatenate((m_new,s_new,u_new),axis=0),len(X),'F')
        return X_new


    def snopt_objFun(self, status, X, needF, F, needG, G):
        obj_f = self.objectiveFunction(X) # Objective Row
        # obj_f = 0.

        cons_f = deepcopy(self.constraints(X, self.X0))

        F[0] = deepcopy(obj_f)
        F[1:] = deepcopy(X)

        cons_f -= X

        # Setting Dynamics for constraints
        F[self.nState+1:self.nState*self.nSegments+1] = cons_f[self.nState:self.nState*self.nSegments]
        F[self.nState*self.nSegments + self.len_s+1:(self.nState + self.len_s + self.nInput)*self.nSegments+1] = cons_f[self.nState*self.nSegments + self.len_s:(self.nState + self.len_s + self.nInput)*self.nSegments]

        # print "Objective Function = ", F

        return status, F#, G


    def cs_optimize(self, muInit, covInit, wtsInit, tFinal, goal):
        self.delta = int(np.ceil(tFinal/self.nSegments))
        self.goal = goal*1.

        # print "\n******************************************************"
        # print "Inputs:\n", "mu = ", muInit, "\ncovInit = ", covInit, "\nwtsInit = ", wtsInit, "\ngoal = ", goal
        # print "\n******************************************************"


        ##### Setting up to use SNOPT ###
        inf   = 1.0e20
        options = SNOPT_options()

        options.setOption('Verbose',False)
        options.setOption('Solution print',False)
        options.setOption('Print filename','ds_goal_snopt.out')
        options.setOption('Print level',0)

        options.setOption('Optimality tolerance', 1e-2)

        options.setOption('Summary frequency',1)
        options.setOption('Major print level',0)
        options.setOption('Minor print level',0)


        X0      = np.random.rand((self.nState+ self.len_s + self.nInput)*self.nSegments)*100.
        X0[:self.nState] = muInit*1.
        # X0[self.nState*self.nSegments:self.nState*self.nSegments + self.len_s] = covInit[self.id1]*1.


        '''
        Do cholesky Factorization on the input matrix
        '''
        if not isPD(covInit):
            covInit = deepcopy(nearestPD(covInit))

        L = la.cholesky(covInit).T

        X0[self.nState*self.nSegments:self.nState*self.nSegments + self.len_s] = np.reshape(L[self.id1] ,(self.len_s,),'F')


        self.X0 = deepcopy(X0)

        Xlow    = np.array([-150.0]*len(X0[:self.nState*self.nSegments]) + [0.]*len(X0[self.nState*self.nSegments:(self.nState+self.len_s)*self.nSegments]) + [-100.0]*len(X0[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments]))

        Xupp    = np.array([150.0]*len(X0[:self.nState*self.nSegments]) + [1500.]*len(X0[self.nState*self.nSegments:(self.nState+self.len_s)*self.nSegments]) + [100.0]*len(X0[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments]))

        n       = len(X0)
        nF      = int(1 + len(X0))

        F_init = [0.]*nF
        Fstate_init = [0]*nF
        constraintRelax = 0.0

        ## Setting the initial values of mu, cov and wts as boundary constraints
        Flow    = np.array([0.] + muInit.tolist() + [-constraintRelax]*len(X0[self.nState:self.nState*self.nSegments]) + covInit[self.id1].tolist() + [-constraintRelax]*len(X0[self.nState*self.nSegments + self.len_s:(self.nState + self.len_s + self.nInput)*self.nSegments]))
        Fupp    = np.array([0.] + muInit.tolist() + [constraintRelax]*len(X0[self.nState:self.nState*self.nSegments]) + covInit[self.id1].tolist() + [constraintRelax]*len(X0[self.nState*self.nSegments + self.len_s:(self.nState+self.len_s + self.nInput)*self.nSegments]))

        ObjRow  = 1

        Start   = 0 # Cold Start
        cw      = [None]*5000
        iw      = [None]*5000
        rw      = [None]*5000


        res = snopta(self.snopt_objFun,n,nF,x0=X0, xlow=Xlow,xupp=Xupp, Flow=Flow,Fupp=Fupp, ObjRow=ObjRow, F=F_init, Fstate=Fstate_init, name='ds_goal', start=Start, options=options)

        if res == None:
            raise ValueError("SNOPT FAILED TO OPTIMIZE!")

        print "SNOPT Result =", np.round(res.x, 4)

        xfinal = res.x

        mu_new = np.reshape(xfinal[:self.nState*self.nSegments],(self.nState,self.nSegments),'F')
        s_new = np.reshape(xfinal[self.nState*self.nSegments:(self.nState + self.len_s)*self.nSegments], (self.len_s, self.nSegments),'F')
        u_new = np.reshape(xfinal[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments],(self.nInput,self.nSegments),'F')

        final_wts = np.ndarray(self.nModel)
        covFinal = deepcopy(covInit)
        covFinal[self.id1] = s_new[:,-1]
        self.sysDynamics.fastWtsMapped(mu_new[:,-1], covFinal, final_wts)

        # if self.do_verbose:
        print '*****************\nSet Goal: ', goal
        print 'Plan Time Horizon: ', tFinal
        print 'Planning for segments: ', self.nSegments
        print 'Each Segment Length: ', self.delta
        print "Generated Plan: \n", np.round(mu_new.T, 3) #[-1,:]
        print "s_new: ", np.round(s_new.T, 3)
        print "u_new: ", np.round(u_new.T, 3)
        print "final_wts: ", final_wts
        print "Final Cost = ", res.F[0]
        print "********************\n"

        return res.F[0], mu_new, s_new, u_new, final_wts
