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
import copy
from numpy import linalg as la

from py_utils import *

class trajectoryOptimization:
    MAX_TRIES = 1

    def __init__(self, nState, nInput, nOutput, nModel, Q, R, Q_f, labda, sysDynamics):
        # Planning Parameters
        self.nState = nState
        self.nInput = nInput
        self.nOutput = nOutput
        self.nModel = nModel
        self.len_s = int(self.nState * (self.nState + 1) / 2)
        self.id1 = np.triu_indices(self.nState)
        self.goal = None
        self.sysDynamics = sysDynamics
        self.extra_euler_res = 1
        self.time_horizon = 10

        self.nSegments = 3

        # Cost Matrices
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.labda = labda

        # Logging
        self.do_verbose = True

    def objectiveFunction(self, X):
        s = X[(self.nState + self.len_s) * (self.nSegments+1) - self.len_s:(self.nState + self.len_s) * (self.nSegments+1)]
        J = 0.

        for i in range(self.nSegments-1):
            m = X[i * self.nState:(i+1) * self.nState]
            u = X[i * self.nInput + (self.nState + self.len_s) * (self.nSegments+1): (i + 1) * self.nInput + (
                    self.nState + self.len_s) * (self.nSegments+1)]
            J = J + (m - self.goal).dot(self.Q).dot(m - self.goal) + u.dot(self.R.dot(u))
            # J = J + (m[:3]-self.goal[:3]).dot(self.Q[:3, :3]).dot(m[:3]-self.goal[:3]) + u.dot(self.R.dot(u))

        m_T = X[self.nSegments* self.nState:(self.nSegments+1) * self.nState]

        J = J + s.T.dot(self.labda).dot(s) + (m_T - self.goal).dot(self.Q_f).dot(m_T - self.goal)
        # J = J + s.T.dot(self.labda).dot(s) + (m_T[:3]-self.goal[:3]).dot(self.Q[:3, :3]).dot(m_T[:3]-self.goal[:3])

        return J

    def constraints(self, X):  
        X_constrained = copy.copy(X)
        X_constrained[:self.nState] = copy.copy(self.X0[:self.nState])
        X_constrained[self.nState*(self.nSegments+1):self.nState*(self.nSegments+1) + self.len_s] = copy.copy(self.X0[self.nState*(self.nSegments+1):self.nState*(self.nSegments+1) + self.len_s])

        mu = copy.copy(X_constrained[:self.nState])
        s = copy.copy(X_constrained[self.nState*(self.nSegments+1):self.nState*(self.nSegments+1) + self.len_s])
        cov = symarray(np.zeros((self.nState, self.nState)))
        cov[self.id1] = s

        u = copy.copy(X[(self.nState+self.len_s)*(self.nSegments+1):])
        u = np.reshape(u,(self.nInput,self.nSegments),'F')        
        
        wts = np.ndarray(self.nModel)
        ds = 0

        for i in range(self.nSegments):
            for t in range(self.delta):
                # Incuding numerical integration for belief evolution
                for t_int in range(int(self.extra_euler_res)):
                    ds_bar = self.sysDynamics.beliefUpdatePlanning(mu, cov, u[:,i]/self.extra_euler_res, mu, cov, wts, ds)

            X_constrained[(i+1)*self.nState:(i+2)*self.nState] = copy.copy(mu)
            X_constrained[self.nState*(self.nSegments+1) + (i+1)*self.len_s : self.nState*(self.nSegments+1) + (i+2)*self.len_s] = copy.copy(cov[self.id1])
            
            if i < self.nSegments-1:  #Skipping last control input
                X_constrained[(self.nState+self.len_s)*(self.nSegments+1) + (i+1)*self.nInput:(self.nState+self.len_s)*(self.nSegments+1) + (i+2)*self.nInput] = copy.copy(u[:,i])

        return X_constrained


    def snopt_objFun(self, status, X, needF, F, needG, G):
        F[0] = self.objectiveFunction(X)  # Objective Row

        cons_f = copy.copy(self.constraints(X))
        F[1:] = copy.copy(X - cons_f)  # less than equal to constraints
        return status, F

    def generate_X0(self, muInit, covInit, goal):
        mu = copy.copy(muInit)
        cov = copy.copy(covInit)
        u = copy.copy((goal - mu) / (self.time_horizon - 1))
        wts = np.ndarray(self.nModel)
        ds = 0

        X0 = np.zeros((self.nState + self.len_s) * (self.nSegments+1) + self.nInput*self.nSegments)
        X0[:self.nState] = copy.copy(mu)
        X0[self.nState*(self.nSegments+1):self.nState*(self.nSegments+1) + self.len_s] = copy.copy(cov[self.id1])
        X0[(self.nState + self.len_s)*(self.nSegments+1):(self.nState + self.len_s)*(self.nSegments+1) + self.nInput] = copy.copy(u)
        
        for i in range(self.nSegments):
            for t in range(self.delta):
                # Incuding numerical integration for belief evolution
                for t_int in range(int(self.extra_euler_res)):
                    ds = self.sysDynamics.beliefUpdatePlanning(mu, cov, u/self.extra_euler_res, mu, cov, wts, ds)

            X0[(i+1)*self.nState:(i+2)*self.nState] = copy.copy(mu)
            X0[self.nState*(self.nSegments+1) + (i+1)*self.len_s : self.nState*(self.nSegments+1) + (i+2)*self.len_s] = copy.copy(cov[self.id1])
            
            if i < self.nSegments-1:  #Skipping last control input
                X0[(self.nState+self.len_s)*(self.nSegments+1) + (i+1)*self.nInput:(self.nState+self.len_s)*(self.nSegments+1) + (i+2)*self.nInput] = copy.copy(u)
        return X0

    def cs_optimize(self, muInit, covInit, wtsInit, goal):
        self.delta = int(np.ceil(self.time_horizon / self.nSegments))
        # self.sysDynamics.setIntergationStepSize(self.delta)
        self.goal = copy.copy(goal)

        print("\n******************************************************")
        print("Inputs:\n", "mu = ", muInit, "\ncovInit = ", covInit, "\nwtsInit = ", wtsInit, "\ngoal = ", goal)
        print("\n******************************************************")

        ##### Setting up to use SNOPT ###
        inf = 1.0e20
        options = SNOPT_options()

        options.setOption('Verbose', False)
        options.setOption('Solution print', False)
        options.setOption('Print filename', 'ds_goal_snopt.out')
        options.setOption('Print level', 0)

        options.setOption('Optimality tolerance', 1e-3)
        options.setOption('Major optimality', 1e-3)
        options.setOption('Scale option', 2)

        options.setOption('Summary frequency', 1)
        options.setOption('Major print level', 0)
        options.setOption('Minor print level', 0)

        # INITIALIZATION VECTOR
        mu_range = 100.
        s_range = 2000.
        u_range = 50.

        self.X0 = self.generate_X0(muInit, covInit, goal)

        Xlow = np.array([-mu_range] * len(self.X0[:self.nState * (self.nSegments+1)]) +
                        [0.] * len(self.X0[self.nState * (self.nSegments+1):
                                           (self.nState + self.len_s) * (self.nSegments+1)]) +
                        [-u_range] * len(self.X0[(self.nState + self.len_s) * (self.nSegments+1):]))

        Xupp = np.array([mu_range] * len(self.X0[:self.nState * (self.nSegments+1)]) +
                        [s_range] * len(self.X0[self.nState * (self.nSegments+1):
                                                (self.nState + self.len_s) * (self.nSegments+1)]) +
                        [u_range] * len(self.X0[(self.nState + self.len_s) * (self.nSegments+1):]))

        n = len(self.X0)
        nF = int(1 + len(self.X0))

        F_init = [0.] * nF
        Fstate_init = [0] * nF
        constraintRelax = 1e-2

        # Setting the initial values of mu, cov and wts as boundary constraints
        Flow = np.array(
            [0.] + muInit.tolist() + [-constraintRelax] * len(self.X0[self.nState:self.nState * (self.nSegments+1)]) +
            covInit[self.id1].tolist() + [-constraintRelax] * len(self.X0[self.nState * (self.nSegments+1) + self.len_s:]))
        Fupp = np.array(
            [0.] + muInit.tolist() + [constraintRelax] * len(self.X0[self.nState:self.nState * (self.nSegments+1)]) +
            covInit[self.id1].tolist() + [constraintRelax] * len(self.X0[self.nState * (self.nSegments+1) + self.len_s:]))

        ObjRow = 1

        Start = 0  # Cold Start
        cw = [None] * 5000
        iw = [None] * 5000
        rw = [None] * 5000

        for trial in range(self.MAX_TRIES):
            res = snopta(self.snopt_objFun, n, nF, x0=self.X0, xlow=Xlow, xupp=Xupp, Flow=Flow, Fupp=Fupp,
                         ObjRow=ObjRow, F=F_init, Fstate=Fstate_init, name='ds_goal', start=Start, options=options)
            if res is None:
                # import pdb; pdb.set_trace()
                print("Failed to find solution for optimization after ", trial + 1, " tries. Retrying!")
                continue
            elif res.info == 1:
                xfinal = res.x

                mu_new = np.reshape(xfinal[:self.nState * (self.nSegments+1)], (self.nState, (self.nSegments+1)), 'F')
                s_new = np.reshape(xfinal[self.nState * (self.nSegments+1):(self.nState + self.len_s) * (self.nSegments+1)],
                                   (self.len_s, (self.nSegments+1)), 'F')
                u_new = np.reshape(xfinal[(self.nState + self.len_s) * (self.nSegments+1):],
                                   (self.nInput, self.nSegments), 'F')

                final_wts = np.ndarray(self.nModel)
                covFinal = copy.copy(covInit)
                covFinal[self.id1] = s_new[:, -1]
                self.sysDynamics.fastWtsMapped(mu_new[:, -1], covFinal, final_wts)

                if self.do_verbose:
                    print('*****************\nSet Goal: ', goal)
                    print('Plan Time Horizon: ', self.time_horizon)
                    print('Planning for segments: ', self.nSegments)
                    print('Each Segment Length: ', self.delta)
                    print("Generated Plan: \n", np.round(mu_new.T, 3))
                    print("s_new: ", np.round(s_new.T, 3))
                    print("u_new: ", np.round(u_new.T, 2))
                    print("final_wts: ", final_wts)
                    print("Final Cost = ", res.F[0])
                    print("********************\n")

                return res.F[0], mu_new, s_new, u_new, final_wts

        print("\nXXXXXXXXXXXXXXX FAILED TO OPTIMIZE! XXXXXXXXXXXXXXX \nReturning Initial Guess\n")
        # return np.inf, np.tile(muInit, (self.nSegments, 1)).T, np.tile(covInit[self.id1],
        # (self.nSegments, 1)).T, np.tile([0.]*self.nInput, (self.nSegments, 1)).T, wtsInit*1.

        mu_new = np.reshape(self.X0[:self.nState * (self.nSegments+1)], (self.nState, (self.nSegments+1)), 'F')
        s_new = np.reshape(self.X0[self.nState * (self.nSegments+1):(self.nState + self.len_s) * (self.nSegments+1)],
                           (self.len_s, (self.nSegments+1)), 'F')
        u_new = np.reshape(self.X0[(self.nState + self.len_s) * (self.nSegments+1):], (self.nInput, self.nSegments), 'F')

        final_wts = np.ndarray(self.nModel)
        covFinal = copy.copy(covInit)
        covFinal[self.id1] = s_new[:, -1]
        self.sysDynamics.fastWtsMapped(mu_new[:, -1], covFinal, final_wts)

        return self.objectiveFunction(self.X0), mu_new, s_new, u_new, final_wts
