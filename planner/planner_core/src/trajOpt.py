#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:55:59 2017

@author: Ajinkya
"""

import time, sys

import numpy as np
from scipy.optimize import *

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
        # self.goal = goal
        self.goal_wts = None
        self.sysDynamics = sysDynamics

        # Cost Matrices
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.labda = labda
        self.alpha_KL = 500.

    def objectiveFunction_w_ds_goal(self, X): 
        '''
        Expected Input : X, trajectory of [mu, s, u, wts] 
        '''
        s = X[(self.nState + self.len_s)*self.nSegments - self.len_s:(self.nState + self.len_s)*self.nSegments]
        J = 0.
        # print "goal_wts =", self.goal_wts


        for i in range(self.nSegments):
            # m = X[i*self.nState:(i*self.nState+self.nState)]
            # u = X[i*self.nInput + (self.nState + self.len_s)*self.nSegments: (i+1)*self.nInput + (self.nState + self.len_s)*self.nSegments]

            wts = X[i*self.nModel + (self.nState + self.len_s + self.nInput)*self.nSegments: (i+1)*self.nModel + (self.nState + self.len_s + self.nInput)*self.nSegments]
            
            if sum(wts) > 1e6:
                wts /= sum(wts)
                
            # print "wts = ", wts
            # print "KL div =", self.alpha_KL*sym_KL(self.goal_wts, wts)
            J = J + self.alpha_KL*sym_KL(self.goal_wts, wts)
            # print "Objective Function Value = ", J, "#############"


        # J = J + s.T.dot(self.labda).dot(s) + (m-self.goal).dot(self.Q_f).dot(m-self.goal)
        # raw_input()
        return J

    def objectiveFunction(self, X):
        s = X[(self.nState + self.len_s)*self.nSegments - self.len_s:(self.nState + self.len_s)*self.nSegments]
        J= 0.

        for i in range(self.nSegments):
            m = X[i*self.nState:(i*self.nState+self.nState)]
            u = X[i*self.nInput + (self.nState + self.len_s)*self.nSegments: (i+1)*self.nInput + (self.nState + self.len_s)*self.nSegments]
            J = J + (m-self.goal).dot(self.Q).dot(m-self.goal) + u.dot(self.R.dot(u))

        J = J + s.T.dot(self.labda).dot(s) + (m-self.goal).dot(self.Q_f).dot(m-self.goal)

        # if hasattr(J, "__len__"):
        #     print "nonScalar objective function =", J
        #     print "obstacle cost =", self.obstacle_cost(m)
        #     sys.exit(1)
        return J
    

    def gradient_obj_fn(self, X):
        grad_J = X*1       
        for i in range(self.nSegments):
            m = X[i*self.nState:(i*self.nState+self.nState)]
            u = X[i*self.nInput + (self.nState + 1)*self.nSegments: (i+1)*self.nInput + (self.nState + 1)*self.nSegments]
            grad_J[i*self.nState:(i*self.nState+self.nState)] = self.Q.dot(m-self.goal) 
            grad_J[i*self.nInput + (self.nState + 1)*self.nSegments: (i+1)*self.nInput + (self.nState + 1)*self.nSegments] = self.R.dot(u)

        s = X[(self.nState + self.len_s)*self.nSegments - self.len_s:(self.nState + self.len_s)*self.nSegments]
        grad_J[(self.nState + self.len_s)*self.nSegments - self.len_s:(self.nState + self.len_s)*self.nSegments] = self.labda.dot(s)
        return grad_J


    def constraints(self, X, X0, delta, constraintRelax):        
        u = X[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments]
        u = np.reshape(u,(self.nInput,self.nSegments),'F')
        
        m_new = np.zeros((self.nState,self.nSegments))
        s_new = np.zeros((self.len_s,self.nSegments))
        u_new = X[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments]
        wts_new = (1/self.nModel)*np.ones((self.nModel, self.nSegments))

        mu = X0[:self.nState]*1
        s = X0[self.nState*self.nSegments:self.nState*self.nSegments + self.len_s]*1
        cov = symarray(np.zeros((self.nState, self.nState)))
        cov[self.id1] = s
        wts = X0[(self.nState+self.len_s + self.nInput)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments + self.nModel]*1
        
        m_new[:,0] = mu*1
        s_new[:,0] = s*1
        wts_new[:,0] = wts*1.

        ds_old = 0

        for i in range(self.nSegments-1):
            for t in range(delta):
                ds_bar = self.sysDynamics.beliefUpdatePlanning(mu, cov, u[:,i], mu, cov, wts, ds_old)


            m_new[:,i+1] = mu*1
            s_new[:,i+1] = cov[self.id1]

            # Normalize wts
            wts /= sum(wts)
            wts_new[:,i+1] = wts*1.
        
        # print "the pivot points from constraints =", m_new
        # Converting the new mu and cov back to the original Shapes
        # print "wts_new in constraints = \n", wts_new.T      

        m_new = np.reshape(m_new,(self.nState*self.nSegments),'F')        
        s_new = np.reshape(s_new,(self.len_s*self.nSegments),'F')        
        wts_new = np.reshape(wts_new, (self.nModel*self.nSegments), 'F')   

        X_new = np.reshape(np.concatenate((m_new,s_new,u_new, wts_new),axis=0),len(X),'F') 
        return X_new


    def createPlan(self, muInit, covInit, wtsInit, tFinal):
        delta = int(np.round(tFinal/self.nSegments))        
        X0 = np.random.rand((self.nState+ self.len_s + self.nInput + self.nModel)*self.nSegments)
        # X0 = np.zeros((self.nState+self.len_s + self.nInput)*self.nSegments)
        # mu0 = np.reshape(multilinspace(muInit, self.goal, num=self.nSegments, endpoint=True), self.nSegments*self.nState)
        # mu0 = warmStart('data/run_warm_start.npz', self.nSegments) # asssuming warm start from a deterministic solution
        # X0[:self.nState*self.nSegments] = mu0*1.

        # mu0 = initialGuess(muInit, self.goal, nPts=self.nSegments)
        # print "Intial Guess =", mu0
        # X0[:self.nState*self.nSegments] = mu0*1

        X0[:self.nState] = muInit*1
        X0[self.nState*self.nSegments:self.nState*self.nSegments + self.len_s] = covInit[self.id1]*1.
        X0[(self.nState+self.len_s + self.nInput)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments + self.nModel] = (1./self.nModel)*np.ones(self.nModel)
        
        bnds = [(-10., 10.) for i in range(len(X0[:self.nState*self.nSegments]))] + [(0., 10.) for i in range(len(X0[self.nState*self.nSegments:(self.nState+self.len_s)*self.nSegments]))] + [(-1.,1.) for i in range(len(X0[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments]))] + [(0., 1.) for i in range(len(X0[(self.nState+self.len_s + self.nInput)*self.nSegments:]))]

        constraintRelax = 0.  
        
        #optimization options
        ops = ({'maxiter': 1e6 , 'disp': False})

        start_time = time.time()
  
        while(1):
            ## Setting up the constraints
            cons = ({'type': 'ineq', 'fun': lambda X: X - self.constraints(X, X0, delta, constraintRelax) + constraintRelax*np.ones(len(X))},
                    {'type': 'ineq', 'fun': lambda X: self.constraints(X, X0, delta, constraintRelax) - X + constraintRelax*np.ones(len(X))})
            #cons = ({'type': 'eq', 'fun': lambda X: X - self.constraints(X, X0, delta, constraintRelax, self.sysDynamics)})    
            
            res = minimize(lambda X: self.objectiveFunction_w_ds_goal(X), X0, method='SLSQP', bounds=bnds, options = ops, constraints = cons)


            xfinal = res.x
            print "Optimization status: ", res.status
            print "Optimization termination message :", res.message
            # print "xfinal from optimization= ", xfinal
            
            # Handling the error criteia in the optimization routine
            if(res.status == 4):
                constraintRelax = constraintRelax + 0.025
                # print "constraint violation by =", res.maxcv
                print "New constraintRelax = ", constraintRelax
            elif(res.status == 8 or res.status == 3):
                X0[self.nState:] = xfinal[self.nState:] + np.random.rand(len(X0[self.nState:]))
                print "Value of the function = ", res.fun
            else:
                break
        
        mu_new = np.reshape(xfinal[:self.nState*self.nSegments],(self.nState,self.nSegments),'F')
        s_new = np.reshape(xfinal[self.nState*self.nSegments:(self.nState + self.len_s)*self.nSegments], (self.len_s, self.nSegments),'F')
        u_new = np.reshape(xfinal[(self.nState+self.len_s)*self.nSegments:(self.nState+self.len_s + self.nInput)*self.nSegments],(self.nInput,self.nSegments),'F')
        wts_new = np.reshape(xfinal[(self.nState+self.len_s+self.nInput)*self.nSegments:],(self.nModel,self.nSegments),'F')
        
        # print np.round(wts_new.T, 3)

        mu_plan = np.zeros((self.nState,int(self.nSegments*delta)+1))
        s_plan = np.zeros((self.len_s, int(self.nSegments*delta)+1))
        u_plan = np.zeros((self.nInput,int(self.nSegments*delta)))
        wts_plan = np.zeros((self.nModel, int(self.nSegments*delta)))

        
        mu_plan[:,0] = mu_new[:,0]*1
        s_plan[:,0] = s_new[:, 0]*1
        cov = symarray(np.zeros((self.nState, self.nState)))
        cov[self.id1] = s_plan[:,0]*1.
        wts_plan[:,0] = wts_new[:,0]*1.

        print "Time spent in optimization call = %s seconds" % (time.time() - start_time)
        
        # Restart the weights for the whole plan
        self.sysDynamics.wts = wtsInit*1.
        active_idx = []

        print "Intial wts in dyna for final plan = ", self.sysDynamics.wts

        print "\nInput mu for planner = ", np.round(mu_plan[:,0], 3)
        # raise NameError('Debug')
        ds_old = 0

        for i in range(self.nSegments-1):
            for t in range(delta):
                w_new = wts_plan[:,i*delta+t]*1.
                mu = mu_plan[:,i*delta+t]*1.
                mu_new = mu_plan[:,i*delta+t+1]*1.
                ds_new = self.sysDynamics.beliefUpdatePlanning(mu, cov, u_new[:,i], mu_new, cov, w_new, ds_old)

                wts_plan[:,i*delta+t+1] = w_new*1.
                mu_plan[:,i*delta+t+1] = mu_new*1.

                s_plan[:,i*delta+t+1] = cov[self.id1]*1.
                u_plan[:,i*delta+t] = u_new[:,i]*1.
                active_idx.append(ds_new)
                
                print "\n Planned Step", i*delta+t+1
                print "Planned mu = ", np.round(mu_plan[:,i*delta+t+1], 3)
                print "Planned s = ", np.round(s_plan[:,i*delta+t+1], 4)
                print "Planned Control, u_plan = ", np.round(u_plan[:,i*delta+t], 2)
                print "wts planned =", np.round(wts_plan[:,i*delta+t+1], 3)

                # print "Obstacle Cost = ", self.delta*self.obstacle_cost(mu_plan[:,i*delta+t+1])         
               
        return mu_plan, s_plan, u_plan, wts_plan, active_idx


########## Stabilization       

class blqr:
    def __init__(self, nState, nInput, nOutput, Q, R, labda, Q_f, W):
        self.Q = Q
        self.R = R
        self.nState = nState
        self.nInput = nInput
        self.nOutput = nOutput 
        self.len_s = self.nState*(self.nState+1)/2
        self.id1 = np.triu_indices(self.nState)
        self.labda = labda
        self.Q_f = Q_f
        self.W = W

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


    def blqr(self, mu, sigma, mu_ref, sig_ref, u_ref, A, B, C, tFinal):
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
        
    
        