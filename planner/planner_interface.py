#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from pomdp_hd_planner import *
from planner_core.src.blqr import *


class planner_interface:
    """
    Class providing a high-level interface to interact with POMDP-HD planner 
    """

    def __init__(self, x, mu, goal):
        ''' Object initializations '''
        self.planner = pomdp_hd()
        self.controller = blqr(self.planner.nState, self.planner.nInput,
                               self.planner.nOutput, self.planner.opt.Q,
                               self.planner.opt.R, self.planner.opt.labda,
                               self.planner.opt.Q_f)

        ''' Set Planner Parmeters'''
        self.planner.do_parallelize = True
        self.planner.do_verbose = False
        # Number fo Samples to calculate discerte belief from continous beleif
        self.planner.dyna.nSamples = 50.
        # Change the loop count in beleif propagation int value
        self.planner.dyna.ds_res_loop_count_ = 1

        ''' Initialization '''
        self.id1 = self.planner.opt.id1
        self.x_actual = copy.copy(x)
        self.mu_actual = copy.copy(mu)
        self.cov_actual = (np.linalg.norm(x-mu)**2 + 25.) * \
            np.eye(self.planner.nState)
        self.s_actual = copy.copy(self.cov_actual[self.id1])
        self.wts_actual = np.ndarray(self.planner.nModel)
        self.goal = copy.copy(goal)

        self.planner.start_mu = copy.copy(self.mu_actual)
        self.planner.start_cov = copy.copy(self.cov_actual)
        self.planner.belief = hybrid_belief(
            self.mu_actual, self.cov_actual, self.wts_actual)
        self.planner.goal_cs = copy.copy(self.goal)
        self.planner.dyna.setGoalGC(copy.copy(self.goal))  # Setting GC


        # Matrices for holding dynamics
        self.A = np.ndarray((self.planner.nState, self.planner.nState))
        self.B = np.ndarray((self.planner.nInput, self.planner.nInput))
        self.C = np.ndarray((self.planner.nOutput, self.planner.nOutput))
        self.V = np.ndarray((self.planner.nState, self.planner.nState))
        self.W = np.ndarray((self.planner.nOutput, self.planner.nOutput))
        self.idx = 0
        self.idx_old = 0
        self.cov_plan = symarray(
            np.zeros((self.planner.nState, self.planner.nState)))
        self.xNew = np.ndarray(self.planner.nState)
        self.z = np.ndarray(self.planner.nState)
        self.muNew = np.ndarray(self.planner.nState)
        self.covNew = np.ndarray((self.planner.nState, self.planner.nState))
        self.wtsNew = np.ndarray(self.planner.nModel)

        ''' Data Storage '''
        self.traj = [copy.copy(self.mu_actual)]
        self.traj_true = [copy.copy(self.x_actual)]
        self.ds_traj = [copy.copy(self.wts_actual)]
        self.cov_traj = [copy.copy(self.s_actual)]

        # Checking if everything is correctly set
        print "start_mu: ", self.planner.start_mu
        print "start_cov:", self.planner.start_cov, "\ngoal: ", self.goal

    def generate_plan(self):
        """
        Function to generate plan based on set continuous space target

        Returns
        -------
            mu_plan : planned belief mean trajectory
            s_plan  : Covariance vector based on planned trajectory
            u_plan  : Planned control to be applied 
        """
        start_time = time.time()
        mu_plan, s_plan, u_plan = self.planner.plan_optimal_path()
        self.planner.first_pass = False

        tActual = np.round((time.time() - start_time), 3)
        print("Total time required by Planner = %s seconds " % tActual)
        return mu_plan, s_plan, u_plan

    def execute_plan_oneStep(self, mu_plan, s_plan, u_plan):
        """
        Function to execute plan one step based on applied controls.
        First it calculates the local belief space control (B-LQR) 
        that it should apply based on the current belief and covariance.
        Executes one step using simulator and returns new robot state 
        and observation

        Parameters
        -------
            mu_plan : Planned next step belief mean
            s_plan  : Planned next step belief covariance
            u_plan  : Planned control input

    
        Returns
        -------
            xNew    : New robot state from simulator
            z       : Observation from simulator

        """
        self.cov_plan[self.id1] = copy.copy(s_plan)

        # B-LQR Control
        self.planner.dyna.getMatrices(
            self.idx, self.A, self.B, self.C, self.V, self.W)

        u_local = self.controller.blqr(
            self.mu_actual, self.s_actual, mu_plan,
            s_plan, u_plan, self.A, self.B, self.C, self.W, 5)

        # Propagation of the Belief continuous Dynamics
        self.idx = self.planner.dyna.predictionStochastic(
            self.mu_actual, self.cov_actual, u_local,
            self.muNew, self.covNew, self.wtsNew, self.idx)

        # Next point on the Trajectory
        z = np.ndarray(self.planner.nState)

        self.planner.dyna.simulateOneStep(
            self.x_actual, u_local, self.xNew, z)
        print "Simulated_onestep"
        return self.xNew, z

    def update_belief(self, z):
        """
        Update belief based on received observation

        Parameters
        -------
            z       : Recievd Observation
    
        """
        # Updates based on the observation
        self.idx = self.planner.dyna.observationUpdate(
            z, self.muNew, self.covNew, self.wtsNew, self.idx)
        return

    def update_stored_values(self):
        """
        Updates values in the planner
        """
        # Update belief Values
        self.mu_actual = copy.copy(self.muNew.T)
        self.planner.start_mu = copy.copy(self.mu_actual)
        self.x_actual = copy.copy(self.xNew)
        self.cov_actual = copy.copy(self.covNew)
        self.planner.start_cov = copy.copy(self.cov_actual)
        self.s_actual = copy.copy(self.cov_actual[self.id1])
        self.wts_actual = copy.copy(self.wtsNew)

    def trigger_restart(self, mu=np.random.rand(3), s=2.0):
        """
        Trigger Random restart of belief in case of too much divergence.
        Updates internal values

        Parameters
        -------
            mu      : Belief mean to start with
            s       : Belief covariance to start with

        """
        self.mu = mu
        # self.s = s
        self.s[0] = s
        self.s[1] = 0.
        self.s[2] = 0.
        self.s[3] = s
        self.s[4] = 0.
        self.s[5] = s
        self.cov[self.id1] = self.s
        print "############################"
        print "Triggering belief restart with "
        print "Mean =", self.mu, "\nCovariance =", self.cov
        print "############################"
        return

    def observation_triggered_restart(self, observation_array):
        """
        Force restart if the covariance on continuoues states
        is very low but observations suggest mean is too diverged

        Parameters
        -------
            observation_array  : An array of observations from last 
                                 'n' steps

        """
        obs_mean = np.mean(observation_array, axis=0)

        if max(self.s) < 0.01:
            if distance.euclidean(obs_mean, self.mu) > 0.01:
                self.trigger_restart(mu=obs_mean)

        return
