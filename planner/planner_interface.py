#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-

import copy
from pomdp_hd_planner import *
from planner_core.src.blqr import *

class planner_interface:
    def __init__(self, x, mu, goal):
        # Class initializations
        self.planner = pomdp_hd()
        self.controller = blqr(self.planner.nState, self.planner.nInput, self.planner.nOutput, self.planner.opt.Q, self.planner.opt.R, self.planner.opt.labda, self.planner.opt.Q_f)

        # Set Parmeters
        self.planner.do_parallelize = True
        self.planner.do_verbose = False

        # Number fo Samples to calculate discerte belief from continous beleif
        self.planner.dyna.nSamples = 50. 
        
        # Change the loop count in beleif propagation int value
        self.planner.dyna.ds_res_loop_count_ = 100 

        # Set Values
        self.id1 = self.planner.opt.id1
        self.x_actual = copy.copy(x)
        self.mu_actual = copy.copy(mu)
        self.cov_actual = (np.linalg.norm(x-mu)**2 + 25.)*np.eye(self.planner.nState)
        self.s_actual = copy.copy(self.cov_actual[self.id1])
        self.wts_actual = np.ndarray(self.planner.nModel)
        self.goal = copy.copy(goal)

        self.planner.start_mu = copy.copy(self.mu_actual)
        self.planner.start_cov = copy.copy(self.cov_actual)
        self.planner.belief = hybrid_belief(self.mu_actual, self.cov_actual, self.wts_actual)
        self.planner.goal_cs = copy.copy(self.goal)
        self.planner.dyna.setGC(copy.copy(self.goal)) # Setting GC

        # Matrices for holding dynamics
        self.A = np.ndarray((self.planner.nState, self.planner.nState))
        self.B = np.ndarray((self.planner.nInput, self.planner.nInput))
        self.C = np.ndarray((self.planner.nOutput, self.planner.nOutput))
        self.V =  np.ndarray((self.planner.nState, self.planner.nState))
        self.W = np.ndarray((self.planner.nOutput, self.planner.nOutput))
        self.idx = 0
        self.idx_old = 0
        self.cov_plan = symarray(np.zeros((self.planner.nState, self.planner.nState)))
        self.xNew = np.ndarray(self.planner.nState)
        self.z = np.ndarray(self.planner.nState)
        self.muNew = np.ndarray(self.planner.nState)
        self.covNew = np.ndarray((self.planner.nState, self.planner.nState))
        self.wtsNew = np.ndarray(self.planner.nModel)

        # Data Storage
        self.traj = [copy.copy(self.mu_actual)]
        self.traj_true = [copy.copy(self.x_actual)]
        self.ds_traj = [copy.copy(self.wts_actual)]
        self.cov_traj = [copy.copy(self.s_actual)]

        #### Checking if everything is correctly set
        print "start_mu: ", self.planner.start_mu, "\nstart_cov:", self.planner.start_cov, "\ngoal: ", self.goal
        # raw_input('Press Enter if everything looks okay!')


    def generate_plan(self):
        start_time = time.time()
        mu_plan, s_plan, u_plan = self.planner.plan_optimal_path()
        self.planner.first_pass = False

        tActual = np.round((time.time() - start_time), 3)
        print("Total time required by Planner = %s seconds "  %tActual)
        return mu_plan, s_plan, u_plan

    def execute_plan_oneStep(self, mu_plan, s_plan, u_plan):
        self.cov_plan[self.id1] = copy.copy(s_plan)

        #LQR Control
        self.planner.dyna.getMatrices(self.idx, self.A, self.B, self.C, self.V, self.W)

        u_local = self.controller.blqr(self.mu_actual, self.s_actual, mu_plan, s_plan, u_plan, self.A, self.B, self.C, self.W, 5)

        # Propagation of the Belief continuous Dynamics
        self.idx = self.planner.dyna.predictionStochastic(self.mu_actual, self.cov_actual, u_local, self.muNew, self.covNew, self.wtsNew, self.idx)

        # Next point on the Trajectory
        z_dummy = np.ndarray(self.planner.nState)
        self.planner.dyna.simulate_oneStep(self.x_actual, u_local, self.xNew, z_dummy)

        return self.xNew, z_dummy

    def update_belief(self, z):
        #Updates based on the observation
        self.idx = self.planner.dyna.observationUpdate(z, self.muNew, self.covNew, self.wtsNew, self.idx)
        return

    def update_stored_values(self):
        # Update belief Values
        self.mu_actual = copy.copy(self.muNew.T)
        self.planner.start_mu = copy.copy(self.mu_actual)
        self.x_actual = copy.copy(self.xNew)
        self.cov_actual = copy.copy(self.covNew)
        self.planner.start_cov = copy.copy(self.cov_actual)
        self.s_actual = copy.copy(self.cov_actual[self.id1])
        self.wts_actual = copy.copy(self.wtsNew)

    def trigger_restart(self, mu=np.random.rand(3) ,s=2.0):
        self.mu = mu
        # self.s = s
        self.s[0] = s
        self.s[1] = 0.
        self.s[2] = 0.
        self.s[3] = s
        self.s[4] = 0.
        self.s[5] = s
        self.cov[self.id1] = self.s
        print "##############\nTriggering belief restart with \n", "Mean =", self.mu, "\nCovariance =", self.cov, "\n###############"
        return

    def observation_triggered_restart(self, observation_array):
        # Force restart if the covariance on continuoues states is very low but observations suggest mean is too diverged
        obs_mean = np.mean(observation_array,axis=0)

        if max(self.s) < 0.01:
            if distance.euclidean(obs_mean, self.mu) > 0.01:
                self.trigger_restart(mu=obs_mean)

        return


    def plotter():
        import matplotlib.pyplot as plt
        fig = plt.figure
        x_vec = [traj[i][0] for i in range(len(traj))]
        y_vec = [traj[i][1] for i in range(len(traj))]
        x_vec_true = [traj_true[i][0] for i in range(len(traj_true))]
        y_vec_true = [traj_true[i][1] for i in range(len(traj_true))]
        plt.plot(x_vec,y_vec, 'r--')
        plt.plot(x_vec_true,y_vec_true, 'b',linewidth = 3.0)

        axes = plt.gca()
        # axes.set_xlim([xmin,xmax])
        axes.set_ylim([-2.0, 3.0])
        plt.show()
    
