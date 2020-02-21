# -*- coding: utf-8 -*-

import sys, time
import numpy as numpy
import copy
from multiprocessing import Process, Pipe
from contextlib import contextmanager
from itertools import izip, permutations

from planner_core.cython.beliefEvolution import *
from planner_core.src.blqr import *
from planner_core.src.py_utils import *

# Global Optimization
import scipy.interpolate
from scipy.optimize import differential_evolution

USE_SNOPT = True

if USE_SNOPT:
    snopt_call = 0
    from planner_core.src.trajOpt_snopt import *
else:
    from planner_core.src.trajOpt import *
# max_cost_differences = []

''' Parallel Processing '''
@contextmanager # For Making it to work with Python 2.7
def terminating(thing):
    try:
        yield thing
    finally:
        thing.terminate()

# For making it to work on classes
def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]



## Data structure to handle hybrid belief
class hybrid_belief(object):
    def __init__(self, mu, cov, wts):
        self.mu = mu
        self.cov = cov
        self.wts = wts

class data_info(object):
    def __init__(self, rollout, cost, plan, final_wts, cs_goal_path):
        self.rollout = rollout
        self.cost = cost
        self.plan = plan
        self.final_wts = final_wts
        self.cs_goal_path = cs_goal_path


def stichCSPath(planned_path):
    complete_path = planned_path[0]
    # print "complete_path =", complete_path
    # print "shape of planned plan = ", np.shape(planned_path)

    for i, plan in enumerate(planned_path[1:]):
            complete_path[0] = np.append(complete_path[0], plan[0], axis=1)
            complete_path[1] = np.append(complete_path[1], plan[1], axis=1)
            complete_path[2] = np.append(complete_path[2], plan[2], axis=1)

    return complete_path


class pomdp_hd:
    def __init__(self):
        ## PlANNING TIME HORIZON
        self.t = 10

        ## PARAMETERS
        self.do_parallelize = True
        self.do_verbose = False
        self.show_traj = False
        self.first_pass = True

        ## Initalize Dynamics definition  
        self.dyna = BeliefEvolution()

        self.nState = self.dyna.nState
        self.nInput = self.dyna.nInput
        self.nOutput = self.dyna.nOutput
        self.nModel = self.dyna.nModel
        self.domain = [(-50., 100.), (-50.0, 100.0)]

        # Hybrid Belief
        self.start_mu = np.array([2.0, 0.5])
        self.start_cov = 0.5*np.eye(self.nState)
        wts = np.ndarray(self.nModel)
        self.start_ds = 0
        self.goal_cs = np.array([0., 0.])
        self.goal_wts = np.ndarray(self.nModel)
        self.goal_ds = 0
        self.belief = hybrid_belief(self.start_mu, self.start_cov, wts)

        ## Trajectory Optimization
        self.nSegments = 3
        self.opt = self.setTrajectoryOptimization()

        # rbf result
        self.rbf = None
        self.rollouts = []
        self.set_of_ds_goals = []


    def setTrajectoryOptimization(self):
        # Cost Matrices
        Q = 1.0*np.eye(self.nState)
        R = 0.5*np.eye(self.nInput)
        Q_f = 125*np.eye(self.nState)
        labda = 0.1*np.eye((self.nState*(self.nState+1)/2))

        opt = trajectoryOptimization(self.nState, self.nSegments, self.nInput, self.nOutput, self.nModel, Q, R, Q_f, labda, self.dyna)

        return opt


    def plan_optimal_path(self):
        # print "In Planner, \nstart_mu: ", self.start_mu, "\nstart_cov:", self.start_cov,
        self.find_set_of_ds_to_cs_goals()
        return self.find_optimal_path()  


    def find_optimal_path(self):       
        all_plans = []
        mu_safe = copy.copy(self.belief.mu)
        cov_safe = copy.copy(self.belief.cov)
        wts_safe = copy.copy(self.belief.wts)

        # First Pass
        self.opt.nSegments = copy.copy(self.nSegments)
        cost, mu_plan, s_plan, u_plan, final_wts = self.opt.cs_optimize(self.start_mu, self.start_cov, self.belief.wts, self.t,  self.goal_cs)
        
        plan_path = [[mu_plan],[s_plan],[u_plan]]
        
        if self.do_verbose:
            print "Full Path in first go: \n", self.generateDetailedCSPlan(plan_path)

        cost = 10
        if cost < 1.:
            print "Cost: ", cost
            return self.generateDetailedCSPlan(plan_path)
        else:
            # raw_input('First pass failed. Press Enter to generate rollouts')
            self.opt.nSegments = copy.copy(self.nSegments)

            self.belief.mu = copy.copy(mu_safe)
            self.belief.cov = copy.copy(cov_safe)
            self.belief.wts = copy.copy(wts_safe)

            # convert goal in continuous states to ds goal
            cov = 1e-6*np.eye(self.nState)
            self.dyna.fastWtsMapped(self.goal_cs, cov, self.goal_wts)
            self.goal_ds = copy.copy(self.wts2ds(self.goal_wts)-1)

            ''' Do rollouts to find a feasible lower cost path  '''
            self.generate_rollouts()
            print "After rollout generation"

            ''' Calculate cost for the rollout'''
            # ## Convert rollout to wts
            if self.do_parallelize:
                all_plans = parmap(self.rollout_cost, self.rollouts)
            else:
                all_plans = []
                for rollout in self.rollouts:
                    rollout_data = self.rollout_cost(rollout)
                    all_plans.append(rollout_data)

            min_cost_path = self.optimal_ds_path(all_plans)

            self.belief.mu = copy.copy(min_cost_path.plan[0][:, -1])
            self.belief.cov[self.opt.id1] = copy.copy(min_cost_path.plan[1][:, -1])

            self.dyna.fastWtsMapped(self.belief.mu, self.belief.cov, self.belief.wts)

            print "min_cost_path: \n", min_cost_path.rollout

            ### Generate last part to go to goal            
            cost, mu_plan, s_plan, u_plan, final_wts = copy.copy(self.opt.cs_optimize(self.belief.mu, self.belief.cov, self.belief.wts, self.t, self.goal_cs))
            mu_plan, s_plan, u_plan = copy.copy(self.generateDetailedCSPlan([[mu_plan],[s_plan],[u_plan]]))

            # Appending the final plan
            min_cost_path.plan[0] = np.append(min_cost_path.plan[0], mu_plan, axis=1)
            min_cost_path.plan[1] = np.append(min_cost_path.plan[1], s_plan, axis=1)
            min_cost_path.plan[2] = np.append(min_cost_path.plan[2], u_plan, axis=1)
            plan_path = [min_cost_path.plan[0], min_cost_path.plan[1], min_cost_path.plan[2]]

            print "*****************************************"
            print "min_cost_path: \n", min_cost_path.rollout, "\ncs goals: ", min_cost_path.cs_goal_path# ,"\ncost :", min_cost_path.cost, "\ncs plan, mu: ", min_cost_path.plan[0].T, "\ncs plan, cov: ", min_cost_path.plan[1],  "\ncs plan, u: ", min_cost_path.plan[2]
            print "*****************************************"


            # return self.generateDetailedCSPlan(plan_path)

            # if self.show_traj:
            #     self.plot_trajectory(min_cost_path)

            return min_cost_path.plan[0], min_cost_path.plan[1], min_cost_path.plan[2]


    def generate_rollouts(self):
        self.rollouts = []

        # all_possible = range(1, self.nModel+1)
        all_possible = range(self.nModel)

        self.end_ds = self.wts2ds(self.goal_wts)
        print "end_ds = ", self.end_ds
        all_possible.remove(self.end_ds) # remove goal state

        # removing start ds
        wts = np.ndarray(self.nModel)
        # self.dyna.fastWtsMapped(copy.copy(self.belief.mu), copy.copy(self.belief.cov), wts)
        self.dyna.fastWtsMapped(copy.copy(self.start_mu), copy.copy(self.start_cov), wts)
        self.start_ds = copy.copy(self.wts2ds(wts))
        print "start_ds =", self.start_ds 

        if self.start_ds != self.end_ds:
            all_possible.remove(self.start_ds)
        else:
            raise ValueError("Start and Goal discrete states are same. Please check guard conditions!")

        # Generate all rollouts:
        for L in range(1, len(all_possible)+1):
            for subset in permutations(all_possible, L):
                subset = list(subset)
                subset.insert(0, self.start_ds)
                subset.append(self.end_ds)
                self.rollouts.append(subset)

        print "All possible rollouts : ", self.rollouts

        # self.rollouts = self.check_if_rollout_feasible(self.rollouts)
        # return

    def check_if_rollout_feasible(self, rollouts):
        feasible_rollouts = copy.copy(rollouts)
        return feasible_rollouts

    def rollout_cost(self, rollout):
        ds_goal_path = self.dsArray2wtsArray(rollout)[1:]
        # We don't need the first ds goal

        if self.do_verbose:
            print "ds_goal_path = ", ds_goal_path

        self.belief.mu = copy.copy(self.start_mu)
        self.belief.cov = copy.copy(self.start_cov)

        # convert start in continuous states to ds
        self.dyna.fastWtsMapped(self.belief.mu, self.belief.cov, self.belief.wts)

        start_wts = copy.copy(self.belief.wts)

        if self.do_verbose:
            print "Initial Belief = ", self.belief.mu, self.belief.cov, self.belief.wts


        # convert path to continuous states
        cs_goal_path = self.ds_traj_to_cs_traj(ds_goal_path)
        if self.do_verbose:
            print "cs_goal_path = ", cs_goal_path

        # raw_input('Check CS goal path')

        # Find optimal path for all continuous state goals
        planned_path = self.complete_optimal_cs_traj(cs_goal_path)
        if self.do_verbose:
            print "planned_path = \n", planned_path

        # Calculate associated cost
        final_wts = planned_path[-1][-1]

        # final_wts = self.belief.wts*1.
        cost = smooth_cost(self.goal_wts, final_wts, cost_type="KL", scale=1e3)

        # Adding cost for executing longer path
        cost += 0e3*self.t*len(ds_goal_path)

        rollout_in_ds = [self.start_ds] + self.wtsArray2dsArray(ds_goal_path)
        return data_info(rollout_in_ds, cost, stichCSPath(planned_path), final_wts, cs_goal_path)


    def optimal_ds_path(self, all_plans):
        global max_cost_differences
        max_cost_differences = []
        cost_diff = []
        print "Here########"
        
        min_cost = 10e6
        min_cost_plan = None

        for plan in all_plans:
            print "For rollout: ", plan.rollout , "Cost = ", plan.cost, "final wts = ", np.round(plan.final_wts,3), "cs_goal_path", plan.cs_goal_path

            if plan.cost < min_cost:
                min_cost = plan.cost*1.
                min_cost_plan = copy.copy(plan)

        max_cost_differences.append(cost_diff)
        # print "min_cost_plan", min_cost_plan.plan
        return min_cost_plan


    def ds_goal_to_cs_goal(self, ds_goal):
        # # generate ds_costmap
        # self.ds_costmap(ds_goal)

        # # do global optimization
        # res = differential_evolution(self.global_objective, self.domain)

        # # return cs goal
        # if self.do_verbose:
        #     wts = np.ndarray(self.nModel)
        #     self.dyna.fastWtsMapped(res.x, 0.01*np.eye(self.nState), wts)
        #     print "result = ", res.x
        #     print "Goal wts : ", ds_goal
        #     print "Wts at final Point : ", np.round(wts ,3)
        
        # # return res.x
        # projected_goal = copy.copy(self.project_on_boundary(self.wts2ds(ds_goal), res.x))
        # print "projected goal: ", projected_goal
        # return projected_goal

        return self.set_of_ds_goals[self.wts2ds(ds_goal)]


    def find_set_of_ds_to_cs_goals(self):
        for i in range(self.nModel):
            ds_goal = [0.]*self.nModel
            ds_goal[i] = 1.

            # generate ds_costmap
            self.ds_costmap(ds_goal)

            # do global optimization
            res = differential_evolution(self.global_objective, self.domain)

            # return cs goal
            if self.do_verbose:
                wts = np.ndarray(self.nModel)
                self.dyna.fastWtsMapped(res.x, 0.01*np.eye(self.nState), wts)
                print "result = ", res.x
                print "Goal wts : ", ds_goal
                print "Wts at final Point : ", np.round(wts ,3)
        
            projected_goal = copy.copy(self.project_on_boundary(self.wts2ds(ds_goal), res.x))
            self.set_of_ds_goals.append(projected_goal)
        print self.set_of_ds_goals


    def ds_traj_to_cs_traj(self, ds_traj):
        cs_traj = []
        for ds_goal in ds_traj:
            cs_goal = self.ds_goal_to_cs_goal(ds_goal)
            cs_traj.append(cs_goal)

        return cs_traj


    def optimal_cs_path(self, start, cs_goal):
        # do snopt based optimization to find the optimal trajectory
        muInit = start.mu
        covInit = start.cov
        wtsInit = start.wts

        start_time2 = time.time()
        cost, mu_plan, s_plan, u_plan, final_wts = self.opt.cs_optimize(muInit, covInit, wtsInit, self.t,  cs_goal)
        
        print "#############################################"
        print "Total Time in one call of SNOPT = %s seconds" % (time.time() - start_time2)
        global snopt_call
        snopt_call += 1
        print "SNOPT call number = ", snopt_call
        print "#############################################"

        
        return mu_plan, s_plan,u_plan, final_wts


    def complete_optimal_cs_traj(self, cs_traj):
        path = []

        for goal in cs_traj:
            mu_plan, s_plan,u_plan, final_wts = copy.copy(self.optimal_cs_path(self.belief, goal))
            mu_plan, s_plan, u_plan = copy.copy(self.generateDetailedCSPlan([[mu_plan],[s_plan],[u_plan]]))
            optimal_path =  [mu_plan, s_plan, u_plan, final_wts]

            if self.do_verbose:
                print "Optimal path: \n", optimal_path

            path.append(optimal_path)

            ## Next Iteration
            self.belief.mu = copy.copy(optimal_path[0][:,-1])
            self.belief.cov[self.opt.id1] = copy.copy(optimal_path[1][:,-1]) # Conversion of s to cov
            # self.belief.cov = 2*np.eye(self.nState) # Conversion of s to cov
            self.belief.wts = copy.copy(final_wts)
            # self.dyna.fastWtsMapped(self.belief.mu, self.belief.cov, self.belief.wts)

            # if self.do_verbose:
                # print "New Start point:\nmu: ", self.belief.mu, "\ncov: ", self.belief.cov, "\nwts: ", self.belief.wts
            # raw_input('Press Enter if should go to next cs_goal')

        return path


    ## Other FUNCTIONS

    def project_on_boundary(self, ds_goal, res):
        # Ideally can use another optimization process to find the nearest point, but here we can use direct formulae
        projection = deepcopy(res)
        if ds_goal == 1:
            projection[1] = -20.
        elif ds_goal == 2:
            projection[0] = -20.

        return projection

    def ds_costmap(self, goal_wts):
        # Local Parameters
        cov = 25.0*np.eye(self.nState)
        wts = np.ndarray(self.nModel)

        # Generate data:
        pts = np.random.random((self.nState,1000))

        # Scaling data based on the domain size
        for i in range(self.nState):
            pts[i,:] *= (self.domain[i][1] - self.domain[i][0])
            pts[i,:] += self.domain[i][0]

        costs = []
        for pt in pts.T:
            self.dyna.fastWtsMapped(pt*1., cov, wts)
            cost = copy.copy(smooth_cost(goal_wts, wts, cost_type="Hellinger", scale=1e3)) # cost for ds mismatch
            # cost += 5e2*numpy.linalg.norm(self.goal_cs - pt)
            
            if (np.linalg.norm(goal_wts - np.array([0., 0., 0., 1.])) > 1.0): 
                # cost += 50.*numpy.linalg.norm(copy.copy(np.array([-140.0, -20.0])) - pt)
                cost += 10.*numpy.linalg.norm(copy.copy(self.goal_cs) - pt)
            else:
                cost += 500.*numpy.linalg.norm(copy.copy(self.goal_cs) - pt)


            costs.append(cost)

        costs = np.array(costs)

        # Interpolate
        x = pts[0,:]
        y = pts[1,:] # - 6*np.ones(len(pts.T))
        self.rbf = scipy.interpolate.Rbf(x, y, costs, function='multiquadric')
        return  

    def global_objective(self, X):
        return self.rbf(X[0], X[1])

    def ds2Wts(self, ds_state):
        wts = np.zeros(self.nModel)
        # wts[ds_state-1] = 1.
        wts[ds_state] = 1. 
        return wts

    def dsArray2wtsArray(self, dsArray):
        wtsArray = []
        for ds in dsArray:
            wtsArray.append(self.ds2Wts(ds))
        return wtsArray

    def wts2ds(self, wts):
        # return int(np.argmax(wts) + 1) 
        return int(np.argmax(wts)) 


    def wtsArray2dsArray(self, wtsArray):
        dsArray = []
        for wts in wtsArray:
            dsArray.append(self.wts2ds(wts))
        return dsArray


    def generateDetailedCSPlan(self, input_plan):
        mu_new = input_plan[0][0]
        s_new = input_plan[1][0]
        u_new = input_plan[2][0]


        # New matrics
        mu_plan = np.zeros((self.nState,int((self.nSegments-1)*self.opt.delta)+1))
        s_plan = np.zeros((self.opt.len_s, int((self.nSegments-1)*self.opt.delta)+1))
        u_plan = np.zeros((self.nInput,int((self.nSegments-1)*self.opt.delta)+1))

        mu_plan[:,0] = mu_new[:,0]*1
        s_plan[:,0] = s_new[:, 0]*1

        ds_old = 0
        w_new = np.ndarray(self.nModel)
        mu_new  = np.ndarray(self.nState)

        mu = copy.copy(mu_plan[:,0])
        cov = symarray(np.zeros((self.nState, self.nState)))
        cov[self.opt.id1] = s_plan[:,0]*1.

        for i in range(self.nSegments-1):
            for t in range(1, self.opt.delta+1):

                ds_new = self.opt.sysDynamics.beliefUpdatePlanning(mu, cov, u_new[:,i], mu_new, cov, w_new, ds_old)

                # wts_plan[:,i*self.opt.delta+t+1] = w_new*1.
                mu_plan[:,i*self.opt.delta+t] = copy.copy(mu_new)
                s_plan[:,i*self.opt.delta+t] = copy.copy(cov[self.opt.id1])
                u_plan[:,i*self.opt.delta+t-1] = copy.copy(u_new[:,i])

                mu = copy.copy(mu_new)
                
                if self.do_verbose:
                    print "\n Planned Step", i*self.opt.delta+t
                    print "Planned mu = ", np.round(mu_plan[:,i*self.opt.delta+t], 3)
                    print "Planned s = ", np.round(s_plan[:,i*self.opt.delta+t], 3)
                    print "Planned Control, u_plan = ", np.round(u_plan[:,i*self.opt.delta+t-1], 2)

        return mu_plan, s_plan, u_plan




    def plot_trajectory(self, plan_traj):
        import matplotlib.pyplot as plt

        print "#############################################"
        print "\n\n\n\nFinal Trajecotry"
        traj = []
        for plan in plan_traj:
            traj_dummy = np.array(plan[0]).T
            for i in range(len(traj_dummy)):
                print "Planned Trajectory", traj_dummy[i]
                traj.append(traj_dummy[i])
                # print "True Trajectory", traj_true[i]

        print "#############################################"

        ### Plotting
        fig = plt.figure(1)

        ### Setting Up Domain
        x1 = 3.0
        y1 = self.domain[1][0]
        x2 = 3.0
        y2 = self.domain[1][1]
        plt.plot([x1, x2], [y1, y2], color='b', linestyle='--', linewidth=2)

        x1 = 6.0
        y1 = self.domain[1][0]
        x2 = 6.0
        y2 = self.domain[1][1]
        plt.plot([x1, x2], [y1, y2], color='b', linestyle='--', linewidth=2)

        x1 = 3.0
        y1 = -2.0
        x2 = 6.0
        y2 = -2.0
        plt.plot([x1, x2], [y1, y2], color='b', linestyle='--', linewidth=2)

        x1 = 3.0
        y1 = 3.0
        x2 = 6.0
        y2 = 3.0
        plt.plot([x1, x2], [y1, y2], color='b', linestyle='--', linewidth=2)


        x_vec = [traj[i][0] for i in range(len(traj))]
        y_vec = [traj[i][1] for i in range(len(traj))]
        # x_vec_true = [traj_true[i][0] for i in range(len(traj_true))]
        # y_vec_true = [traj_true[i][1] for i in range(len(traj_true))]

        bl1, = plt.plot(x_vec,y_vec, 'k-o', linewidth = 2.0)
        # bl2, = plt.plot(x_vec_true,y_vec_true, 'b-D',linewidth = 3.0)

        plt.title('Trajectories for Discrete state based planning')
        # plt.legend([bl1, bl2], ['Planned Trajectory', 'Actual Trajectory'])

        # img = imread("domain/test_domain6.png")
        # # img1 = ndimage.rotate(img, 90)
        # img1 = np.flipud(img)
        # plt.imshow(img1, zorder=0, extent=[0., 10.0, 0.0, 10.0])
        # axes = plt.gca()
        # Start Point
        plt.plot(x_vec[0], y_vec[0], 'ko', x_vec[-1], y_vec[-1], 'gs' , ms=10.0, mew=2.0)
        # plt.plot(x_vec_true[0], y_vec_true[0], 'ko', x_vec_true[-1], y_vec_true[-1], 'gs' , ms=10.0, mew=2.0)   
        # axes.set_xlim([xmin,xmax])
        # axes.set_ylim([-2.0, 3.0])
        fig.show()
        raw_input()

        return



if __name__ == "__main__":
    planner = pomdp_kc()

    planner.dyna.nSamples = 10
    mu = np.array([2.0, 0.5])
    cov = 5*np.eye(planner.nState)
    wts = np.ndarray(planner.nModel)
    planner.belief = hybrid_belief(mu, cov, wts)
    goal = np.array([5.0, 0.0])

    # planner.rollouts = [[1,3], 
    # planner.rollouts = [[1, 5, 3], [1, 2, 3]] #, [1, 4, 3], [1, 2, 5, 3], [1, 4, 5, 3], [1, 2, 4, 5, 3]]

    start_time = time.time()
    planner.start_mu = mu
    planner.start_cov = cov
    planner.goal_cs = goal
    planner.do_parallelize = True
    planner.plan_optimal_path()

    if planner.do_parallelize:
        print "Total Time spent with paraellization = %s seconds" % (time.time() - start_time)
    else:
        print "Total Time spent without paraellization = %s seconds" % (time.time() - start_time)

    global max_cost_differences
    print "cost diffrences", max_cost_differences


    '''
    #######Timing for nSamples Parameter##########
    n_iter = 5
    # Samples = 100
    planner.dyna.nSamples = 100
    start_time = time.time()
    for i in range(n_iter):
        mu = np.array([2.0, 0.5])
        cov = 5*np.eye(planner.nState)
        wts = np.ndarray(planner.nModel)
        planner.belief = hybrid_belief(mu, cov, wts)

        goal = np.array([10.0, 0.])
        planner.find_optimal_path(mu, cov, goal)

    print "N Samples for dynamics: ", planner.dyna.nSamples
    print "Average Time spent in Planning = %s seconds" % ((time.time() - start_time)/n_iter)
    '''

    # planner.belief.mu = np.array([  4.76, -14.52] )
    # # planner.belief.cov = 5.0*np.eye(planner.nState)
    # planner.belief.cov = np.array([[ 0.0310559,  0.],[ 0.,0.0310559]])
    # planner.belief.wts = np.array([ 0.,  1.,  0.,  0.,  0.])
    # goal = np.array([ 4.67, -0.16 ])

    # planner.optimal_cs_path(planner.belief, goal)

    ''' Checking PD functionalties

    # cov = np.array([[ 16.75690398,  10.94250939], [ 10.94250939 ,  7.14562259]])
    # res = np.ndarray((planner.nState, planner.nState))
    # planner.dyna.nearestPDMapped(cov, res)

    # print "Input Matrix = ", cov
    # print "res =", res
    # print "isPD = ", planner.dyna.isPDMapped(res)
    '''
