# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import itertools
from copy import deepcopy

from planner.scripts.utils import *
from planner.bindings.python.beliefEvolution import *
from scipy.optimize import differential_evolution

domain = [(-100., 100.), (-200.0, 200.0)]
rbf = None
n_pts = 250
goal_cs = np.array([50., 0.])


def ds_costmap(goal_wts, goal_cost, cov_multiplier=1.0, cost_type="Hellinger"):
    global domain, rbf, goal_cs
    nState = 2
    nModel = 5

    # Local Parameters
    cov = deepcopy(cov_multiplier*np.eye(nState))
    wts = np.ndarray(nModel)

    # Generate data:
    pts = np.random.random((nState,n_pts))

    # Scaling data based on the domain size
    for i in range(nState):
        pts[i,:] *= (domain[i][1] - domain[i][0])
        pts[i,:] += domain[i][0]

    costs = []
    for pt in pts.T:
        dyna.fastWtsMapped(pt*1., cov, wts)
        cost = deepcopy(smooth_cost(goal_wts, wts, cost_type, scale=1e3)) # cost for ds mismatch
        # cost += goal_cost*np.linalg.norm(goal_cs - pt)
        # if((np.linalg.norm(goal_wts - np.array([0., 0., 0., 0., 1.])) > 0.5) or (np.linalg.norm(goal_wts - np.array([1., 0., 0., 0., 0.])) > 1.0)):
        if((np.linalg.norm(goal_wts - np.array([0., 0., 0., 0., 1.])) > 0.5)):
            # cost += goal_cost*np.linalg.norm(deepcopy(np.array([-140.0, -20.0])) - pt)
            cost += goal_cost*np.linalg.norm(deepcopy(goal_cs) - pt)

        else:
            cost += 500.*np.linalg.norm(deepcopy(goal_cs) - pt)

        costs.append(cost)

    costs = np.array(costs)

    # Interpolate
    x = pts[0,:]
    y = pts[1,:] # - 6*np.ones(len(pts.T))
    rbf = scipy.interpolate.Rbf(x, y, costs, function='multiquadric')
    plot_costmap(x, y, rbf)
    return  

def global_objective(X):
    global rbf
    return rbf(X[0], X[1])


def check_gcs(goal, x):
    if (np.linalg.norm(goal - np.array([1., 0., 0., 0., 0.])) < 1.0):
        if((x[0] > 12.5 and x[0] < 500.) and ((x[1] > -500. and x[1] < -125.))):
            return "True"
        else:
            return "False"

    elif (np.linalg.norm(goal - np.array([0., 1., 0., 0., 0.])) < 1.0):
        if((x[0] > 2.5 and x[0] < 12.5) and ((x[1] > -115. and x[1] < 500.))):
            return "True"
        else:
            return "False"

    elif (np.linalg.norm(goal - np.array([0., 0., 1., 0., 0.])) < 1.0):
        if((x[0] > -500. and x[0] < 2.5) and ((x[1] > -125. and x[1] < -115.))):
            return "True"
        else:
            return "False"

    elif (np.linalg.norm(goal - np.array([0., 0., 0., 0., 1.])) < 1.0):
        if((x[0] > 11.5 and x[0] < 15.5) and ((x[1] > -2.0 and x[1] < 2.0))):
            return "True"
        else:
            return "False"


def plot_costmap(x, y, rbf):
    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(x.min(), x.max(), n_pts), np.linspace(y.min(), y.max(), n_pts)
    xi, yi = np.meshgrid(xi, yi)

    costs = rbf(xi, yi)
    # costs /= costs.max()

    plt.imshow(costs, vmin=costs.min(), vmax=costs.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
    # plt.scatter(x, y, c=costs)
    plt.colorbar()
    plt.title('Discrete State Costmap')

    # Setup domain
    plt.plot([12.5, 12.5], [-125., 200.], 'k')
    plt.plot([2.5, 2.5], [-125., 200.], 'k')
    plt.plot([12.5, -100.], [-125., -125.], 'k')
    plt.plot([12.5, -100.], [-115., -115.], 'k')
    plt.plot(goal_cs[0], goal_cs[1], 'kx', ms=20, mew=5)


    plt.show()
    # raw_input('Press Enter to close plots')
    return


if __name__ == "__main__":
    global domain
    dyna = BeliefEvolution()
    dyna.nSamples = 100

    goal_wts = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0, 0.], [0., 0., 0., 0., 1.]])
    # goal_wts = np.array([[0., 1., 0., 0., 0.]])

    dyna.setGC(deepcopy(goal_cs)) # Setting GC


    dist = []
    # cov_multiplier = [0.01, 0.1, 1.0, 10., 25., 50. ]
    # g_costs = [0., 1e-3, 0.01, 0.05, 0.5, 1.0, 5., 10.]

    cov_multiplier = [25.0]
    g_costs = [10.0]

    Selection_set = []

    for cov_m in cov_multiplier:
        print "*************"
        print "For cov multiplier: ", cov_m, "\n"

        for goal_cost in g_costs:
            print "###############"
            print "For distanace multiplier: ", goal_cost, "\n"
            res_array = []
            for goal in goal_wts:
                # generate ds_costmap
                ds_costmap(goal, goal_cost, cov_m, 'Hellinger')
                # do global optimization
                res = differential_evolution(global_objective, domain)
                res2 = check_gcs(goal, res.x)
                print "Minima for wts: ", goal, " is at : ", res.x, "Test: ", res2
                res_array.append(res2)
                # dist.append(np.linalg.norm(res.x - goal_cs)) # as goal is origin
            # print "Result:", res_array
            if(all([res_array[i] == 'True' for i in range(len(res_array)) ])):
                Selection_set.append([[cov_m], [goal_cost]])


    print "\n\n\nSelected Sets: ", Selection_set
    # ''' Plotting Trajecotry '''
    # import matplotlib.pyplot as plt

    # ### Plotting
    # fig = plt.figure(1)
    # bl1, = plt.plot(g_costs, dist, 'b')

    # print "\n\n\n########"
    # goal_wts = np.array([0., 1., 0., 0., 0.])
    # dist = []
    # for goal_cost in g_costs:
    #     # generate ds_costmap
    #     ds_costmap(goal_wts, goal_cost, 'Hellinger')
    #     # do global optimization
    #     res = differential_evolution(global_objective, domain)
    #     print "For distanace multiplier: ", goal_cost  ,"Minima for [0., 1., 0.] is at : ", res.x 
    #     dist.append(np.linalg.norm(res.x - goal_cs)) # as goal is origin


    # # bl1, = plt.semilogx(g_costs, dist)
    # bl2, = plt.plot(g_costs, dist, 'r')

    # # Goal_wts 2
    # print "\n\n\n########"
    # goal_wts = np.array([0., 0., 1.,0.,0.])
    # dist = []

    # for goal_cost in g_costs:
    #     # generate ds_costmap
    #     ds_costmap(goal_wts, goal_cost, 'Hellinger')
    #     # do global optimization
    #     res = differential_evolution(global_objective, domain)
    #     print "For distanace multiplier: ", goal_cost  ,"Minima for [0., 1., 0.] is at : ", res.x 
    #     dist.append(np.linalg.norm(res.x)) # as goal is origin


    # bl3, = plt.plot(g_costs, dist, 'g')

    # # Goal_wts 3
    # print "\n\n\n########"
    # goal_wts = np.array([0., 0., 0., 1.,0.])
    # dist = []

    # for goal_cost in g_costs:
    #     # generate ds_costmap
    #     ds_costmap(goal_wts, goal_cost, 'Hellinger')
    #     # do global optimization
    #     res = differential_evolution(global_objective, domain)
    #     print "For distanace multiplier: ", goal_cost  ,"Minima for [0., 1., 0.] is at : ", res.x 
    #     dist.append(np.linalg.norm(res.x)) # as goal is origin


    # bl4, = plt.plot(g_costs, dist, 'k')

    # # Goal_wts 4
    # print "\n\n\n########"
    # goal_wts = np.array([0., 0., 0.,0., 1.])
    # dist = []

    # for goal_cost in g_costs:
    #     # generate ds_costmap
    #     ds_costmap(goal_wts, goal_cost, 'Hellinger')
    #     # do global optimization
    #     res = differential_evolution(global_objective, domain)
    #     print "For distanace multiplier: ", goal_cost  ,"Minima for [0., 1., 0.] is at : ", res.x 
    #     dist.append(np.linalg.norm(res.x)) # as goal is origin


    # bl5, = plt.plot(g_costs, dist, 'm')


    # # bl2, = plt.semilogx(g_costs, dist)
    # plt.legend([bl1, bl2, bl3, bl4, bl5], ['wts = {1., 0., 0.,0., 0.}', 'wts={0., 1., 0., 0., 0.}', 'wts={0., 0., 1.,0., 0.}', 'wts={0., 0., 0., 1., 0.}', 'wts={0., 0., 0.,0., 1.}'])


    # plt.xlabel('Multiplier for distance from cs goal when calculating optimal ds -> cs goals')
    # plt.ylabel('Distance from goal')
    # plt.title('Variation of optimal ds->cs goal with cost multiplier')



    # # ### Setting Up Domain
    # # x1 = 0.01
    # # y1 = 225.0
    # # x2 = 1e6
    # # y2 = 225.0
    # # plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=3)

    # fig.show()
    # raw_input('Press Enter to exit!')


    # # # Generate data:
    # # pts = 15*np.random.random((2,100)) - 5
    # # x = pts[0,:]
    # # y = pts[1,:] # - 6*np.ones(len(pts.T))

    # # # Set up a regular grid of interpolation points
    # # xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    # # xi, yi = np.meshgrid(xi, yi)

    # # #### KL
    # # z_kl = []
    # # for mu in pts.T:
    # #   dyna.fastWtsMapped(mu*1., cov, wts)
    # #   z_kl.append(KL(goal_wts, wts))
    # # z_kl = np.array(z_kl)

    # # # Interpolate
    # # rbf = scipy.interpolate.Rbf(x, y, z_kl, function='linear')
    # # z_kli = rbf(xi, yi)

    # # plt.subplot(2, 2, 1)
    # # plt.imshow(z_kli, vmin=z_kl.min(), vmax=z_kl.max(), origin='lower',
    # #            extent=[x.min(), x.max(), y.min(), y.max()])
    # # plt.plot(x, y, c=z_kl)
    # # plt.colorbar()
    # # plt.title('KL Divergence')

    # # #### sym KL
    # # z_skl = []
    # # for mu in pts.T:
    # #   dyna.fastWtsMapped(mu*1., cov, wts)
    # #   z_skl.append(0.5*sym_KL(goal_wts, wts))
    # # z_skl = np.array(z_skl)

    # # # Interpolate
    # # rbf = scipy.interpolate.Rbf(x, y, z_skl, function='linear')
    # # z_skli = rbf(xi, yi)

    # plt.subplot(2, 2, 2)

    # plt.imshow(z_skli, vmin=z_skl.min(), vmax=z_skl.max(), origin='lower',
    #            extent=[x.min(), x.max(), y.min(), y.max()])
    # plt.plot(x, y, c=z_skl)
    # plt.colorbar()
    # plt.title('Symm KL Divergence')


    # #### Bhattacharya
    # z_bhatta = []
    # for mu in pts.T:
    #   dyna.fastWtsMapped(mu*1., cov, wts)
    #   z_bhatta.append(2*Bhattacharya(goal_wts, wts))
    # z_bhatta = np.array(z_bhatta)

    # # Interpolate
    # rbf = scipy.interpolate.Rbf(x, y, z_bhatta, function='linear')
    # z_bhattai = rbf(xi, yi)

    # plt.subplot(2, 2, 3)
    # plt.imshow(z_bhattai, vmin=z_bhatta.min(), vmax=z_bhatta.max(), origin='lower',
    #            extent=[x.min(), x.max(), y.min(), y.max()])
    # plt.plot(x, y, c=z_bhatta)
    # plt.colorbar()
    # plt.title('Bhattacharya')



    # #### Hellinger
    # z_hal = []
    # for mu in pts.T:
    #   dyna.fastWtsMapped(mu*1., cov, wts)
    #   z_hal.append(8*Hellinger(goal_wts, wts))
    # z_hal = np.array(z_hal)

    # # Interpolate
    # rbf = scipy.interpolate.Rbf(x, y, z_hal, function='linear')
    # z_hali = rbf(xi, yi)

    # plt.subplot(2, 2, 4)
    # plt.imshow(z_hali, vmin=z_hal.min(), vmax=z_hal.max(), origin='lower',
    #            extent=[x.min(), x.max(), y.min(), y.max()])
    # plt.plot(x, y, c=z_hal)
    # plt.colorbar()
    # plt.title('Hellinger')

    # plt.show()