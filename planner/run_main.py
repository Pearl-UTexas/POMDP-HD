# -*- coding: utf-8 -*-

import copy
import numpy as np
import time

from planner_interface import planner_interface

if __name__ == "__main__":
    start_time = time.time()

    x = np.array([35., 20.])
    mu = np.array([50.0, 50.0])
    goal = np.array([0.0, 0.])

    planner = planner_interface(x, mu, goal)
    planner.do_parallelize = True
    planner.do_verbose = False

    ## Tolerance on replanning
    replanning_threshld = 5.0 
    
    ## Exit criterion
    max_final_error = 1.0 

    traj = [planner.mu_actual]
    traj_true = [planner.x_actual]
    cov_traj = [planner.cov_actual[planner.planner.opt.id1]]
    ds_traj = [planner.wts_actual]

    while((max(abs(mu - goal)) > max_final_error)):
        mu_plan, s_plan, u_plan = planner.generate_plan()
        print "mu_plan: ", mu_plan
        # raw_input('Press Enter to continue')

        for t in range(len(mu_plan.T)-1):
            x, z = planner.execute_plan_oneStep(mu_plan[:,t], s_plan[:,t], u_plan[:,t])
            planner.update_belief(z)

            print "\ntime Step =", t + 1
            print "Active Model id =",  planner.idx
            print "Observation = ", z
            print "mu after EKF = ", np.round(planner.muNew, 3)
            print "actual x = ", np.round(x, 3)
            print "planned s at t+1 =", s_plan[:, t]
            print "Expected actual s at t + 1=", planner.covNew[planner.id1]
            print "Current mu from optimization = ", np.round(mu_plan[:,t], 3)
            print "Next Step Planned mu from optimization = ", np.round(mu_plan[:,t+1], 3)
            print "Model wts =", planner.wtsNew

            ''' Updates for Next iteration'''
            planner.update_stored_values()
            mu = copy.copy(planner.mu_actual)
            traj.append(planner.mu_actual)
            traj_true.append(planner.x_actual)
            cov_traj.append(s_plan[:, t+1])
            ds_traj.append(planner.wts_actual)

            ### Check if belief diverged too far.
            if(max(abs(mu-mu_plan[:,t+1])) > replanning_threshld):
                print "BLQR unable to stabilize. Replanning \n"
                # raw_input('press Enter to replan')
                break

        print "\n####################"
        print "muFinal = ", np.round(mu, 3)
        print "xFinal = ", np.round(x, 3)
        print "#####################\n"

    tActual = np.round((time.time() - start_time), 3)
    print("Total time required by Planner = %s seconds "  %tActual)  

    ## Data Capture
    fileName = 'data/run'+ "_Qf_" +str(planner.planner.opt.Q_f[0,0]) + "_lambda_"+ str(planner.planner.opt.labda[0,0])+ "_run_" + str(np.random.randint(100)) + '.npz'
    print "Data Saved in ", fileName
    np.savez(fileName, belief_traj=traj, actual_traj=traj_true, discrete_traj=ds_traj, cov_traj=cov_traj, run_time=tActual)

    ####
    print "#############################################"
    print "\n\n\n\nFinal Trajecotry"
    for i in range(len(traj)):
        print "Step: ", i
        print "Planned Trajectory:", traj[i]
        print "True Trajectory   :", traj_true[i]
        print "Covariance :", cov_traj[i]
    print "#############################################"

    ''' Plotting Trajecotry '''
    import matplotlib.pyplot as plt
    from utils.plotting import plot_cov_ellipse

    ### Plotting
    fig = plt.figure(1)

    ### Setting Up Domain
    x1 = -21.0
    y1 = -21.0
    x2 = planner.planner.domain[0][1]
    y2 = -21.0
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=15)

    x1 = -21.0
    y1 = -21.0
    x2 = -21.0
    y2 = planner.planner.domain[1][1]
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=10)

    x_vec = [traj[i][0] for i in range(len(traj))]
    y_vec = [traj[i][1] for i in range(len(traj))]
    x_vec_true = [traj_true[i][0] for i in range(len(traj_true))]
    y_vec_true = [traj_true[i][1] for i in range(len(traj_true))]

    bl1, = plt.plot(x_vec,y_vec, 'k-o', linewidth = 2.0)
    bl2, = plt.plot(x_vec_true,y_vec_true, 'b-D',linewidth = 3.0)

    # plotting covariance
    covar = np.zeros((planner.planner.nState, planner.planner.nState))
    # print "Covar: ", len(covar[planner.planner.opt.id1])
    for i in np.arange(0, len(x_vec), 2):
        covar[planner.planner.opt.id1] = copy.copy(cov_traj[i])
        plot_cov_ellipse(traj[i], covar, nstd=1., alpha=0.25, color='k')

    plt.title('Trajectories for Discrete state based planning')
    plt.legend([bl1, bl2], ['Planned Trajectory', 'Actual Trajectory'])

    # Start Point
    plt.plot(x_vec[0], y_vec[0], 'ko', x_vec[-1], y_vec[-1], 'gs' , ms=10.0, mew=2.0)
    plt.plot(x_vec_true[0], y_vec_true[0], 'ko', x_vec_true[-1], y_vec_true[-1], 'gs' , ms=10.0, mew=2.0)

    # plt.xlim([-20.0,100.])
    # plt.ylim([-20.0, 100.0])
    # plt.axis('equal')
    fig.show()
raw_input('Press Enter to exit!')




