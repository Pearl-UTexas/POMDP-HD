import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(pos, cov, nstd, ax, **kwargs)


def plot_cov_ellipse(pos, cov, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_trajectories(traj, traj_true, cov_traj, idx=0, save_fig=False):
    """
    Plot trajectory 

    Parameters
    ----------
        traj : Belief Trajectory
        traj_true  : Actual Trajectory from Simulation
        cov_traj: Covariance Values recorded along the trajectory
        idx: Plot images till this time step
        save_fig: Whether or not should save figure   

    """

    print "#############################################"
    print "Final Trajecotry"
    for i in range(len(traj)):
        print "Step: ", i
        print "Planned Trajectory:", traj[i]
        print "True Trajectory   :", traj_true[i]
    print "#############################################\n"

    fig = plt.figure(figsize=(15, 15))

    ## Setting Up Domain ---  DEPENDS ON EACH DOMAIN. PLEASE SET IT UP ACCORDINGLY
    x1 = -2.10
    y1 = -2.10
    x2 = 10.0
    y2 = -2.10
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=30)

    x1 = -2.10
    y1 = -2.10
    x2 = -2.10
    y2 = 10.0
    plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=30)


    ## Plot belief Trajectory
    x_vec = [traj[i][0] for i in range(len(traj))]
    y_vec = [traj[i][1] for i in range(len(traj))]

    ## Plot actual Trajectory
    x_vec_true = [traj_true[i][0] for i in range(len(traj_true))]
    y_vec_true = [traj_true[i][1]
                  for i in range(len(traj_true))]

    bl1, = plt.plot(x_vec, y_vec, 'r-o', linewidth=5.0, ms=15.0)
    bl2, = plt.plot(x_vec_true, y_vec_true, 'b-D', linewidth=5.0, ms=15.0)

    ## Plot covariance
    for i in np.arange(0, len(x_vec), 2):
        covar = np.array([[cov_traj[i][0], cov_traj[i][1]],
                          [cov_traj[i][1], cov_traj[i][2]]])
        plot_cov_ellipse(traj[i], covar, nstd=1., alpha=0.25, color='k')

    ## Plot Initial Point
    plt.plot(x_vec[0], y_vec[0], 'ko', ms=15.0, mew=5.0)
    plt.plot(x_vec_true[0], y_vec_true[0], 'ko', ms=15.0, mew=5.0)

    ## Plot Goal Point
    plt.plot(0., 0., 'gs', ms=18.0, mew=5.0)

    ## Set PLot Parameters
    plt.xlim([-2.00, 10.0])
    plt.ylim([-2.00, 10.00])
    plt.xlabel('x axis (in cm)', fontsize=25)
    plt.ylabel('y axis (in cm)', fontsize=25)
    plt.title('Trajectories for Discrete state based planning', fontsize=30)
    plt.legend([bl1, bl2], ['Belief Trajectory',
                            'Actual Trajectory'], fontsize=25)

    plt.tick_params(labelsize=20)

    if save_fig:
        plt.savefig('../data/trajPlots/Step ' + str(idx), bbox_inches='tight')
    else:
        fig.show()
        raw_input('Press Enter to close Plot')
    plt.close()


def plot_frame_by_frame(traj_mu, traj_x, cov_traj, save_fig=False):
    """
    Plot trajectory frame-by-frame (Useful for making animations)

    Parameters
    ----------
        traj_mu : Belief Trajectory
        traj_x  : Actual Trajectory from Simulation
        cov_traj: Covariance Values recorded along the trajectory
        save_fig: Whether or not should save figure   

    """
    for j in range(len(traj_mu)+1):
        plot_trajectories(traj_mu[:j+1], traj_x[:j+1], cov_traj[:j+1], idx=j+1, save_fig=save_fig)


def plot_full_traj(traj_mu, traj_x, cov_traj, save_fig=False):
    """
    Plot full trajectory (Single Plot)

    Parameters
    ----------
        traj_mu : Belief Trajectory
        traj_x  : Actual Trajectory from Simulation
        cov_traj: Covariance Values recorded along the trajectory     
        save_fig: Whether or not should save figure

    """
    plot_trajectories(traj_mu, traj_x, cov_traj, save_fig=save_fig)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        f_name = sys.argv[1]
    else:
        f_name = "../data/better_res_2_labda1e7_q_f_1e4_r_10.npz"

    data = np.load(f_name)
    traj = data['belief_traj']
    traj_true = data['actual_traj']
    cov_traj = data['cov_traj']

    plot_full_traj(traj, traj_true, cov_traj)
    # plot_frame_by_frame(traj, traj_true, cov_traj)
