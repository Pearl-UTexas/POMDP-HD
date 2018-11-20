#include "../../include/dynamics_models/linear_dynamics.h"

namespace dynamics{

void LinearDynamics::printModelParams()
{
    std::string sep = "\n----------------------------------------\n";
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::cout << "Linear Dynamics Model Parameters: \n";
    std::cout << "Number of States: " << nState << "\n";
    std::cout << "Number of Input : " << nInput << "\n";
    std::cout << "Number of Output: " << nOutput << "\n";
    std::cout << "\nLinear Model Matrices:\n";
    std::cout << "A: \n" << A.format(CleanFmt) << "\n";
    std::cout << "B: \n" << B.format(CleanFmt) << "\n";
    std::cout << "C: \n" << C.format(CleanFmt) << "\n";
    std::cout << "V: \n" << V.format(CleanFmt) << "\n";
    std::cout << "W: \n" << W.format(CleanFmt) << "\n";
    std::cout << sep;
}

void LinearDynamics::setModelParams(int n_x, int n_u, int n_y){
        nState = n_x;
        nInput = n_u;
        nOutput = n_y;      
}

void LinearDynamics::propagateState(Eigen::VectorXd x, Eigen::VectorXd u, Eigen::VectorXd& x_new)
{
    x_new = A*x + B*u;
}

void LinearDynamics::propagateStateWithCov(Eigen::VectorXd x, Eigen::MatrixXd cov, Eigen::VectorXd u, Eigen::VectorXd& x_new, Eigen::MatrixXd& cov_new)
{
    x_new = A*x + B*u;
    cov_new = A*cov*A.transpose() + V;
}

void LinearDynamics::getObservation(Eigen::VectorXd x, Eigen::VectorXd& z)
{
    // Generate Noise
    Eigen::VectorXd zero_mean_ = Eigen::VectorXd::Zero(nState);
    Eigen::MatrixXd noise_covar_;
    utils::nearestPD(W, noise_covar_);
    Eigen::EigenMultivariateNormal<double> normX_cholesk(zero_mean_, noise_covar_);
    Eigen::VectorXd obs_noise_ = normX_cholesk.samples(1);

    // Generate Observation
    z = C*x + obs_noise_;
}

void LinearDynamics::getObservationCov(Eigen::MatrixXd cov, Eigen::MatrixXd& cov_new)
{
    Eigen::MatrixXd new_cov;
    new_cov = C*cov*C.transpose() + W;
    utils::nearestPD(new_cov, cov_new); // This ensures that cov_new is PD
}

void LinearDynamics::getObservationNoNoise(Eigen::VectorXd x, Eigen::VectorXd& z)
{
    z = C*x;
}

void LinearDynamics::getObservationWithCov(Eigen::VectorXd x, Eigen::MatrixXd cov, Eigen::VectorXd& z, Eigen::MatrixXd& cov_new)
{
    getObservation(x, z);
    getObservationCov(cov, cov_new); 
}
} // end namespace
