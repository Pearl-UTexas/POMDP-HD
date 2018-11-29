#include "../../include/filters/ukf.h"

namespace filters
{
UKF::UKF()
{
    // Define Dynamics
    ProblemDefinition();
    n_x = nState;
    n_v = nState; // currently we define it same as n_x 
    n_z = nOutput;

    // Parameters
    alpha = 0.1;
    beta = 2;
    kappa = 0;
}

void UKF::findSigmaPts(Eigen::VectorXd x_0, 
        Eigen::MatrixXd aug_mat, double labda, 
        double delta_factor, std::vector<Eigen::VectorXd>& sigma_points)
{
    Eigen::MatrixXd sqrt_mat = aug_mat.llt().matrixL().transpose();

    sigma_points.push_back(x_0); // X0
    Eigen::VectorXd x_i(n_x+n_v);

    for(int i=0; i<n_x+n_v; i++)
    {
        x_i = x_0 + delta_factor*sqrt_mat.col(i);
        sigma_points.push_back(x_i);
    }

    for(int i=0; i<n_x+n_v; i++)
    {
        x_i = x_0 - delta_factor*sqrt_mat.col(i);
        sigma_points.push_back(x_i);
    }

    //for(int i=0; i<sigma_points.size(); i++)
    //   std::cout<< "Sigma Pt " << i << " : " << 
    //       sigma_points.at(i).transpose() << std::endl;
}

void UKF::prediction(Eigen::VectorXd x_hat, 
        Eigen::MatrixXd P_hat, Eigen::VectorXd u, 
        Eigen::VectorXd& x_bar, Eigen::MatrixXd& P_bar)
{
    Eigen::VectorXd x_aug_hat = Eigen::VectorXd(n_x+n_v);
    Eigen::MatrixXd P_aug_hat = Eigen::MatrixXd::Zero(n_x+n_v, n_x+n_v);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n_v);
    x_aug_hat << x_hat, v;
    P_aug_hat.block(0,0, n_x, n_x) = P_hat;
    
    int activeIdx;
    utils::activeModel(x_hat, activeIdx, nState, nModel, set_of_gcs);
    P_aug_hat.block(n_x, n_x, n_v, n_v) = dynamics[activeIdx]->V;

    double labda = pow(alpha, 2)*(n_x+n_v+kappa) - (n_x + n_v);
    double delta_factor = sqrt(n_x+n_v+labda);
    std::vector<Eigen::VectorXd> sigma_points;
    findSigmaPts(x_aug_hat, P_aug_hat, labda, delta_factor, sigma_points);

    // Propagate Sigma Pts through Dynamics
    std::vector<Eigen::VectorXd> predicted_sigma_pts;
    Eigen::VectorXd nx_sigma_pt(n_x), x_i(n_x), v_i, new_sigma_pt(n_x);
    for(int i=0; i<sigma_points.size(); i++)
    {
        nx_sigma_pt = sigma_points.at(i).head(n_x);
        v_i = sigma_points.at(i).tail(n_v);
        utils::activeModel(nx_sigma_pt, activeIdx, 
                nState, nModel, set_of_gcs);
        dynamics[activeIdx]->propagateState(nx_sigma_pt, u, x_i);
        x_i += v_i;
        new_sigma_pt << x_i; //
        predicted_sigma_pts.push_back(new_sigma_pt);
    }

    // Recombine points
    double W_m_0 = labda/(n_x + n_v + labda);
    double W_m_i = 0.5/(n_x + n_v + labda);
    double W_c_0 = (labda/(n_x + n_v + labda))+ 1-pow(alpha,2)+beta;
    double W_c_i = W_m_i;

    x_bar = W_m_0 * predicted_sigma_pts.at(0);
    for(int i=1; i<predicted_sigma_pts.size(); i++)
        x_bar += W_m_i * predicted_sigma_pts.at(i);

    P_bar = W_c_0 * (predicted_sigma_pts.at(0) - x_bar)*
                    (predicted_sigma_pts.at(0) - x_bar).transpose();
    for(int i=1; i<predicted_sigma_pts.size(); i++)
        P_bar += W_c_i * (predicted_sigma_pts.at(i) - x_bar)*
                    (predicted_sigma_pts.at(i) - x_bar).transpose();
}

void UKF::update(Eigen::VectorXd x_bar, 
        Eigen::MatrixXd P_bar, Eigen::VectorXd z, 
        Eigen::VectorXd& x_new, Eigen::MatrixXd& P_new)
{
    Eigen::VectorXd x_aug_bar = Eigen::VectorXd(n_x+n_z);
    Eigen::MatrixXd P_aug_bar = Eigen::MatrixXd::Zero(n_x+n_z, n_x+n_z);
    Eigen::VectorXd z_bar = Eigen::VectorXd::Zero(n_z);
    Eigen::MatrixXd P_zz = Eigen::MatrixXd::Zero(n_z, n_z);
    Eigen::MatrixXd P_xz = Eigen::MatrixXd::Zero(n_x, n_z);
    
    Eigen::VectorXd w = Eigen::VectorXd::Zero(n_z); // Zero Mean Noise Vector
    x_aug_bar << x_bar, w;
    
    int activeIdx;    
    utils::activeModel(x_bar, activeIdx, nState, nModel, set_of_gcs);
    P_aug_bar.block(0,0, n_x, n_x) = P_bar;
    P_aug_bar.block(n_x, n_x, n_z, n_z) = dynamics[activeIdx]->W;

    double labda = pow(alpha, 2)*(n_x+n_z+kappa) - (n_x + n_z);
    double delta_factor = sqrt(n_x+n_z+labda);
    std::vector<Eigen::VectorXd> sigma_points;
    findSigmaPts(x_aug_bar, P_aug_bar, labda, delta_factor, sigma_points);

    // Calculate observation estimate for each Sigma Pt
    std::vector<Eigen::VectorXd> measurement_sigma_pts;
    Eigen::VectorXd nx_sigma_pt(n_x), z_i(n_x), w_i(n_z), new_sigma_pt(n_x);
    
    for(int i=0; i<sigma_points.size(); i++)
    {
        nx_sigma_pt = sigma_points.at(i).head(n_x);
        w_i = sigma_points.at(i).tail(n_z);
        utils::activeModel(nx_sigma_pt, activeIdx, nState, nModel, set_of_gcs);
        dynamics[activeIdx]->getObservationNoNoise(nx_sigma_pt, z_i);
        z_i += w_i;
        new_sigma_pt << z_i;
        measurement_sigma_pts.push_back(new_sigma_pt);
    }

    // Recombine points
    double W_m_0 = labda/(n_x + n_z + labda);
    double W_m_i = 0.5/(n_x + n_z + labda);
    double W_c_0 = (labda/(n_x + n_z + labda))+ 1-pow(alpha,2)+beta;
    double W_c_i = W_m_i;

    // Mean Calculation
    z_bar = W_m_0 * measurement_sigma_pts.at(0);
    for(int i=1; i<measurement_sigma_pts.size(); i++)
        z_bar += W_m_i * measurement_sigma_pts.at(i);
    
    // Covariance of observations
    P_zz = W_c_0 * (measurement_sigma_pts.at(0) - z_bar)*
                    (measurement_sigma_pts.at(0) - z_bar).transpose();

    // Cross Covariance of observations and predictions
    P_xz = W_c_0 * (sigma_points.at(0).head(n_x) - x_bar)*
                    (measurement_sigma_pts.at(0) - z_bar).transpose();
    
    for(int i=1; i<measurement_sigma_pts.size(); i++)
    {
        P_zz += W_c_i * (measurement_sigma_pts.at(i) - z_bar)*
                    (measurement_sigma_pts.at(i) - z_bar).transpose();
        P_xz += W_c_i * (sigma_points.at(i).head(n_x) - x_bar)*
                    (measurement_sigma_pts.at(i) - z_bar).transpose();
    }

    // LMMSE Update
    Eigen::MatrixXd K = P_xz * P_zz.inverse();

    x_new = x_bar + K*(z - z_bar);
    P_new = P_bar - K*P_zz*K.transpose();
}
} // end namespace

