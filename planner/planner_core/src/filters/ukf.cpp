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
    alpha = 1e-1;
    beta = 2;
    kappa = 0;
    
    // Initialize Matrices
    x_aug_hat = Eigen::VectorXd(n_x+n_v);
    P_aug_hat = Eigen::MatrixXd::Zero(n_x+n_v, n_x+n_v);
    x_aug_bar = Eigen::VectorXd(n_x+n_z);
    P_aug_bar = Eigen::MatrixXd::Zero(n_x+n_z, n_x+n_z);
    z_bar = Eigen::VectorXd(n_x+n_z);
    P_zz = Eigen::MatrixXd(n_x+n_z, n_x+n_z);
    P_xz = Eigen::MatrixXd(n_x, n_x+n_z);
}

void UKF::findSigmaPts(Eigen::VectorXd x_0, 
        Eigen::MatrixXd sqrt_mat, double labda, 
        double delta_factor)
{
    sigma_points.clear();
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

void UKF::prediction(Eigen::VectorXd x_hat, Eigen::MatrixXd P_hat, 
        Eigen::VectorXd u, 
        Eigen::VectorXd& x_bar, Eigen::MatrixXd& P_bar)
{
    v = Eigen::VectorXd::Zero(n_v);
    x_aug_hat << x_hat, v;
    P_aug_hat.block(0,0, n_x, n_x) = P_hat;
    
    int activeIdx;
    utils::activeModel(x_hat, activeIdx, nState, nModel, set_of_gcs);
    P_aug_hat.block(n_x, n_x, n_v, n_v) = dynamics[activeIdx]->V;
    S_x_hat = P_aug_hat.llt().matrixL().transpose();

    double labda = pow(alpha, 2)*(n_x+n_v+kappa) - (n_x + n_v);
    double delta_factor = sqrt(n_x+n_v+labda);
    findSigmaPts(x_aug_hat, S_x_hat, labda, delta_factor);

    // Propagate Sigma Pts through Dynamics
    std::vector<Eigen::VectorXd> predicted_sigma_pts;
    Eigen::VectorXd nx_sigma_pt(n_x), x_i(n_x), new_sigma_pt(n_x+n_v);
    for(int i=0; i<sigma_points.size(); i++)
    {
        nx_sigma_pt = sigma_points.at(i).head(n_x);
        utils::activeModel(nx_sigma_pt, activeIdx, 
                nState, nModel, set_of_gcs);
        dynamics[activeIdx]->propagateState(nx_sigma_pt, u, x_i);
        new_sigma_pt << x_i, sigma_points.at(i).tail(n_v);
        predicted_sigma_pts.push_back(new_sigma_pt);
    }

    // Recombine points
    double W_m_0 = labda/(n_x + n_v + labda);
    double W_m_i = 0.5/(n_x + n_v + labda);
    double W_c_0 = (labda/(n_x + n_v + labda))+ 1-pow(alpha,2)+beta;
    double W_c_i = W_m_i;

    Eigen::VectorXd x_aug_dummy(n_x+n_v);
    Eigen::MatrixXd P_aug_dummy(n_x+n_v, n_x+n_v);

    x_aug_dummy = W_m_0 * predicted_sigma_pts.at(0);
    for(int i=1; i<predicted_sigma_pts.size(); i++)
        x_aug_dummy += W_m_i * predicted_sigma_pts.at(i);
    
    P_aug_dummy = W_c_0 * (predicted_sigma_pts.at(0) - x_aug_dummy)*
                    (predicted_sigma_pts.at(0) - x_aug_dummy).transpose();
    for(int i=1; i<predicted_sigma_pts.size(); i++)
        P_aug_dummy += W_c_i * (predicted_sigma_pts.at(i) - x_aug_dummy)*
                    (predicted_sigma_pts.at(i) - x_aug_dummy).transpose();
    
    x_bar = x_aug_dummy.head(n_x);
    P_bar = P_aug_dummy.block(0,0, n_x, n_x);
}

void UKF::update(Eigen::VectorXd x_bar, Eigen::MatrixXd P_bar, 
        Eigen::VectorXd z, 
        Eigen::VectorXd& x_new, Eigen::MatrixXd& P_new)
{
    w = Eigen::VectorXd::Zero(n_z);
    x_aug_bar << x_bar, w;
    P_aug_bar.block(0,0, n_x, n_x) = P_bar;
    
    int activeIdx;
    utils::activeModel(x_bar, activeIdx, nState, nModel, set_of_gcs);
    P_aug_bar.block(n_x, n_x, n_z, n_z) = dynamics[activeIdx]->W;
    S_x_bar = P_aug_bar.llt().matrixL().transpose();

    double labda = pow(alpha, 2)*(n_x+n_z+kappa) - (n_x + n_z);
    double delta_factor = sqrt(n_x+n_z+labda);
    findSigmaPts(x_aug_bar, S_x_bar, labda, delta_factor);
    
    // Caculate observation estimate for each Sigma Pt
    std::vector<Eigen::VectorXd> updated_sigma_pts;
    Eigen::VectorXd nx_sigma_pt(n_x), z_i(n_x), new_sigma_pt(n_x+n_z);
    for(int i=0; i<sigma_points.size(); i++)
    {
        nx_sigma_pt = sigma_points.at(i).head(n_x);
        utils::activeModel(nx_sigma_pt, activeIdx, 
                nState, nModel, set_of_gcs);
        dynamics[activeIdx]->getObservation(nx_sigma_pt, z_i);
        new_sigma_pt << z_i, sigma_points.at(i).tail(n_z);
        updated_sigma_pts.push_back(new_sigma_pt);
    }

    // Recombine points
    double W_m_0 = labda/(n_x + n_z + labda);
    double W_m_i = 0.5/(n_x + n_z + labda);
    double W_c_0 = (labda/(n_x + n_z + labda))+ 1-pow(alpha,2)+beta;
    double W_c_i = W_m_i;
    
    z_bar = W_m_0 * updated_sigma_pts.at(0);
    for(int i=1; i<updated_sigma_pts.size(); i++)
        z_bar += W_m_i * updated_sigma_pts.at(i);
    
    P_zz = W_c_0 * (updated_sigma_pts.at(0) - z_bar)*
                    (updated_sigma_pts.at(0) - z_bar).transpose();
    P_xz = W_c_0 * (sigma_points.at(0).head(n_x) - x_bar)*
                    (updated_sigma_pts.at(0) - z_bar).transpose();
    
    for(int i=1; i<updated_sigma_pts.size(); i++)
    {
        P_zz += W_c_i * (updated_sigma_pts.at(i) - z_bar)*
                    (updated_sigma_pts.at(i) - z_bar).transpose();
        P_xz += W_c_i * (sigma_points.at(i).head(n_x) - x_bar)*
                    (updated_sigma_pts.at(i) - z_bar).transpose();
    }

    // LMMSE Update
    Eigen::MatrixXd P_zz_inv = P_zz.inverse();
    Eigen::VectorXd z_aug(n_x+n_z);
    z_aug << z, w; // Not predicting for noise in observation

    x_new = x_bar + P_xz*P_zz_inv*(z_aug - z_bar);
    P_new = P_bar - P_xz*P_zz_inv*P_xz.transpose();
}

} // end namespace

