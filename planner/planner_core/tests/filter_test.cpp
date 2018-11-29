#include "../include/filters/ukf.h"
#include "../include/filters/kalman_filter.h"

int main()
{
    filters::UKF ukf;
    filters::KalmanFilter kf;
    
    int steps = 100;    
    int activeIdx;
    Eigen::VectorXd x_sim_init(2), x_sim(2), x_init(2);
    Eigen::VectorXd x_ukf(2), x_kf(2), u(2), z(2);
    Eigen::MatrixXd P_init(2,2), P_kf(2,2), P_ukf(2,2);
     
    x_sim_init << 1., -2.;
    u << 1., 1.; 
    x_init << -1.7854, -0.5098;
    P_init << 10.0, 0., 0., 10.0; 

    Eigen::VectorXd x_bar(2);
    Eigen::MatrixXd P_bar(2,2);
    x_sim = x_sim_init;
    x_kf = x_init;
    x_ukf = x_init;
    P_kf = P_init;
    P_ukf = P_init;

     
    for(int i=0; i<steps; i++)
    {   
        // Generate Observation
        utils::activeModel(x_sim, activeIdx, 
                ukf.nState, ukf.nModel, ukf.set_of_gcs);
        ukf.dynamics[activeIdx]->propagateState(x_sim, u, x_sim); // simulation
        ukf.dynamics[activeIdx]->getObservation(x_sim, z);

        // KF
        kf.prediction(x_kf, P_kf, u, x_bar, P_bar);
        kf.update(x_bar, P_bar, z, x_kf, P_kf);
        
        // UKF
        ukf.prediction(x_ukf, P_ukf, u, x_bar, P_bar);
        ukf.update(x_bar, P_bar, z, x_ukf, P_ukf);

    }
    
    std::cout << "\n\n\nFinal : " << std::endl;
    std::cout << "Actual x : " << x_sim.transpose() << std::endl;
    
    std::cout << "\n####### KALMAN FILTER #######" << std::endl;
    std::cout << "Belief x : " << x_kf.transpose() << std::endl;
    std::cout << " P: \n" << P_kf << std::endl;

    std::cout << "\n####### UKF #######" << std::endl;
    std::cout << "Belief x : " << x_ukf.transpose() << std::endl;
    std::cout << "P: \n" << P_ukf << std::endl;

    
    
    /*
    // Kalman Filter
    std::cout << "\n\n\n####### KALMAN FILTER #######" << std::endl;
    x_sim = x_sim_init;
    x = x_init;
    P = P_init;

    for(int i=0; i<steps; i++)
    {   
        utils::activeModel(x_sim, activeIdx,
                kf.nState, kf.nModel, kf.set_of_gcs);
        kf.dynamics[activeIdx]->propagateState(x_sim, u0, x_sim); // simulation
        kf.dynamics[activeIdx]->getObservation(x_sim, z);
        // x_sim = x_sim_new;
        
        kf.prediction(x, P, u0, x_bar, P_bar);
        kf.update(x_bar, P_bar, z, x, P);
      
        std::cout << "\nAt t = " << i+1 << std::endl;
        std::cout << "Actual x : " << x_sim.transpose() << std::endl;
        std::cout << "z : " << z.transpose() << std::endl;
        std::cout << "Belief x : " << x_new.transpose() << std::endl;
        std::cout << " P: \n" << P_new << std::endl;
      
    }
    
    std::cout << "Final : " << std::endl;
    std::cout << "Actual x : " << x_sim.transpose() << std::endl;
    std::cout << "Belief x : " << x.transpose() << std::endl;
    std::cout << " P: \n" << P << std::endl;
    // std::cout << "z_sum : " << (z_sum/steps).transpose() << std::endl;
    */
}
