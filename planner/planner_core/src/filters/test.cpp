#include "../../include/filters/ukf.h"

int main()
{
    filters::UKF ukf;
    
    Eigen::VectorXd x(2), u(2), z(2);
    x << 1., 1.;
    u << 1., 1.; 
    z << 1.5, 1.75;      

    Eigen::MatrixXd P(2,2);
    P << 5.0, 0., 0., 5.0; 

    Eigen::VectorXd x_bar(2), x_new(2);
    Eigen::MatrixXd P_bar(2,2), P_new(2,2);
    
    ukf.prediction(x, P, u, x_bar, P_bar);
    std::cout << "Predicted x_bar: " << x_bar.transpose() << std::endl;
    std::cout << "Predicted P_bar: \n" << P_bar << std::endl;

    ukf.update(x_bar, P_bar, z, x_new, P_new);
    std::cout << "Updated x_new: " << x_new.transpose() << std::endl;
    std::cout << "Updated P_new: \n" << P_new << std::endl;
}
