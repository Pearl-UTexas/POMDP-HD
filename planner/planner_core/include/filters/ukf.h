#ifndef UKF_H_
#define UKF_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include "../problem_definition.h"

namespace filters
{
class UKF : public ProblemDefinition
{
    public:
        // Parameters
        int n_x;
        int n_v; // currently we define it same as n_x
        int n_z;
        double alpha;
        double beta;
        double kappa;

        // Dynamics Model
        // hat = Predicted; bar = Updated 
        Eigen::VectorXd x_hat, v, x_bar, w; 
        Eigen::MatrixXd P_hat, Q, P_bar, R;

        // UKF Calculations
        Eigen::VectorXd x_aug_hat, x_aug_bar, z_bar;
        Eigen::MatrixXd P_aug_hat, P_aug_bar, 
            S_x_hat, S_x_bar, P_zz, P_xz;    
        std::vector<Eigen::VectorXd> sigma_points;

        // Methods
        UKF();
        void findSigmaPts(Eigen::VectorXd, Eigen::MatrixXd, double, double);
        void prediction(Eigen::VectorXd, Eigen::MatrixXd, 
                Eigen::VectorXd,  
                Eigen::VectorXd&, Eigen::MatrixXd&);
        void update(Eigen::VectorXd, Eigen::MatrixXd, 
                Eigen::VectorXd,  
                Eigen::VectorXd&, Eigen::MatrixXd&);

};
} // end filters namespace

#endif // UKF_H_
