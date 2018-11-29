#ifndef UKF_H_
#define UKF_H_

#include "../filter.h"

namespace filters
{
class UKF : public Filter
{
    public:
        // Parameters
        int n_x;
        int n_v;
        int n_z;
        double alpha;
        double beta;
        double kappa;

        // Methods
        UKF();
        void findSigmaPts(Eigen::VectorXd, Eigen::MatrixXd, 
                double, double, std::vector<Eigen::VectorXd>&);
        void prediction(Eigen::VectorXd, Eigen::MatrixXd, 
                Eigen::VectorXd,  
                Eigen::VectorXd&, Eigen::MatrixXd&);
        void update(Eigen::VectorXd, Eigen::MatrixXd, 
                Eigen::VectorXd,  
                Eigen::VectorXd&, Eigen::MatrixXd&);

};
} // end filters namespace

#endif // UKF_H_
