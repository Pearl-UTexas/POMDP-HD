#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "filter.h"

namespace filters
{
class KalmanFilter : public Filter
{
    public:
        // Methods
        KalmanFilter();
        void prediction(Eigen::VectorXd, Eigen::MatrixXd, 
                Eigen::VectorXd,  
                Eigen::VectorXd&, Eigen::MatrixXd&);
        void update(Eigen::VectorXd, Eigen::MatrixXd, 
                Eigen::VectorXd,  
                Eigen::VectorXd&, Eigen::MatrixXd&);

};
} // end filters namespace

#endif // KALMAN_FILTER_H_
