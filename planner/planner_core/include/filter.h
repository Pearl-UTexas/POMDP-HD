#ifndef FILTER_H_
#define FILTER_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include "problem_definition.h"

namespace filters
{
class Filter : public ProblemDefinition
{
    public:
        // Methods
        void prediction(Eigen::VectorXd, Eigen::MatrixXd, 
                Eigen::VectorXd,  
                Eigen::VectorXd&, Eigen::MatrixXd&);
        void update(Eigen::VectorXd, Eigen::MatrixXd, 
                Eigen::VectorXd,  
                Eigen::VectorXd&, Eigen::MatrixXd&);

};
} // end filters namespace

#endif // FILTER_H_
