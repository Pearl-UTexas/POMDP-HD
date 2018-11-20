#ifndef DYNAMICS_H_
#define DYNAMICS_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../eigenmvn.h"
#include "../utils.h"

namespace dynamics{

class Dynamics{
public:
    // State Space Parameters
    int nState;
    int nInput;
    int nOutput;
    
    // params
    void printModelParams();
    void setModelParams(int n_x, int n_u, int n_y);

    // Functions
    void propagateState(Eigen::VectorXd, 
            Eigen::VectorXd, Eigen::VectorXd&);
    void getObservation(Eigen::VectorXd, Eigen::VectorXd&);
};

}

#endif // DYNAMICS_H_
