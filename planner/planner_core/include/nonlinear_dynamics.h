#ifndef NONLINEAR_DYNAMICS_H_
#define NONLINEAR_DYNAMICS_H_

#include <iostream>

namespace dynamics{

class NonlinearDynamics{
    public:
       NonlinearDynamics(int n_x, int n_u, int n_y){
        int nState = n_x;
        int nInput = n_u;
        int nOutput = n_y;      
    }

    ~NonlinearDynamics(){}
    
    // params
    virtual void printModelParams();
    virtual void setModelParms(int n_x, int n_u, int n_y);

    // Functions
    virtual void propagateState(Eigen::VectorXd, Eigen::VectorXd, Eigne::VectorXd);
    virtual void propagateStateWithCov(Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigne::VectorXd, Eigen::MatrixXd);
    
    virtual void getObservation(Eigen::VectorXd, Eigen::VectorXd);
};

}

#endif // NONLINEAR_DYNAMICS_H_
