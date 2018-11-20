#ifndef NONLINEAR_DYNAMICS_H_
#define NONLINEAR_DYNAMICS_H_

#include "../dynamics.h"

namespace dynamics{

class NonlinearDynamics: public Dynamics{
    public:
       NonlinearDynamics(int n_x, int n_u, int n_y){
        nState = n_x;
        nInput = n_u;
        nOutput = n_y;      
    }

    ~NonlinearDynamics(){}
    // params
    void printModelParams();
    void setModelParams(int n_x, int n_u, int n_y);

    // Functions
    void propagateState(Eigen::VectorXd, 
            Eigen::VectorXd, Eigen::VectorXd&);
    void getObservation(Eigen::VectorXd, Eigen::VectorXd&);
    
    void propagateStateWithCov(Eigen::VectorXd, 
            Eigen::MatrixXd, Eigen::VectorXd, 
            Eigne::VectorXd, Eigen::MatrixXd);
};
}

#endif // NONLINEAR_DYNAMICS_H_
