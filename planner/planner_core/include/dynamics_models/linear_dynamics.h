#ifndef LINEAR_DYNAMICS_H_
#define LINEAR_DYNAMICS_H_

#include "dynamics.h"

namespace dynamics{

class LinearDynamics: public Dynamics{
public:
    // System Matrices
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd C;
    Eigen::MatrixXd V;
    Eigen::MatrixXd W;
    
    LinearDynamics(int nx, int nu, int nz){
        nState = nx;
        nInput = nu;
        nOutput = nz;

        A = Eigen::MatrixXd::Identity(nState, nState);
        B = Eigen::MatrixXd::Identity(nInput, nInput);
        C = Eigen::MatrixXd::Identity(nOutput, nOutput);
        V = Eigen::MatrixXd::Identity(nState, nState);
        W = Eigen::MatrixXd::Identity(nOutput, nOutput);
    }

   ~LinearDynamics(){;}
   
    // params
    void printModelParams();
    void setModelParams(int n_x, int n_u, int n_y);

    // Functions
    void propagateState(Eigen::VectorXd, 
            Eigen::VectorXd, Eigen::VectorXd&);
    void getObservation(Eigen::VectorXd, Eigen::VectorXd&);

    void propagateStateWithCov(Eigen::VectorXd, 
            Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd&, 
            Eigen::MatrixXd&);
    void getObservationCov(Eigen::MatrixXd, Eigen::MatrixXd&);
    void getObservationNoNoise(Eigen::VectorXd, Eigen::VectorXd&);
    void getObservationWithCov(Eigen::VectorXd, 
            Eigen::MatrixXd, Eigen::VectorXd&, Eigen::MatrixXd&);
};
}
#endif // LINEAR_DYNAMICS_H_
