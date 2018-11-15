#ifndef LINEAR_DYNAMICS_H_
#define LINEAR_DYNAMICS_H_

#include <iostream>
#include <eigen3/Eigen/Dense>
#include "eigenmvn.h"
#include "utils.h"

namespace dynamics{

class LinearDynamics{
public:
    // State Space Parameters
    int nState;
    int nInput;
    int nOutput;

    // System Matrices
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd C;
    Eigen::MatrixXd V;
    Eigen::MatrixXd W;
    
    LinearDynamics(int n_x, int n_u, int n_y){
        nState = n_x;
        nInput = n_u;
        nOutput = n_y;      

        A = Eigen::MatrixXd::Identity(nState, nState);
        B = Eigen::MatrixXd::Identity(nInput, nInput);
        C = Eigen::MatrixXd::Identity(nOutput, nOutput);
        V = Eigen::MatrixXd::Identity(nState, nState);
        W = Eigen::MatrixXd::Identity(nOutput, nOutput);
    }

    ~LinearDynamics(){
        /*
        delete A; // State Transition Matrix
        delete B; // Control Matrix
        delete C; // Observation Matrix
        delete V; // Process Noise
        delete W; // Observation Noise
        */
    }
    
    // params
    virtual void printModelParams();
    virtual void setModelParams(int n_x, int n_u, int n_y);

    // Functions
    virtual void propagateState(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd&);
    virtual void propagateStateWithCov(Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd&, Eigen::MatrixXd&);

    virtual void getObservation(Eigen::VectorXd, Eigen::VectorXd&);
    virtual void getObservationCov(Eigen::MatrixXd, Eigen::MatrixXd&);
    virtual void getObservationNoNoise(Eigen::VectorXd, Eigen::VectorXd&);
    virtual void getObservationWithCov(Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd&, Eigen::MatrixXd&);
};

}

#endif // LINEAR_DYNAMICS_H_
