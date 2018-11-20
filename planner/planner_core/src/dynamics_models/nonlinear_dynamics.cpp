#include "../../include/dynamics_models/nonlinear_dynamics.h"

namespace dynamics{

void NonlinearDynamics::printModelParams()
{
    std::string sep = "\n----------------------------------------\n";
    
    std::cout << "Nonlinear Dynamics Model Parameters: \n";
    std::cout << "Number of States: " << nState << "\n";
    std::cout << "Number of Input : " << nInput << "\n";
    std::cout << "Number of Output: " << nOutput << "\n";
    std::cout << sep;
}

void NonlinearDynamics::setModelParams(int n_x, int n_u, int n_y){
        int nState = n_x;
        int nInput = n_u;
        int nOutput = n_y;      
}
} // end namespace
