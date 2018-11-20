#include "../include/simulator.h"

void Simulator::initialize(int nDS)
{   
    nModel = nDS - 1;

    // Hybrid Dynamics Definition
    utils::reduce_map_size(set_of_gcs, set_of_gcs, 1);
}


void Simulator::simulateOneStep(Eigen::Map<Eigen::VectorXd>& x, Eigen::Map<Eigen::VectorXd>& u, Eigen::Map<Eigen::VectorXd>& x_new, Eigen::Map<Eigen::VectorXd>& z_new)
{
    // Set activeIdx
    Eigen::VectorXd x1, z1, proc_noise_;

    // Numerical Integration
    float res = 0.0001;
    x1 = x;

    for(int i=0; i<int(1/res); i++)
    {
        utils::activeModel(x1, activeIdx, nState, nModel, set_of_gcs);

        /* Allowing motion if moving away from the wall*/
        if(activeIdx == 1 && u(0) > 0) 
            activeIdx = 0;
        else if(activeIdx == 2 && u(1) < 0)
            activeIdx = 0;
        else if(activeIdx == 3)
            {
                if(u(0)>0 && u(1) < 0)
                    activeIdx = 0;
                else if(u(0)>0)
                    activeIdx = 2;
                else if(u(1) < 0)
                    activeIdx = 1;
            }
       
        dynamics[activeIdx]->propagateState(x1, u*res, x1);

        // System Process Noise
        // utils::nearestPD(V, covar); // Now covar is a positive definite matrix
        // Eigen::EigenMultivariateNormal<double> normX_cholesk(zero_mean, covar);
        // proc_noise_ = normX_cholesk.samples(1);

        // x += proc_noise_;
    }
    
    std::cout << "x_new:" << x1;

    // Recieve observations
    utils::activeModel(x1, activeIdx, nState, nModel, set_of_gcs);
    dynamics[activeIdx]->getObservation(x1, z1);
    
    // Update outputs
    x_new = x1;
    z_new = z1;
}

