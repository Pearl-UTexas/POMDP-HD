#ifndef SIMULATOR_H_
#define SIMULATOR_H_

#include "problem_definition.h"

using namespace std;

class Simulator : public ProblemDefinition
{
    public:
        int activeIdx; 

        /*Declaring Memeber Functions */
        Simulator(){ ProblemDefinition();};

        /*Simulation functions*/        
        void initialize(int);
        void simulateOneStep(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&);
};

#endif // SIMULATOR_H_

