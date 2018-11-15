#ifndef BELIEF_EVOLUTION_H_
#define BELIEF_EVOLUTION_H_

#include "problem_definition.h"
#include "simulator.h"

using namespace std;

class BeliefEvolution : public ProblemDefinition
{
	public:
        // Parameters
        int nSamples, ds_res_loop_count_;
        double goal_threshold;
		double eps1 = 0.05;
        
        // Simulator
        std::shared_ptr<Simulator> simulator;
    
        // Data Storage 
        int activeIdx;
		std::map<int, Eigen::VectorXd> mu_set_;
		std::map<int, Eigen::MatrixXd> cov_set_;
		Eigen::VectorXd wts;

		/* Member Functions*/
		BeliefEvolution();
        ~BeliefEvolution();

        // Define Goal guard Conditions
        void setGoalGC(Eigen::Map<Eigen::VectorXd>&);

        // Define Active Model
        // void activeModel(Eigen::Map<Eigen::VectorXd>&, int&);

        void getMatrices(int&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::MatrixXd>&);

        /* Belief Evolution Functions*/
		void fastWts(Eigen::VectorXd&, Eigen::MatrixXd&, Eigen::VectorXd&);

		void fastWtsMapped(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&);

		void simpleWts(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, int, Eigen::Map<Eigen::VectorXd>&);

		void observationUpdate(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&, int);

		void beliefUpdatePlanning(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&, int);

		void predictionStochastic(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&, int&);

		void predictionDeterministic(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::VectorXd>&, int);

		/* Utils*/
        bool isPDMapped(Eigen::Map<Eigen::MatrixXd>&);
        void nearestPDMapped(Eigen::Map<Eigen::MatrixXd>&, Eigen::Map<Eigen::MatrixXd>&);
        void increaseIntegrationResolution(float, float);

        void simulateOneStep(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&);
};

#endif // BELIEF_EVOLUTION_H_

