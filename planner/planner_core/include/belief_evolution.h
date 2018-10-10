#include <iostream>
#include <iomanip>   
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions> // For changing resolution of ds dynamics
#include <map>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "eigenmvn.h"
#include "omp.h"
#include <cmath>
#include "utils.h"

using namespace std;


class Simulator
{
    public:
        int nState;
        int nInput;
        int nOutput;
        int nModel;
        int activeIdx;
        // Dynamics* dyna;

        // Dynamics
        // Dynamics* dyna;
        std::map<int, Eigen::MatrixXd> mA; // State Transition Matrix
        std::map<int, Eigen::MatrixXd> mB; // Control Matrix
        std::map<int, Eigen::MatrixXd> mC; // OBservation Matrix 
        std::map<int, Eigen::MatrixXd> mV; // Process Noise
        std::map<int, Eigen::MatrixXd> mW; // Observation Noise
        
        std::map<int, std::vector<Guards> > set_of_gcs;


        /*Declaring Memeber Functions */
        Simulator(){};

        /*Simulation functions*/        
        void initialize(int, int, int, int, std::map<int, Eigen::MatrixXd>&, std::map<int, Eigen::MatrixXd>&, std::map<int, Eigen::MatrixXd>&, std::map<int, Eigen::MatrixXd>&, std::map<int, Eigen::MatrixXd>&, std::map<int, std::vector<Guards> >&);
        void simulation(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&);
        void observation(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&);
};


class BeliefEvolution
{
	public:
		int nState;
		int nInput;
		int nOutput;
		int nModel;

		double eps1 = 0.05;

		//Utils
		Eigen::MatrixXd I_mat_;	

		// Dynamics
		// Dynamics* dyna;
		std::map<int, Eigen::MatrixXd> mA; // State Transition Matrix
        std::map<int, Eigen::MatrixXd> mB; // Control Matrix
        std::map<int, Eigen::MatrixXd> mC; // OBservation Matrix 
        std::map<int, Eigen::MatrixXd> mV; // Process Noise
        std::map<int, Eigen::MatrixXd> mW; // Observation Noise

        // Eigen::MatrixXd W;
        std::map<int, std::vector<Guards> > set_of_gcs;

		int activeIdx;
		std::map<int, Eigen::VectorXd> mu_set_;
		std::map<int, Eigen::MatrixXd> cov_set_;
		Eigen::VectorXd wts;

        // Parameters
        int nSamples;
        int ds_res_loop_count_ = 1;
        // Eigen::Map<Eigen::VectorXd> goal;


        // Simulator
        Simulator* simulator;

		/* Member Functions*/
		BeliefEvolution();

		/* Dynamics Definition */
		// Define Matrices for Each of the dynamics
        void setMatrices();

        // Define guard Conditions
        void setGC(Eigen::Map<Eigen::VectorXd>&);

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

        void increase_resolution_of_ds_dynamics(float, float, Eigen::MatrixXd&, Eigen::MatrixXd&, Eigen::MatrixXd&, Eigen::MatrixXd&, Eigen::MatrixXd&, Eigen::MatrixXd&, Eigen::MatrixXd&, Eigen::MatrixXd&);

        void change_ds_resolution_local(std::map<int, Eigen::MatrixXd>&, std::map<int, Eigen::MatrixXd>&, std::map<int, Eigen::MatrixXd>&, std::map<int, Eigen::MatrixXd>&);

        void simulate_oneStep(Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&, Eigen::Map<Eigen::VectorXd>&);
};

