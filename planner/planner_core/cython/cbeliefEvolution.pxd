# file: cbeliefEvolution.pxd
from eigency.core cimport *
from libcpp cimport bool
import numpy as np


cdef extern from "../include/belief_evolution.h":
    cppclass BeliefEvolution:
        BeliefEvolution() 

        #Attributes
        int nState
        int nInput
        int nOutput
        int nModel
        int nSamples
        int ds_res_loop_count_


        # Map[MatrixXd] W;
        Map[VectorXd] wts;
        # Map[VectorXd] goal;

        # Dynamics* dyna;
        # Simulator* simulator;

        # void fastWts(Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &);
        # void simpleWts(Map[VectorXd] &, Map[MatrixXd] &, int, Map[VectorXd] &);
        # void setDynamics(Dynamics*)

        void setGC(Map[VectorXd]&)
        # void activeModel(Map[VectorXd]&, int&)
        void getMatrices(int&, Map[MatrixXd]&, Map[MatrixXd]&, Map[MatrixXd]&, Map[MatrixXd]&, Map[MatrixXd]&)
        void nearestPDMapped(Map[MatrixXd]&, Map[MatrixXd]&)
        bool isPDMapped(Map[MatrixXd]&)
        void fastWtsMapped(Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &)
        int observationUpdate(Map[VectorXd] &, Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &, int&)
        int beliefUpdatePlanning(Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &, Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &, int&)
        int predictionStochastic(Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &, Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &, int&)
        int predictionDeterministic(Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &, Map[VectorXd] &, Map[MatrixXd] &, Map[VectorXd] &, int&)
        void simulate_oneStep(Map[VectorXd] &, Map[VectorXd] &, Map[VectorXd] &, Map[VectorXd] &)

