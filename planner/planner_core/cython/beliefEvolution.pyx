cimport cbeliefEvolution
import numpy as np
cimport numpy as np
from eigency.core cimport *

cdef class BeliefEvolution:
    cdef cbeliefEvolution.BeliefEvolution* cobj

    def __cinit__(self):
        self.cobj = new cbeliefEvolution.BeliefEvolution()
        # self.simulator = self.

        if self.cobj is NULL:
            raise MemoryError('Not enough memory.')

    def __dealloc__(self):
        del self.cobj

    property nState:
        def __get__(self):
            return self.cobj.nState
        def __set__(self, int var):
            self.cobj.nState = var

    property nInput:
        def __get__(self):
            return self.cobj.nInput
        def __set__(self, int var):
            self.cobj.nInput = var

    property nOutput:
        def __get__(self):
            return self.cobj.nOutput
        def __set__(self, int var):
            self.cobj.nOutput = var            

    property nModel:
        def __get__(self):
            return self.cobj.nModel
        def __set__(self, int var):
            self.cobj.nModel = var

    property wts:
        def __get__(self):
            return ndarray(self.cobj.wts)
        def __set__(self, np.ndarray val):
            self.cobj.wts = Map[VectorXd](val)

    property nSamples:
        def __get__(self):
            return self.cobj.nSamples
        def __set__(self, int var):
            self.cobj.nSamples = var 

    property goal_threshold:
        def __get__(self):
            return self.cobj.goal_threshold
        def __set__(self, double var):
            self.cobj.goal_threshold = var 
    
    property ds_res_loop_count_:
        def __get__(self):
            return self.cobj.ds_res_loop_count_
        def __set__(self, int var):
            self.cobj.ds_res_loop_count_ = var 

    def fastWtsMapped(self, np.ndarray mu, np.ndarray cov, np.ndarray wts_new):
        return self.cobj.fastWtsMapped(Map[VectorXd](mu), Map[MatrixXd] (cov), Map[VectorXd](wts_new))

    def setGoalGC(self, np.ndarray goal):
        return self.cobj.setGoalGC(Map[VectorXd](goal))

    def getMatrices(self, int idx, np.ndarray A, np.ndarray B, np.ndarray C, np.ndarray V, np.ndarray W):
        return self.cobj.getMatrices(int(idx), Map[MatrixXd](A), Map[MatrixXd](B), Map[MatrixXd](C), Map[MatrixXd](V), Map[MatrixXd](W))

    def isPDMapped(self, np.ndarray A):
        return self.cobj.isPDMapped(Map[MatrixXd](A))

    def nearestPDMapped(self, np.ndarray A, np.ndarray A_hat):
        return self.cobj.nearestPDMapped(Map[MatrixXd](A), Map[MatrixXd](A_hat))

    def observationUpdate(self, np.ndarray z, np.ndarray mu_new, np.ndarray cov_new, np.ndarray wts_new, int ds_new):
        self.cobj.observationUpdate(Map[VectorXd](z), Map[VectorXd](mu_new), Map[MatrixXd](cov_new), Map[VectorXd](wts_new), int(ds_new))
        return ds_new

    def beliefUpdatePlanning(self, np.ndarray mu, np.ndarray cov, np.ndarray u, np.ndarray mu_new, np.ndarray cov_new, np.ndarray wts_new, int ds_new):
        self.cobj.beliefUpdatePlanning(Map[VectorXd](mu), Map[MatrixXd](cov), Map[VectorXd](u), Map[VectorXd](mu_new), Map[MatrixXd](cov_new), Map[VectorXd](wts_new), int(ds_new))
        return ds_new

    def predictionDeterministic(self, np.ndarray mu, np.ndarray cov, np.ndarray u, np.ndarray mu_new, np.ndarray cov_new, np.ndarray wts_new, int ds_new):
        self.cobj.predictionDeterministic(Map[VectorXd](mu), Map[MatrixXd](cov), Map[VectorXd](u), Map[VectorXd](mu_new), Map[MatrixXd](cov_new), Map[VectorXd](wts_new), int(ds_new))
        return ds_new

    def predictionStochastic(self, np.ndarray mu, np.ndarray cov, np.ndarray u, np.ndarray mu_new, np.ndarray cov_new, np.ndarray wts_new, int ds_new):
        self.cobj.predictionStochastic(Map[VectorXd](mu), Map[MatrixXd](cov), Map[VectorXd](u), Map[VectorXd](mu_new), Map[MatrixXd](cov_new), Map[VectorXd](wts_new), int(ds_new))
        return ds_new

    def simulateOneStep(self, np.ndarray x, np.ndarray u, np.ndarray x_new, np.ndarray z_new):
        return  self.cobj.simulateOneStep(Map[VectorXd](x),Map[VectorXd](u),Map[VectorXd](x_new), Map[VectorXd](z_new))
