#include "../include/problem_definition.h"

ProblemDefinition::ProblemDefinition()
{
    nState = 2;
    nInput = 2;
    nOutput = 2;
    nModel = 3;
    defineHybridDynamicsModel();
    defineGuardConditions();
}

/* Dynamics Functions*/
void ProblemDefinition::defineHybridDynamicsModel()
{
    // Dynamics 1
    auto dyna_ = make_shared<dynamics::LinearDynamics> (dynamics::LinearDynamics(nState, nInput, nOutput));
    dyna_->V *= 1.;
    dyna_->W *= 10.;
    dynamics.push_back(make_shared<dynamics::LinearDynamics> (*dyna_));
    dyna_.reset();

    // Dynamics 2
    dyna_ = make_shared<dynamics::LinearDynamics> (dynamics::LinearDynamics(nState, nInput, nOutput));
    dyna_->B(1,1) = 0.;
    dyna_->W *= 100.;
    dyna_->W(1,1) = 1e-3;
    dynamics.push_back(make_shared<dynamics::LinearDynamics> (*dyna_));
    dyna_.reset();
    
    // Dynamics 3
    dyna_ = make_shared<dynamics::LinearDynamics> (dynamics::LinearDynamics(nState, nInput, nOutput));
    dyna_->B(0,0) = 0.;
    dyna_->W *= 100.;
    dyna_->W(0,0) = 1e-3;
    dynamics.push_back(make_shared<dynamics::LinearDynamics> (*dyna_));
    dyna_.reset();

    // Dynamics 4
    dyna_ = make_shared<dynamics::LinearDynamics> (dynamics::LinearDynamics(nState, nInput, nOutput));
    dyna_->W *= 100.;
    dynamics.push_back(make_shared<dynamics::LinearDynamics> (*dyna_));
    dyna_.reset();
}


void ProblemDefinition::defineGuardConditions()
{
   double INF = 1000.;

    // GCs for model 0
    Guards gcs;
    std::vector<Guards> multiple_gcs;
    GC_tuple dummy;
    // State 0
    dummy.min_val = -20.;
    dummy.max_val = INF;
    gcs.conditions.push_back(dummy);
    // State 1
    dummy.min_val = -20.;
    dummy.max_val = INF;
    gcs.conditions.push_back(dummy);
    // Add More GCs if needed and then set set_of_gcs to this
    multiple_gcs.push_back(gcs);
    set_of_gcs[0] = multiple_gcs;

    // Clearing vectors
    gcs.conditions.clear();
    multiple_gcs.clear();


    // GCs for model 1
    // State 0
    dummy.min_val = -20.; //-2.;
    dummy.max_val = INF;
    gcs.conditions.push_back(dummy);
    // State 1
    dummy.min_val = -25 ;//-INF;//-2.5;
    dummy.max_val = -20;
    gcs.conditions.push_back(dummy);
    // Add More GCs if needed and then set set_of_gcs to this
    multiple_gcs.push_back(gcs);
    set_of_gcs[1] = multiple_gcs;

    // Clearing vectors
    gcs.conditions.clear();
    multiple_gcs.clear();


    // GCs for model 2
    gcs.conditions.clear();
    // State 0
    dummy.min_val = -25; //-INF; //-2.5;//-INF;
    dummy.max_val = -20;
    gcs.conditions.push_back(dummy);
    // State 1
    dummy.min_val = -20;
    dummy.max_val = INF;
    gcs.conditions.push_back(dummy);
    // Add More GCs if needed and then set set_of_gcs to this
    multiple_gcs.push_back(gcs);
    set_of_gcs[2] = multiple_gcs;

    // Clearing vectors
    gcs.conditions.clear();
    multiple_gcs.clear();
}
