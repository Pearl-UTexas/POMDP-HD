#include "../include/belief_evolution.h"

BeliefEvolution::BeliefEvolution()
{
    nState = 2;
    nInput = 2;
    nOutput = 2;
    nModel = 4;
    nSamples = 100;

    setMatrices();
    //setGC(goal);

    Eigen::VectorXd I_; 
    I_.setOnes(nModel); 
    wts = (1./nModel)*I_;

    // simulator = new Simulator();
    // simulator->initialize(nState, nInput, nOutput, nModel, mA, mB, mC, mV, mW, set_of_gcs);

}

/* Dynmics Functions*/
void BeliefEvolution::setMatrices()
{
    for(int i=0; i < nModel; i++)
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(nState, nState);
        Eigen::MatrixXd B = Eigen::MatrixXd::Identity(nInput, nInput);
        Eigen::MatrixXd C = Eigen::MatrixXd::Identity(nOutput, nOutput);
        Eigen::MatrixXd V = Eigen::MatrixXd::Identity(nState, nState);
        Eigen::MatrixXd W = Eigen::MatrixXd::Identity(nOutput, nOutput);


        if(i==0)
        {
            B *= 1.0;
            W *= 400.;
        }
        else if(i==1) // Constraint parallel to x
        {
            B(1,1) = 0.; // Movement only in x
            W *= 400.;
            W(1,1) = 1.;
        }
        else if(i==2) // Constraint parallel to y
        {
            B(0,0) = 0.; // Movement only in y
            W *= 400.;
            W(0,0) = 1.; // Good observation only in x
        }
        // Extra goal ds
        else if(i==3)
        {
            B *= 1.; // Movement in all directions
            W *= 400.;
        }

        mA[i] = A;
        mB[i] = B;
        mC[i] = C;
        mV[i] = 0*V;
        mW[i] = W;
}
}


void BeliefEvolution::setGC(Eigen::Map<Eigen::VectorXd>& goal)
{
   double INF = 500.;
   double threshold = 2.0;

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
    dummy.min_val = -INF ;//-INF;//-2.5;
    dummy.max_val = -20.;
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
    dummy.min_val = -INF; //-INF; //-2.5;//-INF;
    dummy.max_val = -20.;
    gcs.conditions.push_back(dummy);
    // State 1
    // dummy.min_val = -20.;
    dummy.min_val = -20.;
    dummy.max_val = INF;
    gcs.conditions.push_back(dummy);
    // Add More GCs if needed and then set set_of_gcs to this
    multiple_gcs.push_back(gcs);
    set_of_gcs[2] = multiple_gcs;

    // Clearing vectors
    gcs.conditions.clear();
    multiple_gcs.clear();

    // GCs for model 3
    gcs.conditions.clear();
    // State 0
    dummy.min_val = goal(0) - threshold;
    dummy.max_val = goal(0) + threshold;
    gcs.conditions.push_back(dummy);
    // State 1
    dummy.min_val = goal(1) - threshold;
    dummy.max_val = goal(1) + threshold;
    gcs.conditions.push_back(dummy);
    // Add More GCs if needed and then set set_of_gcs to this
    multiple_gcs.push_back(gcs);
    set_of_gcs[3] = multiple_gcs;

    // Clearing vectors
    gcs.conditions.clear();
    multiple_gcs.clear();

    cout << "Set Goal = " << goal << endl;
    cout << "GCs set" << endl;

    simulator = new Simulator();
    simulator->initialize(nState, nInput, nOutput, nModel, mA, mB, mC, mV, mW, set_of_gcs);
    cout << "Simulator Initiated" << endl;
}

void BeliefEvolution::getMatrices(int& idx, Eigen::Map<Eigen::MatrixXd>& A, Eigen::Map<Eigen::MatrixXd>& B, Eigen::Map<Eigen::MatrixXd>& C, Eigen::Map<Eigen::MatrixXd>& V, Eigen::Map<Eigen::MatrixXd>& W)
{
    A = mA[idx];
    B = mB[idx];
    C = mC[idx];
    V = mV[idx];
    W = mW[idx];
}


/* Belief Evolution Functions*/
void BeliefEvolution::fastWts(Eigen::VectorXd& mu, Eigen::MatrixXd& cov, Eigen::VectorXd& wts_new)
{   
    // cout << "Inputs: \nmu: " << mu.transpose() << "\ncov:\n" << cov << endl; 
    int n_pts_= nSamples;
    bool use_cholesky = false;
    // if(!manual_seed_){
        // seed = time(0);
    // }
    seed = 0;
    Eigen::EigenMultivariateNormal<double> normX_cholesk(mu, cov, use_cholesky, seed);
    // Eigen::EigenMultivariateNormal<double> normX_cholesk(mu, cov);
    wts_new = Eigen::VectorXd::Zero(nModel);
    
    // Generating Points
    Eigen::MatrixXd pts_ = normX_cholesk.samples(n_pts_); // nState X nPts

    int idx;
    std::vector<int> idxs;
    Eigen::Map<Eigen::VectorXd> pt(pts_.col(0).data(), nState, 1);

    // #pragma omp parallel num_threads(4)
    for(int i=0; i<pts_.cols(); i++)
    {
        // pt = pts_.col(i);
        // Eigen::VectorXd::Map(&vec[0], pt.size()) = pt;
        pt = Eigen::Map<Eigen::VectorXd>(pts_.col(i).data(), nState, 1);
        utils::activeModel(pt, idx, nState, nModel, set_of_gcs);
        idxs.push_back(idx);
        // cout << i << endl;
        // cout << "pt: " << pt.transpose() << " . Hence, idx: " << idx << endl;
    }

    std::map<int, unsigned int> wts_map = utils::counter(idxs);

    for(int i=0; i<nModel; i++)
    {
        wts_new(i) = wts_map[i]; 
    }

    //Normalize wts
    wts_new /= wts_new.sum();
}


/* Belief Evolution Functions*/
void BeliefEvolution::fastWtsMapped(Eigen::Map<Eigen::VectorXd>& mu, Eigen::Map<Eigen::MatrixXd>& cov, Eigen::Map<Eigen::VectorXd>& wts_new)
{   
    Eigen::VectorXd mu1;
    Eigen::MatrixXd cov1;
    Eigen::VectorXd wts1;

    mu1 = mu;
    cov1 = cov;
    
    fastWts(mu1, cov1, wts1);
    wts_new = wts1;
}


void BeliefEvolution::beliefUpdatePlanning(Eigen::Map<Eigen::VectorXd>& mu, Eigen::Map<Eigen::MatrixXd>& cov, Eigen::Map<Eigen::VectorXd>& u, Eigen::Map<Eigen::VectorXd>& mu_new, Eigen::Map<Eigen::MatrixXd>& cov_new, Eigen::Map<Eigen::VectorXd>& wts_new, int ds_new)
{

    Eigen::LLT<Eigen::MatrixXd> lltOfCov(cov); // compute the Cholesky decomposition of cov
    if(lltOfCov.info() == Eigen::NumericalIssue)
    {   
        /* If we don't get as input a PD matrix, there is no need to propagate the dynamics. Hence, we just return back the erroreneous values*/
        mu_new = mu;
        cov_new = cov; // Converting covariance to near PD matrix
        wts_new = wts;

        Eigen::MatrixXf::Index max_index;
        wts_new.maxCoeff(&max_index);
        ds_new = max_index;
        return;
    } 

    else
    {
        if(mu.hasNaN())
        {
            cout << "Invalid Inputs:\nmu\n " << mu << "\nCov =\n" << cov << endl;
            throw std::runtime_error("mu has Nans");
        }


        // cout << "Invoking beliefUpdatePlanning" << endl;
        Eigen::MatrixXd A, B, C, V, W, gamma, dummy, prod_mat;
        I_mat_ = Eigen::MatrixXd::Identity(nOutput, nOutput);

        // ds Prediction
        Eigen::MatrixXd P_dnew_dk;
        P_dnew_dk = eps1*Eigen::MatrixXd::Identity(nModel, nModel);
        Eigen::VectorXd old_wts = wts;
        Eigen::VectorXd mu1;
        Eigen::MatrixXd cov1;
        Eigen::VectorXd wts1, wts_bar;

        // ds Observation update
        Eigen::VectorXd P_z_ds = Eigen::VectorXd::Zero(nModel);
        Eigen::VectorXd new_wts = Eigen::VectorXd::Zero(nModel);
        Eigen::MatrixXd obs_err_cov;

        // Continuous Dynamics
        for(int i=0; i<nModel; i++)
        {   
            A = mA[i];
            B = mB[i];
            C = mC[i];
            V = mV[i];
            W = mW[i];

            // // Introducing Nonlinearity
            // if((i == 1 && u(1) < 0) || (i == 2 && u(0) < 0) || (i == 3 && (u(0)<0 && u(1) < 0))) // allowing motion if moving away from the wall
            //     B = mB[0];

            mu_set_[i] = A*mu + B*u;     
            gamma = A*cov*A.transpose() + V; 

            // May or may not need this
            // W = 0.5*pow((5.0 - mu[0]),2)*I_mat_;

            dummy = C*gamma*C.transpose() + W;
            utils::nearestPD(dummy, prod_mat); // To ensure that inverse exists
            prod_mat = prod_mat.inverse();
            // cov_set_[i] = gamma - gamma*C.transpose()*((C*gamma*C.transpose() + W).inverse())*(C*gamma);
            cov_set_[i] = gamma - gamma*C.transpose()*(prod_mat)*(C*gamma);

            // Discrete State Update
            // Prediction
            mu1 = mu_set_[i];
            cov1 = cov_set_[i];
            fastWts(mu1, cov1, wts1);
            P_dnew_dk.row(i) = wts1;
        }
        // cout << "P_dnew_dk \n" << P_dnew_dk << endl;
        wts_bar = P_dnew_dk.transpose()*old_wts;
        // cout << "New wts = " << wts_bar << endl;
        if(wts_bar.hasNaN() || wts_bar.maxCoeff() < 1e-7)
        {
            cout << "wts in dyna = \n" << wts << endl;
            cout << "Check wts_bar = \n" << wts_bar << endl;
            cout << "Setting wts_bar at = \n" << old_wts << endl;
            cout << "P_dnew_dk = \n" << P_dnew_dk << endl;
            wts_bar = old_wts;
            throw std::runtime_error( "Error encountered in ds_wts calculation!" );
        }
        else
            wts_bar /= wts_bar.sum();

        utils::multiply_vectorMap_w_vector(mu_set_, wts_bar, mu_new);
        //utils::multiply_matrixMap_w_vector(cov_set_, wts_bar, cov_new);

        //* Observation Update *//
        try{
            for(int i=0; i<nModel; i++) 
            {   
                mu1 = mu_set_[i];
                cov1 = cov_set_[i];
                if(!utils::isPD(cov1))
                    utils::nearestPD(cov1, cov1);
            
                // cout << "Calling mvnpdf!" << endl;
                P_z_ds(i) = utils::mvn_pdf(mu_new, mu1, cov1);
                // cout << "Calculate P_z_ds:, mu_new" << mu_new.transpose() <<"\nmu = " << mu1.transpose() << "cov1 = \n" << cov1 << endl;
                // cout << "P_z_ds = "<< P_z_ds(i) << endl;
            }

            if(P_z_ds.hasNaN() || P_z_ds.maxCoeff() < 1e-8)
            {
                new_wts = wts_bar;
                // cout << "Something wrong with P_z_ds, Invoking If Condition" << endl;
                // cout << "P_z_ds = " << P_z_ds.transpose() << endl;
                // throw std::runtime_error( "Error encountered in P_z_ds calculation!" );
            }
            else
            {
                P_z_ds /= P_z_ds.sum();
                new_wts = P_z_ds.array() * wts_bar.array();

                // if(new_wts.hasNaN() || new_wts.maxCoeff() < 0)
                if(!new_wts.allFinite() || new_wts.isZero())
                    new_wts = wts_bar;
                else
                    new_wts /= new_wts.sum();
                
                // cout << "P_z_ds = "<< P_z_ds.transpose()<< endl;
                // cout << "wts_bar = " << wts_bar.transpose() << endl;
                // cout << "new_wts = " << new_wts.transpose() << endl; 
                // cout << "sum of new_wts = " << new_wts.sum() << endl;
            }
        } 

        catch(const char* msg) 
        {
            cerr << msg << endl;
        }

        wts = new_wts;
        // cout << "Discrete State update done!\n" << endl;
        utils::multiply_vectorMap_w_vector(mu_set_, new_wts, mu_new);
        // Update for proper multi-Model mixing
        for(int i=0; i<nModel; i++)
        {   
            dummy = cov_set_[i] + (mu_set_[i]-mu_new)*(mu_set_[i]-mu_new).transpose();
            utils::nearestPD(dummy, dummy);
            cov_set_[i] = dummy;
            // cout << "covset[i]: \n" << cov_set_[i] << endl;
        }
        utils::multiply_matrixMap_w_vector(cov_set_, new_wts, cov_new);

        Eigen::MatrixXf::Index max_index;
        wts.maxCoeff(&max_index);

        ds_new = max_index;
        activeIdx = ds_new;
        wts_new = wts;
        // cout << "Belief Evolution done!\n" << endl;
        // cout << "mu_new: "<< mu_new << "\ncov_new: \n" << cov_new << "\nnew_wts:" << new_wts <<endl;
    }
}


void BeliefEvolution::predictionStochastic(Eigen::Map<Eigen::VectorXd>& mu, Eigen::Map<Eigen::MatrixXd>& cov, Eigen::Map<Eigen::VectorXd>& u, Eigen::Map<Eigen::VectorXd>& mu_new, Eigen::Map<Eigen::MatrixXd>& cov_new, Eigen::Map<Eigen::VectorXd>& wts_new, int& ds_new)
{
    Eigen::MatrixXd A, B, V, P_dnew_dk;
    std::map<int, Eigen::MatrixXd> mA_new, mB_new, mC_new, mW_new;

    // cout << "Cov vals Input: \n" << cov << endl;

    // Discrete Dynamics
    P_dnew_dk = eps1*Eigen::MatrixXd::Identity(nModel, nModel);
    Eigen::VectorXd old_wts = wts;
    Eigen::VectorXd wts_bar;

    Eigen::VectorXd mu1;
    Eigen::MatrixXd cov1;
    Eigen::VectorXd wts1;

    change_ds_resolution_local(mA_new, mB_new, mC_new, mW_new);

    // Continuous Dynamics
    for(int l_count=0; l_count<ds_res_loop_count_; l_count++)
    {
        for(int i=0; i<nModel; i++)
        {   
            // A = mA[i];
            // B = mB[i];
            // V = mV[i];

            A = mA_new[i];
            B = mB_new[i];
            V = mV[i];

            // // Introducing Nonlinearity
            // if((i == 1 && u(1) < 0) || (i == 2 && u(0) < 0) || (i == 3 && (u(0)<0 && u(1) < 0))) // allowing motion if moving away from the wall
            //     B = mB[0];

            // cout << "New Matrices: \n" << A << "\n" << B << endl;

            mu_set_[i] = A*mu + B*u;
            cov_set_[i] = A*cov*A.transpose() + V;

            // Discrete Dynamics
            mu1 = mu_set_[i];
            cov1 = cov_set_[i];
            fastWts(mu1, cov1, wts1);
            // cout << "Calculated wts =\n" << wts1 << endl;
            P_dnew_dk.row(i) = wts1;
        }

        wts_bar = P_dnew_dk.transpose()*old_wts;
        // cout << "New wts = " << wts_bar << endl;
        if(wts_bar.hasNaN())
            wts_bar = old_wts;
        else
            wts_bar /= wts_bar.sum();

        wts = wts_bar;
        utils::multiply_vectorMap_w_vector(mu_set_, wts_bar, mu_new);
        utils::multiply_matrixMap_w_vector(cov_set_, wts_bar, cov_new);
        mu = mu_new;
        cov = cov_new;
    }

    Eigen::MatrixXf::Index max_index;
    wts.maxCoeff(&max_index);

    ds_new = max_index;
    activeIdx = ds_new;

    wts_new = wts;
    // cout << "###################reutrned Vals = \n mu_new:\n"<< mu_new << "\n cov_new\n" << cov_new << "\nwts\n" << wts  << "\nwts_new\n" << wts_new <<"\nidx =\n" << ds_new << endl;
}


void BeliefEvolution::predictionDeterministic(Eigen::Map<Eigen::VectorXd>& mu, Eigen::Map<Eigen::MatrixXd>& cov, Eigen::Map<Eigen::VectorXd>& u, Eigen::Map<Eigen::VectorXd>& mu_new, Eigen::Map<Eigen::MatrixXd>& cov_new, Eigen::Map<Eigen::VectorXd>& wts_new, int ds_new)
{
    // Eigen::VectorXd mu1;
    // mu1 = mu;
    utils::activeModel(mu, activeIdx, nState, nModel, set_of_gcs);
    Eigen::MatrixXd A = mA[activeIdx];
    Eigen::MatrixXd B = mB[activeIdx];
    Eigen::MatrixXd V = mV[activeIdx];

    mu_set_[activeIdx] = A*mu + B*u + V;
    cov_set_[activeIdx] = A*cov*A.transpose() + V; 
    // Reset weights
    wts = Eigen::VectorXd::Zero(nModel);
    wts(activeIdx) = 1.0;

    utils::multiply_vectorMap_w_vector(mu_set_, wts, mu_new);
    utils::multiply_matrixMap_w_vector(cov_set_, wts, cov_new);

    Eigen::MatrixXf::Index max_index;
    wts.maxCoeff(&max_index);

    ds_new = max_index;
    activeIdx = ds_new;
    wts_new = wts;
    // cout << "reutrned Vals = \n mu_set:\n"<< mu_new << "\n cov_set\n" << cov_new << "\nwts\n" << wts <<"\nidx =\n" << ds_new << endl;
}



void BeliefEvolution::observationUpdate(Eigen::Map<Eigen::VectorXd>& z, Eigen::Map<Eigen::VectorXd>& mu_new, Eigen::Map<Eigen::MatrixXd>& cov_new, Eigen::Map<Eigen::VectorXd>& wts_new, int ds_new)
{
    // cout << "wts Earlier = \n" << wts << endl;
    Eigen::MatrixXd C;
    Eigen::MatrixXd W;
    Eigen::VectorXd mu;
    Eigen::MatrixXd cov;
    Eigen::VectorXd zModel;
    Eigen::MatrixXd gamma;
    Eigen::MatrixXd dummy;
    Eigen::MatrixXd prod_mat;
    I_mat_ = Eigen::MatrixXd::Identity(nOutput, nOutput);

    // Discrete State Update
    Eigen::VectorXd wts_bar = wts; //  Previous wts
    Eigen::VectorXd P_z_ds = Eigen::VectorXd::Zero(nModel);
    Eigen::VectorXd new_wts = Eigen::VectorXd::Zero(nModel);
    Eigen::MatrixXd obs_err_cov;

    // Continuous State update
    for(int i=0; i<nModel; i++) 
    {   
        C = mC[i];
        W = mW[i];
        mu = mu_set_[i];

        zModel = C*mu;
        gamma = cov_set_[i];

        // W = 0.5*pow((5.0 - mu[0]),2)*I_mat_;

        /* Innovation Update*/
        dummy = C*gamma*C.transpose() + W;
        utils::nearestPD(dummy, prod_mat);
        prod_mat = prod_mat.inverse();

        mu += gamma*C.transpose()*(prod_mat)*(z - zModel);
        cov = gamma - gamma*C.transpose()*(prod_mat)*(C*gamma);
        
        mu_set_[i] = mu;
        cov_set_[i] = cov;

        // Discrete State
        utils::nearestPD(cov, obs_err_cov); // converting obs_err_cov mat to PD
        P_z_ds(i) = utils::mvn_pdf(z, mu, obs_err_cov);
        // cout << "mu_set_\tmu:\n" << mu << endl;
        // cout << "P_z_ds = "<< P_z_ds(i) << endl;
    }

    // if(std::any_of(P_z_ds(0), P_z_ds.tail(0), [](double p){ return p < 1e-6 || isnan(p);}))
    if(P_z_ds.hasNaN() || P_z_ds.maxCoeff() < 1e-6)
    {
        new_wts = wts_bar;
        cout << "Something wrong with P_z_ds or very low values in P_z_ds, Invoking If Condition" << endl;
    }
    else
    {
        P_z_ds /= P_z_ds.sum();
        new_wts = P_z_ds.array() * wts_bar.array();

        if(new_wts.hasNaN() || new_wts.sum()<1e-10)
            new_wts = wts_bar;
        else
            new_wts /= new_wts.sum();
    }

    wts = new_wts;

    utils::multiply_vectorMap_w_vector(mu_set_, new_wts, mu_new);
    // Update for proper multi-Model mixing
    for(int i=0; i<nModel; i++)
    {
        cov_set_[i] += (mu_set_[i]-mu_new)*(mu_set_[i]-mu_new).transpose();
    }
    utils::multiply_matrixMap_w_vector(cov_set_, new_wts, cov_new);

    Eigen::MatrixXf::Index max_index;
    wts.maxCoeff(&max_index);

    ds_new = max_index;
    activeIdx = ds_new;
    wts_new = wts;

    cout << "wts New= \n" << wts.transpose() << endl;
    // cout << "In EKF Update: \nmu =\t" << mu_new << "\ncov = \n" << cov_new << endl;
}

bool BeliefEvolution::isPDMapped(Eigen::Map<Eigen::MatrixXd>& A)
{
    Eigen::LLT<Eigen::MatrixXd> lltOfA(A); // compute the Cholesky decomposition of cov
    if(lltOfA.info() == Eigen::NumericalIssue)
        return false;
    else
        return true;
}


void BeliefEvolution::nearestPDMapped(Eigen::Map<Eigen::MatrixXd>& A, Eigen::Map<Eigen::MatrixXd>& A_hat)
{
    Eigen::MatrixXd A1;
    Eigen::MatrixXd A_hat1;

    A1 = A;   
    utils::nearestPD(A1, A_hat1);
    A_hat = A_hat1;
}


/* Functions to interact with Simualtor */
void BeliefEvolution::simulate_oneStep(Eigen::Map<Eigen::VectorXd>& x, Eigen::Map<Eigen::VectorXd>& u, Eigen::Map<Eigen::VectorXd>& x_new, Eigen::Map<Eigen::VectorXd>& z_new)
{
    // if(!manual_seed_){
    //     simulator->seed = time(0);
    // }
    simulator->seed = 0;
    return simulator->simulation(x, u, x_new, z_new);
}

/* Utilities */
// FOllowing function can be used to increase the resolution of time step for a discrete time system. Assumptions: System is controllable and observable. Zero-order hold on the control input in the changed resolution till the original time-step. No Process noise. Observation error is random white noise
void BeliefEvolution::increase_resolution_of_ds_dynamics(float k1, float k2, Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& C, Eigen::MatrixXd& W, Eigen::MatrixXd& A_new, Eigen::MatrixXd& B_new, Eigen::MatrixXd& C_new, Eigen::MatrixXd& W_new)
{
    float ratio = k1/k2;
    A_new = A.pow(ratio);
    B_new = ratio*B;
    C_new = C;
    W_new = W;
}

void BeliefEvolution::change_ds_resolution_local(std::map<int, Eigen::MatrixXd>& mA_new, std::map<int, Eigen::MatrixXd>& mB_new, std::map<int, Eigen::MatrixXd>& mC_new, std::map<int, Eigen::MatrixXd>& mW_new)
{
    // Input : New Matrix maps
    //Function: CHnage the resolution of dynamics Matrices in the system

    // Place holders for new Matrices
    Eigen::MatrixXd A, B, C, W, A_new, B_new, C_new, W_new;
    // ds_res_loop_count_ = int(1/k2);
    float k1 = 1.;
    float k2 = float(ds_res_loop_count_);

    cout << "New Sampling ratio: " << k2 << endl;

    for(int i=0; i<nModel; i++)
    {
        A = mA[i];
        B = mB[i];
        C = mC[i];
        W = mW[i];

        increase_resolution_of_ds_dynamics(k1, k2, A, B, C, W, A_new, B_new, C_new, W_new);

        mA_new[i] = A_new;
        mB_new[i] = B_new;
        mC_new[i] = C_new;
        mW_new[i] = W_new;

    }
}

//////////////////////////////
///////* Simulator Class *////
//////////////////////////////
void Simulator::initialize(int nCS, int nControls, int nObs, int nDS, std::map<int, Eigen::MatrixXd> &A, std::map<int, Eigen::MatrixXd> &B, std::map<int, Eigen::MatrixXd> &C, std::map<int, Eigen::MatrixXd> &V, std::map<int, Eigen::MatrixXd> &W, std::map<int, std::vector<Guards> > &gcs)
{   
    nState = nCS;
    nInput = nControls;
    nOutput = nObs;
    nModel = nDS - 1;

    // Matrices
    utils::reduce_map_size(A, mA, 1);
    utils::reduce_map_size(B, mB, 1);
    utils::reduce_map_size(C, mC, 1);
    utils::reduce_map_size(V, mV, 1);
    utils::reduce_map_size(W, mW, 1);

    // Hybrid Dynamics Definition
    utils::reduce_map_size(gcs, set_of_gcs, 1);

    // std::cout << "In Simulator:\n" << "nState: " << nState << "\nnModel:" << nModel << std::endl;
    // utils::printGCs(set_of_gcs);
}


void Simulator::simulation(Eigen::Map<Eigen::VectorXd>& x, Eigen::Map<Eigen::VectorXd>& u, Eigen::Map<Eigen::VectorXd>& x_new, Eigen::Map<Eigen::VectorXd>& z_new)
{
    // Set Matrices
    Eigen::MatrixXd A(nState, nState), V(nState, nState), covar;
    Eigen::MatrixXd B(nState, nInput);
    Eigen::VectorXd zero_mean = Eigen::VectorXd::Zero(nState);


    // Numerical Integration
    float res = 0.001;
    for(int i=0; i<int(1/res); i++)
    {
        utils::activeModel(x, activeIdx, nState, nModel, set_of_gcs);

        // HARD-CODING for 2D domain for now
        if(x[0] < -20. && x[1] < -20.){
            if(u(0) < 0 && u(1) < 0)
                continue;
            if(u(0) > 0)
                activeIdx = 1;
            else activeIdx = 2;
        }        

        // std::cout << "\nx :" << x << std::endl;
        // std::cout << "Active Idx Before update:" << activeIdx << std::endl;

        /* Allowing motion if moving away from the wall*/
        if(activeIdx == 1 && u(1) > 0){
            activeIdx = 0;
        }
        else if(activeIdx == 2 && u(0) > 0){
            activeIdx = 0;
        }

        // std::cout << "Active Idx After update:" << activeIdx << std::endl;

        A << mA[activeIdx];
        B << mB[activeIdx];
        V << mV[activeIdx];

        // System Process Noise
        // utils::nearestPD(V, covar); // Now covar is a positive definite matrix
        // bool use_cholesky = true;
        // uint64_t seed=0;
        // if(seed != 0){
        //    seed = time(0);
        // }
        // Eigen::EigenMultivariateNormal<double> normX_cholesk(zero_mean, covar, use_choleksy, seed);
        // proc_noise_ = normX_cholesk.samples(1);

        x = A*x + B*(u*res); // + proc_noise_;

    }
    
    x_new = x;
    // Recieve observations
    observation(x_new, z_new);

    // cout << "New state = " << x_new << "\n";
    // cout << "observation = " << z_new << "\n";
}


void Simulator::observation(Eigen::Map<Eigen::VectorXd>& x_new, Eigen::Map<Eigen::VectorXd>& z)
{
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nState, nState);
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(nState);


    // Eigen::MatrixXd covar = 0.5*pow(5.0-x_new(0),2)*I;
    Eigen::VectorXd x1;
    x1 = x_new;
    utils::activeModel(x1, activeIdx, nState, nModel, set_of_gcs);
    Eigen::MatrixXd dummy(nOutput, nOutput), covar;
    dummy << mW[activeIdx];
    utils::nearestPD(dummy, covar); // Now covar is a positive definite matrix

    // cout << "covar" << covar << "\n";
    
    bool use_cholesky = false;
    Eigen::EigenMultivariateNormal<double> normX_cholesk(mean, covar, use_cholesky, seed);
    // Eigen::EigenMultivariateNormal<double> normX_cholesk(mean, covar);  
    Eigen::VectorXd obs_noise_ = normX_cholesk.samples(1);
    cout << "obs_noise_ " << obs_noise_ << "\n";

    z = mC[activeIdx]*x_new + obs_noise_;
}
