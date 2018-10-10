#include "utils.h"

/* Utilities */
void utils::reduce_map_size(std::map<int, Eigen::MatrixXd> &A, std::map<int, Eigen::MatrixXd> &B, int reduce_by)
{
    for(unsigned int i=0; i <= A.size() - reduce_by; i++)
        B[i] = A[i];
}

void utils::reduce_map_size(std::map<int, std::vector<Guards> > &A, std::map<int, std::vector<Guards> > &B, int reduce_by)
{
    for(unsigned int i=0; i <= A.size() - reduce_by; i++)
        B[i] = A[i];
}


void utils::activeModel(const Eigen::VectorXd& x, int& activeIdx, int nState, int nModel, std::map<int, std::vector<Guards> > &set_of_gcs)
{
   int activeIdx_ = 0; //nModel - 1;
   
   for(int i=1; i< nModel; i++)
   {
        std::vector<Guards> gcs = set_of_gcs[i];
        std::vector<bool> res1(gcs.size(), false);

        for(unsigned int j=0; j<gcs.size(); j++)
        {   
            Guards gc = gcs.at(j);
            std::vector<bool> res(nState, false);

            for(int k=0; k<nState; k++)
            {
                if(x(k) > gc.conditions.at(k).min_val && x(k)<gc.conditions.at(k).max_val)  
                res[k] = true;

                else res[k] = false;
            }

            // Check if all values of the array are true
            if (std::all_of(std::begin(res), std::end(res), [](bool p){return p;}))
            {
                res1.at(j)  = true;
            }
        }

        if(std::any_of(std::begin(res1), std::end(res1), [](bool p){return p;}))
        {
            activeIdx_ = i;
        }

        else continue;
   } 

   activeIdx = activeIdx_;
}


void utils::multiply_vectorMap_w_vector(std::map<int, Eigen::VectorXd>& map_, Eigen::VectorXd& vec, Eigen::Map<Eigen::VectorXd>& res)
{
    res = Eigen::VectorXd::Zero(map_[0].size());

    for(unsigned int j=0; j<map_.size(); j++)
    {   
        res += map_[j]*vec(j);
    }
    // cout << "res = \n" << res << endl;
}


void utils::multiply_matrixMap_w_vector(std::map<int, Eigen::MatrixXd>& map_, Eigen::VectorXd& vec, Eigen::Map<Eigen::MatrixXd>& res)
{
    res = Eigen::MatrixXd::Zero(map_[0].rows(), map_[0].cols());

    for(unsigned int j=0; j<map_.size(); j++)
    {
        res += map_[j]*vec(j);
    }
    // cout << "res =\n" << res << endl;
}



std::map<int, unsigned int> utils::counter(const std::vector<int>& vals) {
    std::map<int, unsigned int> rv;

    for (auto val = vals.begin(); val != vals.end(); ++val) {
        rv[*val]++;
    }

    return rv;
}


void utils::display(const std::map<int, unsigned int>& counts) {
    for (auto count = counts.begin(); count != counts.end(); ++count) {
        std::cout << "Value " << count->first << " has count "
                  << count->second << std::endl;
    }
}

/*Math utils*/
double utils::mvn_pdf(const Eigen::VectorXd &x, const Eigen::VectorXd &meanVec, const Eigen::MatrixXd &covMat)
{
    const double logSqrt2Pi = 0.5*std::log(2*M_PI);
    typedef Eigen::LLT<Eigen::MatrixXd> Chol;
    Chol chol(covMat);
    // Handle non positive definite covariance somehow:
    if(chol.info()!=Eigen::Success) 
        {   
            // cout << "decomposition failed. Trying Nearest PD again!" << endl;
            Eigen::MatrixXd cov_new = covMat;
            nearestPD(cov_new, cov_new);
            Chol chol2(cov_new);
            if(chol2.info() != Eigen::Success) throw "decomposition failed again!";
        }
    const Chol::Traits::MatrixL& L = chol.matrixL(); 
    double quadform = (L.solve(x - meanVec)).squaredNorm();
    return std::exp(-x.rows()*logSqrt2Pi - 0.5*quadform) / L.determinant();
}


bool utils::isPD(Eigen::MatrixXd& A)
{
    Eigen::LLT<Eigen::MatrixXd> lltOfA(A); // compute the Cholesky decomposition of cov
    if(lltOfA.info() == Eigen::NumericalIssue)
        return false;
    else
        return true;
}


void utils::nearestPD(Eigen::MatrixXd& A, Eigen::MatrixXd& A_hat)
{
    Eigen::MatrixXd B, V, S, H, I_;
    double n_rows = A.rows();
    double n_cols = A.cols();
    I_ = Eigen::MatrixXd::Identity(n_rows, n_cols); // Generate Identity Matrix

    B = (A + A.transpose())/2;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    V = svd.matrixV();
    S = svd.singularValues().asDiagonal();

    // cout << "S Matrix =" << S << endl;

    H = V.transpose()*S*V;
    A_hat = (B + H)/2;
    A_hat = (A_hat  + A_hat.transpose())/2;

    int k = 1;
    double mineig;
    //double eps = 1e-16; // A small random number
    Eigen::VectorXd eivals; 

    while(!isPD(A_hat))
    {
        eivals = A_hat.eigenvalues().real(); // Considering only the real part of the matrix
        mineig = eivals.minCoeff();
        // cout << "mineig = \t" << mineig << endl;
        // A_hat += (-mineig*pow(k,2) +  boost::math::float_advnace(mineig))*I_;
        // A_hat += (-mineig*pow(k,2) +  eps(mineig))*I_;
        A_hat += (-mineig*pow(k,2) +  1e-10)*I_;

        k++;
        // cout << "A_hat = \n" << A_hat << endl;
    }

}

/// Folowing function is euivalent to MATLAB eps function to find the distance between the current number and the next floating point number. 
double utils::eps(float x) {
    float const xp = std::abs(x);
    double const x1 = std::nextafter(xp, xp + 1.0f);
    std::cout << "eps_update:\t" << x1-xp<< std::endl;
    return x1 - xp;
}


