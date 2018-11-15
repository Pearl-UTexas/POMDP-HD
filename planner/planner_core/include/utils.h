#ifndef UTILS_H_
#define UTILS_H_

#include <map>
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>

/* Custom Data Structures */
struct GC_tuple
{
    double min_val;
    double max_val;
};

struct Guards
{   
    std::vector<GC_tuple> conditions;
};


/* Utilities */
namespace utils{
void reduce_map_size(std::map<int, Eigen::MatrixXd> &, std::map<int, Eigen::MatrixXd> &, int); 
void reduce_map_size(std::map<int, std::vector<Guards> > &, std::map<int, std::vector<Guards> > &, int);

void activeModel(const Eigen::VectorXd&, int&, int, int, std::map<int, std::vector<Guards> > &);

void multiply_vectorMap_w_vector(std::map<int, Eigen::VectorXd>& , Eigen::VectorXd&, Eigen::Map<Eigen::VectorXd>&);

void multiply_matrixMap_w_vector(std::map<int, Eigen::MatrixXd>& , Eigen::VectorXd&, Eigen::Map<Eigen::MatrixXd>&);

std::map<int, unsigned int> counter(const std::vector<int>&);

void display(const std::map<int, unsigned int>&);

/* Math Utils */
double mvn_pdf(const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::MatrixXd &);

bool isPD(Eigen::MatrixXd&);

void nearestPD(Eigen::MatrixXd&, Eigen::MatrixXd&);

double eps(float);

} // end namespace

#endif // UTILS_H_
