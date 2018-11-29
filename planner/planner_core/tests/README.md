To compile filter_test.cpp
'''
g++ filter_test.cpp -o filter_test ../src/filters/ukf.cpp ../src/filters/kalman_filter.cpp ../src/problem_definition.cpp ../src/dynamics_models/linear_dynamics.cpp ../src/utils.cpp -std=c++11 
'''
