# POMDP-KC
POMDP based manipulation planning by leveraging knowledge of Kinematic Constraints

## Run# example 
1. Compile cython package
```
cd planner/core/cython
python setup.py build_ext --inplace
```
2. Find solution
```
cd planner/
python run_main.py
```

### Test on a new problem
1. Define problem definition
```
cd planner/core/src/cpp/ 
gedit belief_evolution.cpp
```
2. Update parameter values for trajectory optimization

3. Compile cython package
```
cd planner/core/cython
python setup.py build_ext --inplace
```
4. Find solution
```
cd planner/
python run_main.py
```


### Define a new problem
TODO


### Contact
In case of any questions, please feel free to contact at *ajinkya@utexas.edu*.
