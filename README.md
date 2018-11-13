# POMDP-HD
Hierarchical POMDP Planner for Hybrid Dynamics. Relevant details can be found in our paper: [Efficient Hierarchical Robot Motion Planning Under Uncertainty and Hybrid Dynamics](https://arxiv.org/abs/1802.04205) 

## Run an example 
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
cd planner/core/src/dynamics/ 
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

### Dependencies
1. snopt-python. Clone from our fork at ```https://github.com/jainajinkya/snopt-python/tree/snopt-only```

### Contact
In case of any questions, please feel free to contact at *ajinkya@utexas.edu*.

