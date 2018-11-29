from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import eigency

setup(
    # Information
    name = "beliefEvolution",
    version = "1.0.0",
    license = "BSD",
    # Build instructions
    ext_modules = cythonize([Extension("beliefEvolution",
                             ["beliefEvolution.pyx", 
                              "../src/belief_evolution.cpp", 
                              "../src/problem_definition.cpp", 
                              "../src/simulator.cpp", 
                              "../src/utils.cpp",
                              "../src/dynamics_models/linear_dynamics.cpp",
                              "../src/filters/ukf.cpp"],
                             include_dirs=["../include", "/usr/include/eigen3"]+ 
                             eigency.get_includes(include_eigen=False), 
                             extra_compile_args = ["-std=c++11", "-fopenmp"], 
                             extra_link_args=['-lgomp'], language='c++')]),
)
