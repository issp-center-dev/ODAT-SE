# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Import required libraries
import numpy as np  # For numerical operations

# Import ODAT-SE framework components
import odatse  # Main ODAT-SE package
import odatse.util.toml  # For parsing TOML configuration files
import odatse.algorithm.mapper_mpi as pm_alg  # MPI-based mapper algorithm
#import odatse.algorithm.min_search as pm_alg  # Alternative minimization search algorithm (commented out)
import odatse.solver.function  # Function solver module


def my_objective_fn(x: np.ndarray) -> float:
    """
    A simple objective function that calculates the mean of squared elements.
    
    Parameters
    ----------
    x : np.ndarray
        Input array of numerical values
        
    Returns
    -------
    float
        The mean of squared elements in x
    """
    return np.mean(x * x)  # Calculate element-wise square and then take the mean


# Load configuration from TOML file
file_name = "input.toml"  # Configuration file path
inp = odatse.util.toml.load(file_name)  # Parse the TOML file
info = odatse.Info(inp)  # Create an Info object with the configuration parameters

# Set up the function solver
solver = odatse.solver.function.Solver(info)  # Initialize a function solver with configuration
solver.set_function(my_objective_fn)  # Set our custom objective function

# Create a runner to control the solver execution
runner = odatse.Runner(solver, info)  # Initialize runner with solver and configuration

# Initialize and run the algorithm
alg = pm_alg.Algorithm(info, runner)  # Create an MPI mapper algorithm instance
retv = alg.main()  # Execute the main algorithm and store the return value

# Output the results
print(retv)  # Print the algorithm's return value
