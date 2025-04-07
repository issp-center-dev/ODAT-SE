# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Import required libraries
import numpy as np          # For numerical operations and array handling
import matplotlib.pyplot as plt  # For creating plots and visualizations
import argparse             # For command-line argument parsing (not used in current implementation)


def himmelblau(x, y):
    """
    Himmelblau's function - a multi-modal function commonly used as a benchmark
    for optimization algorithms.
    
    Parameters
    ----------
    x : array_like
        First coordinate
    y : array_like
        Second coordinate
        
    Returns
    -------
    float or ndarray
        Function value at (x,y): (x² + y - 11)² + (x + y² - 7)²
    """
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

if __name__ == "__main__":
    # Load Bayesian optimization data from file
    data = np.loadtxt("output/BayesData.txt")
    skip = 0  # Number of initial data points to skip in visualization

    # Define plotting range for both x and y axes
    xrange = (-6, 6)
    yrange = (-6, 6)

    # Create a grid for contour plotting
    npts = 201  # Number of points in each dimension (resolution)
    c_x, c_y = np.mgrid[xrange[0] : xrange[1] : npts * 1j, yrange[0] : yrange[1] : npts * 1j]
    c_z = himmelblau(c_x, c_y)  # Calculate function values at all grid points
    levels = np.logspace(0.35, 3.2, 8)  # Logarithmically spaced contour levels

    # Create a figure with two subplots
    fig = plt.figure(figsize=(10,5))

    # Left subplot: Showing sample points (likely observed points)
    ax = fig.add_subplot(1,2,1)
    ax.set_aspect("equal", adjustable="box")  # Maintain equal aspect ratio
    ax.contour(c_x, c_y, c_z, levels, colors="k")  # Plot contour lines
    ax.plot(data[skip : -1, 4], data[skip : -1, 5], "o")  # Plot scatter points from columns 5 and 6
    ax.set_xticks(np.linspace(*xrange, num=5, endpoint=True))  # Set x-axis ticks
    ax.set_yticks(np.linspace(*yrange, num=5, endpoint=True))  # Set y-axis ticks
    ax.set_title("Observed Points")  # Add title to subplot

    # Right subplot: Showing optimization trajectory
    ax = fig.add_subplot(1,2,2)
    ax.set_aspect("equal", adjustable="box")  # Maintain equal aspect ratio
    ax.contour(c_x, c_y, c_z, levels, colors="k")  # Plot same contour lines
    ax.plot(data[skip : -1, 1], data[skip : -1, 2], "-o")  # Connect points from columns 2 and 3 with lines
    ax.set_xticks(np.linspace(*xrange, num=5, endpoint=True))  # Set x-axis ticks
    ax.set_yticks(np.linspace(*yrange, num=5, endpoint=True))  # Set y-axis ticks
    ax.set_title("Optimization Trajectory")  # Add title to subplot

    # Add axis labels to both subplots
    for ax in fig.get_axes():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # Save the figure to a file
    fig.savefig("res.png")
    plt.tight_layout()  # Adjust subplot parameters for better layout
