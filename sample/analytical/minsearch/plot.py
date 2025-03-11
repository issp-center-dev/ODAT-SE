# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# This script visualizes the Himmelblau function and plots optimization trajectories
# over its contour map. The Himmelblau function is a common test function for 
# optimization algorithms with multiple local minima.

import numpy as np
import matplotlib.pyplot as plt
import argparse

# Dictionary defining the style for plotted trajectories
plot_style = { "linewidth": 2.0, "markersize": 4.0 }

def himmelblau(x, y):
    """Calculate the Himmelblau function value for given coordinates.
    
    The Himmelblau function is defined as:
    f(x,y) = (x² + y - 11)² + (x + y² - 7)²
    
    It has four local minima at approximately:
    (3.0, 2.0), (-2.8, 3.1), (-3.8, -3.3), and (3.6, -1.8)
    
    Parameters
    ----------
    x : float or numpy.ndarray
        x-coordinate(s)
    y : float or numpy.ndarray
        y-coordinate(s)
        
    Returns
    -------
    float or numpy.ndarray
        Value(s) of the Himmelblau function
    """
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Plot optimization trajectories over the Himmelblau function")
    parser.add_argument("--xcol", type=int, default=0, help="Column index for x coordinates in input files (default: 0)")
    parser.add_argument("--ycol", type=int, default=1, help="Column index for y coordinates in input files (default: 1)")
    parser.add_argument("--output", default="res.pdf", help="Output file name (default: res.pdf)")
    parser.add_argument("--format", default="-o", help="Plot format string for matplotlib (default: '-o')")
    parser.add_argument("--skip", type=int, default=0, help="Number of rows to skip in input files (default: 0)")
    parser.add_argument("inputfiles", nargs="+", help="Input files containing optimization trajectories")
    args = parser.parse_args()

    # Generate a grid of points to calculate the Himmelblau function values
    npts = 201  # Number of points in each dimension
    c_x, c_y = np.mgrid[-6 : 6 : npts * 1j, -6 : 6 : npts * 1j]  # Create a 2D grid
    c_z = himmelblau(c_x, c_y)  # Calculate function values over the grid
    # Use logarithmic spacing for contour levels to better visualize the function's structure
    levels = np.logspace(0.35, 3.2, 8)  # 8 contour levels from 10^0.35 to 10^3.2

    # Create the figure and set up the plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect("equal", adjustable="box")  # Maintain equal scaling on both axes
    # Plot contour lines of the Himmelblau function
    ax.contour(c_x, c_y, c_z, levels, colors="k")

    # Plot each optimization trajectory from the input files
    cmap = plt.get_cmap("tab10")  # Use the tab10 colormap for distinct colors
    for idx, filename in enumerate(args.inputfiles):
        # Load data from the file, skipping the specified number of rows
        data = np.loadtxt(filename, unpack=True, skiprows=args.skip)
        # Plot the trajectory using the specified columns and format
        ax.plot(data[args.xcol], data[args.ycol], args.format, color=cmap(idx%10), **plot_style)

    # Set axis ticks at regular intervals
    ax.set_xticks(np.linspace(-6, 6, num=5, endpoint=True))
    ax.set_yticks(np.linspace(-6, 6, num=5, endpoint=True))

    # Save the figure to the specified output file
    fig.savefig(args.output)
