# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt
import argparse


def himmelblau(x, y):
    """
    Calculate the Himmelblau function value at given coordinates.
    
    The Himmelblau function is a multi-modal function often used to test
    optimization algorithms. It has four identical local minima.
    
    Parameters
    ----------
    x : float or numpy.ndarray
        x-coordinate(s)
    y : float or numpy.ndarray
        y-coordinate(s)
        
    Returns
    -------
    float or numpy.ndarray
        Value(s) of the Himmelblau function at the given coordinates
    """
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def load_data(filenames, xcol=0, ycol=1, **kwopts):
    """
    Load data from multiple text files.
    
    Parameters
    ----------
    filenames : list of str
        List of input file paths
    xcol : int, default=0
        Column index for x-coordinates in the input files
    ycol : int, default=1
        Column index for y-coordinates in the input files
    **kwopts : dict
        Additional keyword arguments passed to numpy.loadtxt
        
    Returns
    -------
    tuple of numpy.ndarray
        Arrays containing x and y coordinates from all input files
    """
    xs = []
    ys = []
    for filename in filenames:
        # Load data from each file using numpy.loadtxt
        data = np.loadtxt(filename, unpack=True, **kwopts)
        xs += list(data[xcol])
        ys += list(data[ycol])
    return np.array(xs), np.array(ys)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--xcol", type=int, default=0, 
                       help="Column index for x-coordinates in input files")
    parser.add_argument("--ycol", type=int, default=1,
                       help="Column index for y-coordinates in input files")
    parser.add_argument("--output", default="res.pdf",
                       help="Output file path")
    parser.add_argument("--format", default="-o",
                       help="Plot format string for matplotlib")
    parser.add_argument("--skip", type=int, default=0,
                       help="Number of initial data points to skip")
    parser.add_argument("inputfiles", nargs="+",
                       help="Input data files")
    args = parser.parse_args()

    # Load data from input files
    xs, ys = load_data(args.inputfiles, xcol=args.xcol, ycol=args.ycol)

    # Create a grid for contour plot of the Himmelblau function
    npts = 201  # Number of grid points in each dimension
    c_x, c_y = np.mgrid[-6 : 6 : npts * 1j, -6 : 6 : npts * 1j]
    c_z = himmelblau(c_x, c_y)
    # Use logarithmic spacing for contour levels to better visualize the function
    levels = np.logspace(0.35, 3.2, 8)

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect("equal", adjustable="box")  # Ensure equal scaling for x and y axes
    # Plot contour lines of the Himmelblau function
    ax.contour(c_x, c_y, c_z, levels, colors="k")
    # Plot the data points, skipping the initial 'skip' points
    ax.plot(xs[args.skip : -1], ys[args.skip : -1], args.format)
    # Set reasonable tick marks
    ax.set_xticks(np.linspace(-6, 6, num=5, endpoint=True))
    ax.set_yticks(np.linspace(-6, 6, num=5, endpoint=True))
    # Save the figure to the specified output file
    fig.savefig(args.output)
