# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

def himmelblau(x, y):
    """
    Calculate the Himmelblau function value for given coordinates.
    
    The Himmelblau function is a multi-modal function often used for testing
    optimization algorithms, defined as: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    
    Parameters
    ----------
    x : float or numpy.ndarray
        x-coordinate(s)
    y : float or numpy.ndarray
        y-coordinate(s)
        
    Returns
    -------
    float or numpy.ndarray
        Himmelblau function value(s) at the given coordinates
    """
    return (x**2+y-11)**2+(x+y**2-7)**2

def read_file(filename, columns=[], **kwopts):
    """
    Read data from a text file and extract specified columns.
    
    Parameters
    ----------
    filename : str
        Path to the input file
    columns : list
        List of column indices to extract from the file
    **kwopts : dict
        Additional keyword arguments to pass to numpy.loadtxt
        
    Returns
    -------
    list
        List of arrays, each containing data from the specified columns
    """
    ds = np.loadtxt(filename, unpack=True, **kwopts)
    return [ds[i] for i in columns]

def plot_data(ax, x, y, f):
    """
    Plot scattered data points with color based on function values.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the scatter plot
    x : numpy.ndarray
        x-coordinates of the points
    y : numpy.ndarray
        y-coordinates of the points
    f : numpy.ndarray
        Function values at each point, used for color mapping
    """
    ax.scatter(x, y, c=f, s=1, marker="o", cmap="plasma")

def plot_contour(ax):
    """
    Plot contour lines of the Himmelblau function.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the contour plot
    """
    # Create a grid of points for contour calculation
    npts = 101
    cx, cy = np.mgrid[-6:6:npts*1j, -6:6:npts*1j]
    # Calculate Himmelblau function values on the grid
    cz = himmelblau(cx, cy)
    # Define logarithmically spaced contour levels
    lvls = np.logspace(0.35, 3.2, 8)
    # Draw contour lines in black
    ax.contour(cx, cy, cz, lvls, colors="k")

if __name__ == "__main__":
    import sys
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="res.png", 
                        help="Output image filename (default: res.png)")
    parser.add_argument("inputfile", nargs="+", 
                        help="One or more input data files to plot")
    args = parser.parse_args()

    # Create figure and configure axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal", adjustable="box")

    # Process each input file
    for infile in args.inputfile:
        # Read function values and coordinates from columns 3, 4, and 5
        f, x, y = read_file(infile, [3, 4, 5])
        plot_data(ax, x, y, f)

    # Add contour lines of the Himmelblau function
    plot_contour(ax)
    # Save the resulting figure
    fig.savefig(args.output)
