# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Import necessary libraries
import numpy as np          # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import argparse             # For command-line argument parsing


def himmelblau(x, y):
    """Calculate the Himmelblau function.
    
    The Himmelblau function is a multi-modal optimization test function
    defined as f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2.
    It has 4 identical local minima.
    
    Parameters
    ----------
    x : array_like
        x-coordinate input
    y : array_like
        y-coordinate input
        
    Returns
    -------
    float or ndarray
        Value of the Himmelblau function at the given coordinates
    """
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Plot optimization paths over the Himmelblau function")
    parser.add_argument("--xcol", type=int, default=0, help="Column index for x coordinates in input files")
    parser.add_argument("--ycol", type=int, default=1, help="Column index for y coordinates in input files")
    parser.add_argument("-o", "--output", default="res.png", help="Output filename for the plot")
    parser.add_argument("--format", default="o", help="Format string for plot markers")
    parser.add_argument("--skip", type=int, default=0, help="Number of initial data points to skip")
    parser.add_argument("--layout", type=str, default="2x2", help="Subplot layout (rows x columns)")
    parser.add_argument("inputfiles", nargs="+", help="Input files containing optimization paths")
    args = parser.parse_args()

    # Define the range for plotting the Himmelblau function
    xrange = (-6, 6)
    yrange = (-6, 6)

    # Create a grid for the contour plot of the Himmelblau function
    npts = 201  # Number of points in each dimension
    c_x, c_y = np.mgrid[xrange[0] : xrange[1] : npts * 1j, yrange[0] : yrange[1] : npts * 1j]
    c_z = himmelblau(c_x, c_y)  # Calculate function values on the grid
    levels = np.logspace(0.35, 3.2, 8)  # Define contour levels on logarithmic scale

    # Parse the layout string (e.g., "2x2" becomes [2, 2])
    layout = [int(s) for s in args.layout.split("x")]

    # Create the figure
    fig = plt.figure(figsize=(10,10))

    # Process each input file and create a subplot
    for idx, filename in enumerate(args.inputfiles, 1):
        # Extract temperature value from the first line of the file
        with open(filename, "r") as f:
            tval = float(f.readlines()[0].split("=")[1])

        print(filename, tval)  # Print file name and temperature value

        # Load the data points (optimization path)
        data = np.loadtxt(filename, unpack=False)

        # Create a subplot in the specified layout
        ax = fig.add_subplot(*layout, idx)
        ax.set_aspect("equal", adjustable="box")  # Ensure equal scaling for x and y axes
        
        # Plot contour lines of the Himmelblau function
        ax.contour(c_x, c_y, c_z, levels, colors="k")
        
        # Plot the optimization path
        ax.plot(data[args.skip:-1, args.xcol], data[args.skip:-1, args.ycol], args.format)
        
        # Set tick marks at regular intervals
        ax.set_xticks(np.linspace(xrange[0], xrange[1], num=5, endpoint=True))
        ax.set_yticks(np.linspace(yrange[0], yrange[1], num=5, endpoint=True))
        
        # Set subplot title with temperature value
        ax.set_title(f"T={tval:.2f}")

    # Save the figure to the specified output file
    fig.savefig(args.output)
