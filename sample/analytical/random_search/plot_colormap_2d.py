# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt


def himmel(x, y):
    """
    Himmelblau's function - a multi-modal optimization test function.
    f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    Has four identical local minima at:
    (3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    """
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


# Create a 2D grid of points for visualization
npts = 201  # Number of points in each dimension
c_x, c_y = np.mgrid[-6 : 6 : npts * 1j, -6 : 6 : npts * 1j]  # Create coordinate grid
c_z = himmel(c_x, c_y)  # Evaluate Himmelblau function on the grid
levels = np.logspace(0.35, 3.2, 8)  # Define contour levels using logarithmic spacing

# Read data points from file
x = []  # x-coordinates
y = []  # y-coordinates
f = []  # function values
file_input = open("output/ColorMap.txt", "r")
lines = file_input.readlines()
file_input.close()
for line in lines:
    if line.strip():  # Skip empty lines
        data = line.split()  # Split line by whitespace
        x.append(float(data[0]))  # First column: x coordinate
        y.append(float(data[1]))  # Second column: y coordinate
        f.append(np.log10(float(data[2])))  # Third column: log10 of function value

# Calculate color scale limits for scatter plot
vmin = np.amin(np.array(f))  # Minimum value for color scale
vmax = np.amax(np.array(f))  # Maximum value for color scale

# Create visualization
# Plot contour lines of the Himmelblau function
plt.contour(c_x, c_y, c_z, levels, colors="k", zorder=10.0, alpha=1.0, linewidths=0.5)
# Plot data points colored by their function values
plt.scatter(
    x,
    y,
    c=f,  # Color by function value
    s=50,  # Point size
    vmin=vmin,
    vmax=vmax,
    cmap="Blues_r",  # Blue colormap (reversed)
    linewidth=2,
    alpha=1.0,
    zorder=1.0,
)
plt.xlim(-6.0, 6.0)  # Set x-axis limits
plt.ylim(-6.0, 6.0)  # Set y-axis limits
plt.colorbar(label="log10(f)")  # Add colorbar with label
plt.savefig("output/ColorMapFig.pdf")  # Save figure as PDF
# plt.savefig("output/ColorMapFig.png")  # Commented out PNG export
