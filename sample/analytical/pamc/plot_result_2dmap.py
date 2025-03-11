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
    Calculate the Himmelblau function value.
    
    The Himmelblau function is a multi-modal function often used for testing
    optimization algorithms. It has four identical local minima.
    
    Parameters
    ----------
    x : float or array_like
        First coordinate
    y : float or array_like
        Second coordinate
    
    Returns
    -------
    float or array_like
        Function value: (x^2+y-11)^2+(x+y^2-7)^2
    """
    return (x**2+y-11)**2+(x+y**2-7)**2

def read_file(filename, columns=[], **kwopts):
    """
    Read and filter data from a file.
    
    Parameters
    ----------
    filename : str
        Path to the data file
    columns : list
        Indices of columns to extract
    **kwopts : dict
        Additional keyword arguments passed to numpy.loadtxt
    
    Returns
    -------
    list
        List of arrays containing the requested columns, filtered to include
        only rows where the first column modulo 100 equals 0
    """
    ds = np.loadtxt(filename, unpack=True, **kwopts)
    m = ds[0] % 100 == 0  # Filter rows where step % 100 == 0

    return [ds[i][m] for i in columns]

def plot_contour(ax):
    """
    Plot contour lines of the Himmelblau function.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw on
    """
    npts = 101
    cx, cy = np.mgrid[-6:6:npts*1j, -6:6:npts*1j]  # Create a 2D grid
    cz = himmelblau(cx, cy)  # Calculate function values on the grid
    lvls = np.logspace(0.35, 3.2, 8)  # Define contour levels on logarithmic scale
    ax.contour(cx, cy, cz, lvls, colors="k")  # Plot black contour lines

def plot(ax, _xs, _ys, _fs, fmin=None, fmax=None, flabel=None, cmap="plasma"):
    """
    Create a scatter plot of points colored by a third variable, with contour lines.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw on
    _xs : array_like
        x-coordinates
    _ys : array_like
        y-coordinates
    _fs : array_like
        Values to use for coloring points
    fmin : float, optional
        Minimum value for color scale
    fmax : float, optional
        Maximum value for color scale
    flabel : str, optional
        Label for the colorbar
    cmap : str, optional
        Colormap name to use
    """
    if fmin is None:
        fmin = np.min(_fs)
    if fmax is None:
        fmax = np.max(_fs)

    ax.scatter(
        _xs,
        _ys,
        c=_fs,
        s=20,
        marker="o",
        vmin=fmin,
        vmax=fmax,
        cmap=cmap,
    )

    plot_contour(ax)  # Add contour lines of the Himmelblau function

    cb = plt.colorbar(
        cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=fmin, vmax=fmax), cmap=cmap),
        ax=ax,
        label=flabel,
    )

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xlim((-6.0, 6.0))
    ax.set_ylim((-6.0, 6.0))
    ax.axis("square")  # Ensure equal scaling for x and y axes


def main():
    """
    Main function to read data files and create visualization plots.
    
    Reads optimization results from multiple files (for different temperatures 
    and processes), then creates several plots showing the distribution of points
    and their function values or temperatures.
    """
    Tlist = [t for t in range(21)]  # Temperature indices from 0 to 20
    nproc = 4  # Number of parallel processes

    # Initialize arrays to store the combined data
    fs = np.array([])  # Function values
    xs = np.array([])  # x-coordinates
    ys = np.array([])  # y-coordinates
    Ts = np.array([])  # Temperature values

    # Read data from all result files
    for T in Tlist:
        for mpirank in range(nproc):
            filename = os.path.join("output", str(mpirank), f"result_T{T}.txt")
            print("read {}".format(filename))

            # File columns:
            # #0 step
            # #1 walker
            # #2 T
            # #3 fx
            # #4 x1
            # #5 x2
            # #6 weight
            # #7 ancester
            t, f, x, y = read_file(filename, [2, 3, 4, 5])

            # Append data to the combined arrays
            fs = np.concatenate([fs, f])
            xs = np.concatenate([xs, x])
            ys = np.concatenate([ys, y])
            Ts = np.concatenate([Ts, t])

    # Define temperature range for plotting
    Tmin = 0.1
    Tmax = 10.0

    # Create plot showing points colored by temperature
    fig_T = plt.figure()
    ax_T = fig_T.add_subplot(1, 1, 1)
    plot(ax_T, xs, ys, Ts, Tmin, Tmax, "T")
    fig_T.savefig("result_T.png")
    fig_T.savefig("result_T.pdf")

    # Calculate function value range for plotting
    fmin = np.min(fs)
    fmax = np.max(fs)

    # Create plot showing points colored by function value
    fig_fx = plt.figure()
    ax_fx = fig_fx.add_subplot(1, 1, 1)
    plot(ax_fx, xs, ys, fs, fmin, fmax * 0.2, "f(x)", "Purples_r")
    fig_fx.savefig("result_fx.png")
    fig_fx.savefig("result_fx.pdf")

    # Create a combined figure with both plots side by side
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 2, 1)
    plot(ax, xs, ys, Ts, Tmin, Tmax, "T")
    ax = fig.add_subplot(1, 2, 2)
    plot(ax, xs, ys, fs, fmin, fmax * 0.2, "f(x)", "Purples_r")
    fig.savefig("result.png")
    fig.savefig("result.pdf")


if __name__ == "__main__":
    main()
