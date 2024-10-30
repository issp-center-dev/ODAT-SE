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
    return (x**2+y-11)**2+(x+y**2-7)**2

def read_file(filename, columns=[], **kwopts):
    ds = np.loadtxt(filename, unpack=True, **kwopts)
    return [ds[i] for i in columns]

def plot_data(ax, x, y, f):
    ax.scatter(x,y,c=f,s=1,marker="o",cmap="plasma")

def plot_contour(ax):
    npts = 101
    cx, cy = np.mgrid[-6:6:npts*1j, -6:6:npts*1j]
    cz = himmelblau(cx, cy)
    lvls = np.logspace(0.35, 3.2, 8)
    ax.contour(cx, cy, cz, lvls, colors="k")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="res.png")
    parser.add_argument("inputfile", nargs="+")
    args = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect("equal", adjustable="box")

    for infile in args.inputfile:
        f,x,y = read_file(infile, [3,4,5])
        plot_data(ax, x, y, f)

    plot_contour(ax)
    fig.savefig(args.output)
