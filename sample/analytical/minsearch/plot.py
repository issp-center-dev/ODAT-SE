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

plot_style = { "linewidth": 2.0, "markersize": 4.0 }

def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xcol", type=int, default=0)
    parser.add_argument("--ycol", type=int, default=1)
    parser.add_argument("--output", default="res.pdf")
    parser.add_argument("--format", default="-o")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("inputfiles", nargs="+")
    args = parser.parse_args()

    npts = 201
    c_x, c_y = np.mgrid[-6 : 6 : npts * 1j, -6 : 6 : npts * 1j]
    c_z = himmelblau(c_x, c_y)
    levels = np.logspace(0.35, 3.2, 8)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect("equal", adjustable="box")
    ax.contour(c_x, c_y, c_z, levels, colors="k")

    cmap = plt.get_cmap("tab10")
    for idx,filename in enumerate(args.inputfiles):
        data = np.loadtxt(filename, unpack=True, skiprows=args.skip)
        ax.plot(data[args.xcol], data[args.ycol], args.format, color=cmap(idx%10), **plot_style)

    ax.set_xticks(np.linspace(-6, 6, num=5, endpoint=True))
    ax.set_yticks(np.linspace(-6, 6, num=5, endpoint=True))

    fig.savefig(args.output)
