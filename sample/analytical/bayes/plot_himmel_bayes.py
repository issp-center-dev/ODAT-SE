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
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

if __name__ == "__main__":

    data = np.loadtxt("output/BayesData.txt")
    skip = 0

    xrange = (-6, 6)
    yrange = (-6, 6)

    npts = 201
    c_x, c_y = np.mgrid[xrange[0] : xrange[1] : npts * 1j, yrange[0] : yrange[1] : npts * 1j]
    c_z = himmelblau(c_x, c_y)
    levels = np.logspace(0.35, 3.2, 8)

    fig = plt.figure(figsize=(10,5))

    ax = fig.add_subplot(1,2,1)
    ax.set_aspect("equal", adjustable="box")
    ax.contour(c_x, c_y, c_z, levels, colors="k")
    ax.plot(data[skip : -1, 4], data[skip : -1, 5], "o")
    ax.set_xticks(np.linspace(*xrange, num=5, endpoint=True))
    ax.set_yticks(np.linspace(*yrange, num=5, endpoint=True))

    ax = fig.add_subplot(1,2,2)
    ax.set_aspect("equal", adjustable="box")
    ax.contour(c_x, c_y, c_z, levels, colors="k")
    ax.plot(data[skip : -1, 1], data[skip : -1, 2], "-o")
    ax.set_xticks(np.linspace(*xrange, num=5, endpoint=True))
    ax.set_yticks(np.linspace(*yrange, num=5, endpoint=True))

    fig.savefig("res.png")
