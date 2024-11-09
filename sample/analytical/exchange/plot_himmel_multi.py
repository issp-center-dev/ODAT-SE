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
    parser = argparse.ArgumentParser()
    parser.add_argument("--xcol", type=int, default=0)
    parser.add_argument("--ycol", type=int, default=1)
    parser.add_argument("-o", "--output", default="res.png")
    parser.add_argument("--format", default="o")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--layout", type=str, default="2x2")
    parser.add_argument("inputfiles", nargs="+")
    args = parser.parse_args()

    xrange = (-6, 6)
    yrange = (-6, 6)

    npts = 201
    c_x, c_y = np.mgrid[xrange[0] : xrange[1] : npts * 1j, yrange[0] : yrange[1] : npts * 1j]
    c_z = himmelblau(c_x, c_y)
    levels = np.logspace(0.35, 3.2, 8)

    layout = [int(s) for s in args.layout.split("x")]

    fig = plt.figure(figsize=(10,10))

    for idx, filename in enumerate(args.inputfiles, 1):

        with open(filename, "r") as f:
            tval = float(f.readlines()[0].split("=")[1])

        print(filename, tval)

        data = np.loadtxt(filename, unpack=False)

        ax = fig.add_subplot(*layout,idx)
        ax.set_aspect("equal", adjustable="box")
        ax.contour(c_x, c_y, c_z, levels, colors="k")
        ax.plot(data[args.skip:-1, args.xcol], data[args.skip:-1, args.ycol], args.format)
        ax.set_xticks(np.linspace(xrange[0], xrange[1], num=5, endpoint=True))
        ax.set_yticks(np.linspace(yrange[0], yrange[1], num=5, endpoint=True))
        ax.set_title(f"T={tval:.2f}")

    fig.savefig(args.output)
