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
    m = ds[0] % 100 == 0

    return [ds[i][m] for i in columns]

def plot_contour(ax):
    npts = 101
    cx, cy = np.mgrid[-6:6:npts*1j, -6:6:npts*1j]
    cz = himmelblau(cx, cy)
    lvls = np.logspace(0.35, 3.2, 8)
    ax.contour(cx, cy, cz, lvls, colors="k")

def plot(ax, _xs, _ys, _fs, fmin=None, fmax=None, flabel=None, cmap="plasma"):
    if fmin is None:
        fmin = np.min(fs)
    if fmax is None:
        fmax = np.max(fs)

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

    plot_contour(ax)

    cb = plt.colorbar(
        cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=fmin, vmax=fmax), cmap=cmap),
        ax=ax,
        label=flabel,
    )

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_xlim((-6.0, 6.0))
    ax.set_ylim((-6.0, 6.0))
    ax.axis("square")


def main():
    Tlist = [t for t in range(21)]
    nproc = 4

    fs = np.array([])
    xs = np.array([])
    ys = np.array([])
    Ts = np.array([])

    for T in Tlist:
        for mpirank in range(nproc):
            filename = os.path.join("output", str(mpirank), f"result_T{T}.txt")
            print("read {}".format(filename))

            # #0 step
            # #1 walker
            # #2 T
            # #3 fx
            # #4 x1
            # #5 x2
            # #6 weight
            # #7 ancester
            t, f, x, y = read_file(filename, [2, 3, 4, 5])

            fs = np.concatenate([fs, f])
            xs = np.concatenate([xs, x])
            ys = np.concatenate([ys, y])
            Ts = np.concatenate([Ts, t])


    Tmin = 0.1
    Tmax = 10.0

    fig_T = plt.figure()
    ax_T = fig_T.add_subplot(1, 1, 1)
    plot(ax_T, xs, ys, Ts, Tmin, Tmax, "T")
    fig_T.savefig("result_T.png")
    fig_T.savefig("result_T.pdf")


    fmin = np.min(fs)
    fmax = np.max(fs)

    fig_fx = plt.figure()
    ax_fx = fig_fx.add_subplot(1, 1, 1)
    plot(ax_fx, xs, ys, fs, fmin, fmax * 0.2, "f(x)", "Purples_r")
    fig_fx.savefig("result_fx.png")
    fig_fx.savefig("result_fx.pdf")


    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 2, 1)
    plot(ax, xs, ys, Ts, Tmin, Tmax, "T")
    ax = fig.add_subplot(1, 2, 2)
    plot(ax, xs, ys, fs, fmin, fmax * 0.2, "f(x)", "Purples_r")
    fig.savefig("result.png")
    fig.savefig("result.pdf")


if __name__ == "__main__":
    main()
