# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import tomli
import sys
import os
import datetime
import argparse

class Param:
    def __init__(self, inputfile, burn_in, nproc=1):
        with open(inputfile, "rb") as f:
            dict_toml = tomli.load(f)

        self.output_dir = dict_toml["base"].get("output_dir", ".")
        self.dimension = int(dict_toml["base"].get("dimension", 2))

        self.x1_min, self.x2_min = dict_toml["algorithm"]["param"]["min_list"]
        self.x1_max, self.x2_max = dict_toml["algorithm"]["param"]["max_list"]

        self.Tmin = float(dict_toml["algorithm"]["exchange"]["Tmin"])
        self.Tmax = float(dict_toml["algorithm"]["exchange"]["Tmax"])
        self.Tlogspace = dict_toml['algorithm']['exchange'].get('Tlogspace', False)
        self.numsteps = int(dict_toml["algorithm"]["exchange"]["numsteps"])
        self.numsteps_exchange = int(dict_toml["algorithm"]["exchange"]["numsteps_exchange"])
        self.nreplica_per_proc = int(dict_toml["algorithm"]["exchange"].get("nreplica_per_proc", 1))

        self.nproc = nproc
        self.nreplica = self.nreplica_per_proc * self.nproc

        self.burn_in = burn_in

def read_result(param):
    data = None
    for p in range(param.nproc):
        result_file = os.path.join(param.output_dir, str(p), "result.txt")
        print("read {}".format(result_file))
        d = np.loadtxt(result_file)
        d[:,1] += param.nreplica_per_proc * p

        # omit burn_in
        #m = d[:,0] % param.numsteps_exchange >= param.numsteps_exchange * param.burn_in
        m = d[:,0] >= param.numsteps * param.burn_in
        d = d[m]

        if data is None:
            data = d
        else:
            data = np.concatenate([data, d])
    return data

def plot_single(_d, param, output_file):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    _plot(_d, param, ax)
    fig.savefig(output_file, dpi=300)
    plt.clf()
    plt.close()

def plot_multi(_ds, param, layout, output_file):
    ny, nx = layout

    nfigs = min(len(_ds), nx * ny)
    nrows = (nfigs // nx) if (nfigs % nx == 0) else (nfigs // nx + 1)

    fig = plt.figure(figsize=(nx*6, ny*6))
    for idx, _d in enumerate(_ds[:nfigs], 1):
        ax = fig.add_subplot(*layout, idx)

        #cb = idx == nx
        cb = False
        xlabel = (idx-1) // nx == nrows-1
        ylabel = (idx-1) % nx == 0

        _plot(_d, param, ax, cb, xlabel, ylabel)

    fig.savefig(output_file, dpi=300)
    plt.clf()
    plt.close()

def _plot(_d, param, ax, show_colorbar=True, show_xlabel=True, show_ylabel=True):
    Tidx, T, data = _d

    num_of_sample = len(data)
    weight_l = np.ones(num_of_sample) / num_of_sample

    hst2d = ax.hist2d(
        data[:, 4],
        data[:, 5],
        norm=clr.LogNorm(vmin=10**-4, vmax=10**-1),
        range=[[param.x1_min, param.x1_max], [param.x2_min, param.x2_max]],
        cmap="Reds",
        weights=weight_l,
        bins=100,
    )

    line_x1 = np.arange(param.x1_min, param.x1_max + 1, 1)
    line_x2_1 = line_x1
    line_x2_2 = -line_x1 + 1
    ax.plot(line_x1, line_x2_1, c="black", alpha=0.3, lw=0.5)
    ax.plot(line_x1, line_x2_2, c="black", alpha=0.3, lw=0.5)

    if show_colorbar:
        cb = plt.colorbar(hst2d[3])

    ax.set_title("T = {:.3f}".format(T))
    if show_xlabel:
        ax.set_xlabel("x1")
    if show_ylabel:
        ax.set_ylabel("x2")
    ax.set_xlim(param.x1_min, param.x1_max)
    ax.set_ylim(param.x2_min, param.x2_max)
    ax.set_aspect("equal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mpiprocess", action="store", type=int, required=True, help="MPI process")
    parser.add_argument("-i", "--inputfile", action="store", type=str, required=True, help="input toml file")
    parser.add_argument("-b", "--burn_in", action="store", type=float, required=True, help="burn-in ratio")
    parser.add_argument("-o", "--output", action="store", default=None, help="output directory")
    parser.add_argument("--layout", action="store", default=None, help="layout of multiple plots")
    parser.add_argument("--tlist", action="store", default=None, help="list of T indices to plot in multi-plot mode")
    parser.add_argument("--format", action="store", type=str, default="png", help="output format")
    args = parser.parse_args()


    param = Param(args.inputfile, args.burn_in, args.mpiprocess)

    data = read_result(param)

    Tlist = sorted(set(data[:, 2]))

    data_table = []
    for Tidx, T in enumerate(Tlist):
        vv = data[np.isclose(data[:,2], T)]
        data_table.append((Tidx, T, vv))

    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = "{0}_histogram".format(datetime.date.today().strftime("%Y%m%d"))
    os.makedirs(output_dir, exist_ok=True)


    if args.layout is None:
        # single plot mode
        for Tidx, T in enumerate(Tlist):
            output_file = os.path.join(output_dir,
                                       f"{Tidx:03d}_T_{T:.3f}_burn_in_{args.burn_in}.{args.format}")
            # "hist_T{:03d}.png".format(Tidx)

            print("plot {}, T={:.3f}, file={}".format(Tidx, T, output_file))

            plot_single(data_table[Tidx], param, output_file)
    else:
        # multiplot mode
        layout = [int(s) for s in args.layout.split(",")]
        output_file = os.path.join(output_dir, f"hist.{args.format}")

        print("plot multiple figs, file={}".format(output_file))

        if args.tlist is None:
            plot_multi(data_table, param, layout, output_file)
        else:
            tbl = [data_table[int(k)] for k in args.tlist.split(",") if int(k) < len(data_table)]
            plot_multi(tbl, param, layout, output_file)
