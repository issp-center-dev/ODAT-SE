# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import sys
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    fx_data = np.loadtxt(filename, unpack=True, comments="#")
    beta = fx_data[0]
    logz = fx_data[4]
    return beta, logz

def calc_log_pdb(beta, logz, n_mu, w_mu, V):
    assert len(n_mu) == len(w_mu)
    n = np.sum(n_mu)
    w_mu /= np.sum(w_mu)
    log_pdb = logz - np.log(V) + (n / 2) * np.log(beta) + np.sum((n_mu / 2) * np.log(w_mu)) - (n / 2) * np.log(np.pi)
    return log_pdb

def print_log_pdb(filename, *data):
    if not filename:
        return

    beta, log_pdb, *_ = data
    
    with open(filename, "w") as fp:

        idx = np.argmax(log_pdb)
        fp.write("# max log_P(D;beta) = {} at Tstep = {}, beta = {}\n".format(log_pdb[idx], idx, beta[idx]))

        if len(data) == 2:
            fp.write("# $1: Tstep\n"
                     "# $2: beta\n"
                     "# $3: model_evidence\n")
        elif len(data) == 3:
            fp.write("# $1: Tstep\n"
                     "# $2: beta\n"
                     "# $3: average model_evidence\n"
                     "# $4: variance\n")

        for idx, v in enumerate(zip(*data)):
            fp.write("  ".join(map(str, [idx, *v]))+"\n")

def plot_log_pdb(filename, beta, log_pdb, var):
    # Plot the graph
    fig, ax = plt.subplots()
    plt.grid()
    plt.xscale('log')

    ax.set_xlabel('beta')
    ax.set_ylabel('model evidence')

    if var is None:
        ax.scatter(beta, log_pdb, s=50, marker='x', c='red', label='data')
    else:
        ax.errorbar(beta, log_pdb, yerr=var, marker='x', markersize=8, linestyle="none", c='red', label='data')

    if filename:
        fig.savefig(filename)
    else:
        plt.show()

    plt.clf()
    plt.close()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Calculate model evidence values for input data.')
    parser.add_argument("-V", "--Volume", type=float, default=1.0, help="Normalization factor of prior probability distribution.")
    parser.add_argument("-w", "--weight", type=str, help="Relative weights of spots.")
    parser.add_argument("-o", "--output", type=str, default="model_evidence.png", help="Path to output plot image.")
    parser.add_argument("-f", "--result", type=str, default="model_evidence.txt", help="Path to output file.")
    parser.add_argument("-n", "--ndata", type=str, required=True, help="Number of data points of spots.")
    parser.add_argument("data_files", nargs="+", type=str, help="Path to data files.")
    args = parser.parse_args()

    # default values
    V = 1.0
    n = [1]
    w_mu = [1.0]

    if args.Volume:
        V = args.Volume
    if args.ndata:
        n = [int(s) for s in args.ndata.split(",")]
    if args.weight:
        w_mu = [float(s) for s in args.weight.split(",")]

    data_files = args.data_files
    output_file = args.output
    result_file = args.result

    log_pdbs = []
    beta = None

    for data_file in data_files:
        beta, logz = load_data(data_file)
        log_pdb = calc_log_pdb(beta, logz, np.array(n), np.array(w_mu), V)
        log_pdbs.append(log_pdb)

    if len(data_files) == 1:
        print_log_pdb(result_file, beta, log_pdbs[0])
        plot_log_pdb(output_file, beta, log_pdbs[0], None)
    else:
        data = np.stack(log_pdbs)
        avg = np.mean(data, axis=0)
        var = np.std(data, axis=0)
        print_log_pdb(result_file, beta, avg, var)
        plot_log_pdb(output_file, beta, avg, var)
        

if __name__ == "__main__":
    main()
