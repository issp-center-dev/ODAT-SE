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


def load_data(filename: str) -> tuple:
    """
    Load beta and log evidence values from a data file.
    
    Parameters
    ----------
    filename : str
        Path to the data file containing beta and log evidence values.
        
    Returns
    -------
    tuple
        A tuple containing two numpy arrays:
        - beta: inverse temperature values
        - logz: log evidence values
    """
    fx_data = np.loadtxt(filename, unpack=True, comments="#")
    beta = fx_data[0]  # First column contains beta values
    logz = fx_data[4]  # Fifth column contains log evidence values
    return beta, logz


def calc_log_pdb(beta: np.ndarray, logz: np.ndarray, n_mu: np.ndarray, 
                w_mu: np.ndarray, V: float) -> np.ndarray:
    """
    Calculate the logarithm of the model evidence P(D|β).
    
    Parameters
    ----------
    beta : np.ndarray
        Array of inverse temperature values.
    logz : np.ndarray
        Array of log evidence values.
    n_mu : np.ndarray
        Number of data points for each spot.
    w_mu : np.ndarray
        Relative weights for each spot.
    V : float
        Normalization factor of prior probability distribution.
        
    Returns
    -------
    np.ndarray
        Array of log model evidence values.
    """
    assert len(n_mu) == len(w_mu), "Length of n_mu and w_mu must be the same"
    
    # Calculate total number of data points
    n = np.sum(n_mu)
    
    # Normalize weights to sum to 1
    w_mu /= np.sum(w_mu)
    
    # Calculate log model evidence using the formula:
    # log P(D|β) = log Z - log V + (n/2) log β + sum((n_μ/2) log w_μ) - (n/2) log π
    log_pdb = logz - np.log(V) + (n / 2) * np.log(beta) + np.sum((n_mu / 2) * np.log(w_mu)) - (n / 2) * np.log(np.pi)
    
    return log_pdb


def print_log_pdb(filename: str, *data) -> None:
    """
    Write model evidence data to a file.
    
    Parameters
    ----------
    filename : str
        Path to the output file. If None, no file is written.
    *data : tuple
        Variable length argument list containing:
        - beta: inverse temperature values
        - log_pdb: log model evidence values
        - var (optional): variance of log model evidence values
        
    Returns
    -------
    None
    """
    if not filename:
        return

    beta, log_pdb, *_ = data
    
    with open(filename, "w") as fp:
        # Write the maximum log model evidence value and its corresponding beta
        idx = np.argmax(log_pdb)
        fp.write("# max log_P(D;beta) = {} at Tstep = {}, beta = {}\n".format(log_pdb[idx], idx, beta[idx]))

        # Write column headers based on the number of data arrays
        if len(data) == 2:
            fp.write("# $1: Tstep\n"
                     "# $2: beta\n"
                     "# $3: model_evidence\n")
        elif len(data) == 3:
            fp.write("# $1: Tstep\n"
                     "# $2: beta\n"
                     "# $3: average model_evidence\n"
                     "# $4: variance\n")

        # Write the data values
        for idx, v in enumerate(zip(*data)):
            fp.write("  ".join(map(str, [idx, *v]))+"\n")


def plot_log_pdb(filename: str, beta: np.ndarray, log_pdb: np.ndarray, var: np.ndarray = None) -> None:
    """
    Plot the model evidence as a function of beta.
    
    Parameters
    ----------
    filename : str
        Path to save the plot. If None, the plot is displayed instead.
    beta : np.ndarray
        Array of inverse temperature values.
    log_pdb : np.ndarray
        Array of log model evidence values.
    var : np.ndarray, optional
        Array of variance values for log model evidence. If provided, error bars are shown.
        
    Returns
    -------
    None
    """
    # Create a new figure and axes
    fig, ax = plt.subplots()
    plt.grid()
    plt.xscale('log')  # Use logarithmic scale for x-axis (beta)

    # Set axis labels
    ax.set_xlabel('beta')
    ax.set_ylabel('model evidence')

    # Plot data points with or without error bars
    if var is None:
        ax.scatter(beta, log_pdb, s=50, marker='x', c='red', label='data')
    else:
        ax.errorbar(beta, log_pdb, yerr=var, marker='x', markersize=8, 
                   linestyle="none", c='red', label='data')

    # Save or display the plot
    if filename:
        fig.savefig(filename)
    else:
        plt.show()

    # Clean up plot objects
    plt.clf()
    plt.close()


def main():
    """
    Main function to parse arguments and calculate model evidence.
    
    Processes command line arguments, loads data files, calculates model evidence,
    and generates output files and plots.
    """
    import argparse

    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Calculate model evidence values for input data.')
    parser.add_argument("-V", "--Volume", type=float, default=1.0, 
                        help="Normalization factor of prior probability distribution.")
    parser.add_argument("-w", "--weight", type=str, 
                        help="Relative weights of spots.")
    parser.add_argument("-o", "--output", type=str, default="model_evidence.png", 
                        help="Path to output plot image.")
    parser.add_argument("-f", "--result", type=str, default="model_evidence.txt", 
                        help="Path to output file.")
    parser.add_argument("-n", "--ndata", type=str, required=True, 
                        help="Number of data points of spots.")
    parser.add_argument("data_files", nargs="+", type=str, 
                        help="Path to data files.")
    args = parser.parse_args()

    # Initialize default values
    V = 1.0
    n = [1]
    w_mu = [1.0]

    # Override defaults with command line arguments if provided
    if args.Volume:
        V = args.Volume
    if args.ndata:
        n = [int(s) for s in args.ndata.split(",")]
    if args.weight:
        w_mu = [float(s) for s in args.weight.split(",")]

    # Get input and output file paths
    data_files = args.data_files
    output_file = args.output
    result_file = args.result

    # Process each data file and calculate model evidence
    log_pdbs = []
    beta = None

    for data_file in data_files:
        beta, logz = load_data(data_file)
        log_pdb = calc_log_pdb(beta, logz, np.array(n), np.array(w_mu), V)
        log_pdbs.append(log_pdb)

    # Generate output based on number of input files
    if len(data_files) == 1:
        # Single file case: output raw model evidence
        print_log_pdb(result_file, beta, log_pdbs[0])
        plot_log_pdb(output_file, beta, log_pdbs[0], None)
    else:
        # Multiple files case: calculate average and variance
        data = np.stack(log_pdbs)
        avg = np.mean(data, axis=0)
        var = np.std(data, axis=0)
        print_log_pdb(result_file, beta, avg, var)
        plot_log_pdb(output_file, beta, avg, var)
        

if __name__ == "__main__":
    main()
