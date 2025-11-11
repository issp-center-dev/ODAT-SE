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
    Calculate the logarithm of the model evidence P(D;\beta).
    
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
    # log P(D;\beta) = log Z - log V + (n/2) log \beta + sum((n_\mu/2) log w_\mu) - (n/2) log \pi
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

def auto_range(beta, log_pdb, focus_factor=0.5):
    """
    Automatically calculate plot range focusing on the maximum model evidence region
    
    Parameters
    ----------
    beta : np.ndarray
        Array of inverse temperature values
    log_pdb : np.ndarray
        Array of log model evidence values
    focus_factor : float
        Focus tightness factor (0 to 1). Smaller values result in tighter focus
        
    Returns
    -------
    tuple
        Tuple containing (beta_min, beta_max, y_min, y_max)
    """
    # Filter out NaN and Inf values
    valid_mask = np.isfinite(log_pdb) & np.isfinite(beta) & (beta > 0)
    if not np.any(valid_mask):
        print("Warning: No valid data points for auto-focus")
        return np.min(beta), np.max(beta), np.min(log_pdb), np.max(log_pdb)
    valid_beta = beta[valid_mask]
    valid_log_pdb = log_pdb[valid_mask]
    max_idx = np.argmax(valid_log_pdb)
    beta_opt = valid_beta[max_idx]
    log_pdb_max = valid_log_pdb[max_idx]
    
    # Method 1: Find points within a certain drop from maximum (e.g., 3 units for log scale)
    drop_threshold = 3.0  # Adjust this value to control how much to include around peak
    near_max_mask = (valid_log_pdb > log_pdb_max - drop_threshold)
    
    # Method 2: Use gradient to find where the curve starts dropping sharply
    gradient = np.gradient(valid_log_pdb)
    
    # Find the region around maximum where gradient is relatively small
    gradient_threshold = np.percentile(np.abs(gradient), 75)
    gentle_slope_mask = np.abs(gradient) < gradient_threshold
    
    # Combine both methods: points near max OR with gentle slope
    significant_mask = near_max_mask | gentle_slope_mask
    
    # Find continuous region around maximum
    left_idx = max_idx
    right_idx = max_idx
    
    # Expand left
    while left_idx > 0 and (significant_mask[left_idx-1] or 
                            valid_log_pdb[left_idx-1] > log_pdb_max - drop_threshold * 2):
        left_idx -= 1
    
    # Expand right
    while right_idx < len(valid_log_pdb) - 1 and (significant_mask[right_idx+1] or
                                                   valid_log_pdb[right_idx+1] > log_pdb_max - drop_threshold * 2):
        right_idx += 1
    
    # Add some padding
    padding = max(5, int(len(valid_log_pdb) * 0.05))
    left_idx = max(0, left_idx - padding)
    right_idx = min(len(valid_log_pdb) - 1, right_idx + padding)
    
    # Get beta range for x-axis
    beta_min_focus = valid_beta[left_idx]
    beta_max_focus = valid_beta[right_idx]
    
    # Expand range based on focus_factor
    if beta_min_focus > 0 and beta_max_focus > 0:
        beta_log_center = np.log10(beta_opt)
        beta_log_half_range = max(0.5, (np.log10(beta_max_focus) - np.log10(beta_min_focus)) / 2)
        
        # Apply focus factor (0 = tight focus, 1 = loose focus)
        expansion = 1 + focus_factor * 2
        beta_min = 10 ** (beta_log_center - beta_log_half_range * expansion)
        beta_max = 10 ** (beta_log_center + beta_log_half_range * expansion)
        
        # Ensure we don't go beyond data bounds
        beta_min = max(beta_min, valid_beta[0] * 0.8)
        beta_max = min(beta_max, valid_beta[-1] * 1.2)
    else:
        beta_min = beta_min_focus
        beta_max = beta_max_focus
    
    # Calculate y-axis range based on the focused x range
    in_range_mask = (valid_beta >= beta_min) & (valid_beta <= beta_max)
    if np.any(in_range_mask):
        y_data_in_range = valid_log_pdb[in_range_mask]
        
        # Remove outliers for y-range calculation using IQR method
        q1 = np.percentile(y_data_in_range, 25)
        q3 = np.percentile(y_data_in_range, 75)
        iqr = q3 - q1
        
        # Define outliers as points beyond 1.5*IQR from quartiles
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        y_filtered = y_data_in_range[(y_data_in_range >= lower_bound) & 
                                     (y_data_in_range <= upper_bound)]
        
        if len(y_filtered) > 0:
            y_min = np.min(y_filtered)
            y_max = np.max(y_filtered)
        else:
            y_min = np.min(y_data_in_range)
            y_max = np.max(y_data_in_range)
        
        # Add margin
        y_range = y_max - y_min
        y_margin = max(y_range * 0.1, 0.5)
        y_min = y_min - y_margin
        y_max = y_max + y_margin
        
        # Make sure we include the maximum point
        y_max = max(y_max, log_pdb_max + y_margin)
    else:
        # Fallback
        y_min = log_pdb_max - 10
        y_max = log_pdb_max + 2
    
    print(f"Auto-focus: beta range [{beta_min:.2e}, {beta_max:.2e}], y range [{y_min:.2f}, {y_max:.2f}]")
    
    return beta_min, beta_max, y_min, y_max

def plot_log_pdb(filename: str, beta: np.ndarray, log_pdb: np.ndarray, var: np.ndarray = None, auto_focus = False, focus_factor = 0.5) -> None:
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
    auto_focus : bool
        Whether to use automatic focus feature
    focus_factor : float
        Auto-focus tightness (0 to 1)
        
    Returns
    -------
    None
    """
    # Create a new figure and axes
    fig, ax = plt.subplots()
    plt.grid()
    plt.xscale('log')  # Use logarithmic scale for x-axis (beta)

    ax.set_xlabel('beta')
    ax.set_ylabel('model evidence')

    # Plot data points with or without error bars
    if var is None:
        ax.scatter(beta, log_pdb, s=50, marker='x', c='red', label='data')
    else:
        ax.errorbar(beta, log_pdb, yerr=var, marker='x', markersize=8, 
                   linestyle="none", c='red', label='data')

    # Highlight maximum point
    valid_mask = np.isfinite(log_pdb) & np.isfinite(beta)
    if not np.any(valid_mask):
        raise ValueError("No valid data points")
    valid_log_pdb = log_pdb[valid_mask]
    valid_beta = beta[valid_mask]
    max_idx = np.argmax(valid_log_pdb)
    beta_opt = valid_beta[max_idx]
    log_pdb_max = valid_log_pdb[max_idx]
    
    ax.scatter(beta_opt, log_pdb_max, s=200, marker='o', 
              facecolors='none', edgecolors='blue', linewidth=2, 
              label=f'Max at $\\beta={beta_opt:.2e}$')
    
    # Set axis properties
    if auto_focus:
        beta_min, beta_max, y_min_auto, y_max_auto = auto_range(beta, log_pdb, focus_factor)
        ax.set_xlim([beta_min, beta_max])
        ax.set_ylim([y_min_auto, y_max_auto])
    ax.set_xlabel('$\\beta$ (inverse temperature)', fontsize=12)
    ax.set_ylabel('$\\log P(D|\\beta)$ (model evidence)', fontsize=12)
    ax.legend(loc='best')
    ax.set_title(f'Model Evidence vs Beta (Max: {log_pdb_max:.4f} at $\\beta={beta_opt:.2e}$)', fontsize=14)
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
    parser.add_argument('--auto-focus', action='store_true',
                       help='Auto-focus on maximum model evidence region')
    parser.add_argument('--focus-factor', type=float, default=0.5,
                       help='Auto-focus tightness (0-1, smaller is tighter, default: 0.5)')
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
    
    # Check for invalid arguments
    if args.Volume<=0:
        sys.exit("Error: normalization factor must be greater than 0.")
    if args.focus_factor<0 or args.focus_factor>1:
        sys.exit("Error: focus factor must be between 0 and 1.")

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
