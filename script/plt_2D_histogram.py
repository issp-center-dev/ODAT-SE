# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import argparse
import tomli  # TOML parser for configuration files
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For logarithmic color scaling
import itertools  # For generating combinations of columns

# Try to import tqdm for progress bar, set to None if not available
try:
    from tqdm import tqdm
except:
    tqdm = None


def parse_options():
    """
    Parse command line arguments and configuration files.
    
    Returns
    -------
    dict
        Dictionary containing all configuration options.
    """
    # Default configuration options
    config = {
        "columns": None,          # Column names to draw histograms
        "weight_column": -1,      # Column index for weights (default: last column)
        "bins": 60,               # Number of bins for histograms
        "range": None,            # Range of variables [xmin, xmax]
        "data_dir": ".",          # Directory containing data files
        "output_dir": ".",        # Directory for output files
        "format": ["png"],        # Output image format(s)
        "progress": False,        # Whether to show progress bar
        "xlabel": None,           # Label for x-axis
        "field_list": [],         # List of field labels
        "input_files": [],        # List of input files
    }

    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Generate histograms for files in a directory.')
    parser.add_argument("--config", help="Configuration file.")
    parser.add_argument("--params", help="Parameter file used for calculation.")
    parser.add_argument("-c", "--columns", help="Column names to draw histograms.")
    parser.add_argument("-w", "--weight_column", type=int, help="Column id for weights. default=-1 (last column).")
    parser.add_argument("-b", "--bins", type=int, help="Number of bins.")
    parser.add_argument("-r", "--range", type=str, help="Range of variables, \"xmin,xmax\".")
    parser.add_argument("-d", "--data_dir", type=str, help="Path to data directory")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to output directory.")
    parser.add_argument("-f", "--format", type=str, help="File type of output images. default=png")
    parser.add_argument("--xlabel", type=str, help="Label for x-axis.")
    parser.add_argument("--field_list", type=str, help="Field labels.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")
    parser.add_argument("input_files", nargs="*", type=str, help="Path to data file(s).")
    args = parser.parse_args()

    # Read input parameter file used for calculation
    if args.params:
        with open(args.params, "rb") as f:
            params = tomli.load(f)
        # Extract relevant parameters from the parameter file
        if "algorithm" in params:
            if "param" in params["algorithm"]:
                if "min_list" in params["algorithm"]["param"]:
                    # Get min and max values for each parameter
                    min_list = params["algorithm"]["param"]["min_list"]
                    max_list = params["algorithm"]["param"]["max_list"]
                    # Create range list for each parameter
                    config["range"] = [[float(xmin), float(xmax)] for xmin, xmax in zip(min_list, max_list)]
            if "label_list" in params["algorithm"]:
                # Set field list with standard format: beta, fx, parameters, weight
                config["field_list"] = ["beta", "fx"] + params["algorithm"]["label_list"] + ["weight"]

    # Read config file in TOML format if specified
    if args.config:
        with open(args.config, "rb") as f:
            config.update(tomli.load(f))
        # Convert format string to list if needed
        if isinstance(config["format"], str):
            config["format"] = config["format"].split(",")

    # Override config with command-line arguments if specified
    if args.columns:
        config["columns"] = [s.strip() for s in args.columns.split(",")]
    if args.weight_column:
        config["weight_column"] = args.weight_column
    if args.bins:
        config["bins"] = args.bins
    if args.range is not None:
        if args.range == "":
            config["range"] = None
        else:
            config["range"] = [float(s) for s in args.range.split(",")]
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.format:
        config["format"] = args.format.split(",")
    if args.progress:
        config["progress"] = args.progress
    if args.input_files:
        config["input_files"] = args.input_files
    if args.field_list:
        config["field_list"] = [s.strip() for s in args.field_list.split(",")]
    if args.xlabel:
        config["xlabel"] = args.xlabel

    return config

def show_options(opt):
    """
    Display the current configuration options.
    
    Parameters
    ----------
    opt : dict
        Dictionary containing configuration options.
    """
    for k, v in opt.items():
        print(f"{k} = {v}")

def find_files(data_dir):
    """
    Find all result files in the specified directory.
    
    Parameters
    ----------
    data_dir : str
        Directory to search for files.
        
    Returns
    -------
    list
        Sorted list of paths to result files.
    """
    return sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.startswith("result_T") and f.endswith("_summarized.txt")])

def main():
    """
    Main function to generate 2D histograms from data files.
    """
    # Parse command line options and configuration files
    opt = parse_options()
    show_options(opt)

    # Extract key configuration parameters
    columns = opt["columns"]
    weight_column = opt["weight_column"]
    bins = opt["bins"]
    field_list = opt["field_list"]

    # Determine axis range settings
    hist_range = opt["range"]
    if hist_range is None:
        # Auto-determine axis ranges from data
        auto_axis = True
        share_axis = False
    else:
        auto_axis = False
        if isinstance(hist_range, list):
            if isinstance(hist_range[0], list):
                # Different range for each variable
                share_axis = False
            else:
                # Same range for all variables
                share_axis = True
        else:
            raise ValueError("unknown data type for range parameter: {}".format(type(hist_range)))
    
    # Initialize variables for field mapping
    field_id = None
    z_columns = None

    # Create output directory if it doesn't exist
    os.makedirs(opt["output_dir"], exist_ok=True)

    # Get list of files to process
    file_list = opt["input_files"] if opt["input_files"] else find_files(opt["data_dir"])

    # Add progress bar if requested and tqdm is available
    if tqdm and opt["progress"]:
        file_list = tqdm(file_list)

    # Process each file
    err = 0
    for file_path in file_list:
        try:
            # Load the file using np.loadtxt
            data = np.loadtxt(file_path, delimiter=None)
            ndata, ncols = data.shape

            # Extract beta value in scientific notation for filename
            beta_value = f"{data[0, 0]:.3e}"

            # Get weights from the specified column
            weights = data[:, weight_column]

            # Normalize weights to sum to 1
            normalized_weights = weights / np.sum(weights)

            # Set up column labels if not provided
            if not field_list:
                # Assume standard format: beta, fx, x1, .., xn, weight
                field_list = ["beta", "fx"] + [f"x{i-1}" for i in range(2, ncols-1)] + ["weight"]
            
            # Create mapping from field names to column indices
            if not field_id:
                field_id = {s: id for id, s in enumerate(field_list)}

            # Determine which columns to use for histograms
            if not columns:
                # Default to all data columns (excluding beta, fx, and weight)
                columns = [field_list[i] for i in range(2, ncols-1)]

            # Generate all possible pairs of columns for 2D histograms
            pairs = list(itertools.combinations(columns, 2))
            
            # Generate 2D histograms for all pairs of columns
            for ix, iy in pairs:
                # Get column indices for this pair
                id_x = field_id[ix]
                id_y = field_id[iy]
                
                # Extract data for these columns
                x = data[:, id_x]
                y = data[:, id_y]

                # Determine axis ranges based on configuration
                if auto_axis:
                    # Let numpy determine ranges automatically
                    xrange = None
                    yrange = None
                elif share_axis:
                    # Use same range for both axes
                    xrange = hist_range
                    yrange = hist_range
                else:
                    # Use specific ranges for each axis
                    # Adjust index to account for beta and fx columns
                    xrange = hist_range[id_x - 2]  # Adjust index for range list
                    yrange = hist_range[id_y - 2]  # Adjust index for range list

                # Create 2D histogram with normalized weights
                H, xedges, yedges = np.histogram2d(
                    x, y, bins=bins, range=[xrange, yrange], weights=normalized_weights
                )

                # Replace zeros with a small positive number to avoid log(0) in LogNorm
                H[H == 0] = 1e-10

                # Create figure and plot the 2D histogram
                fig, ax = plt.subplots(figsize=(8, 6))
                X, Y = np.meshgrid(xedges, yedges)
                # Use LogNorm for color scaling to better visualize the distribution
                cmap = ax.pcolormesh(X, Y, H.T, cmap='Reds', shading='auto', 
                                    norm=LogNorm(vmin=1e-3, vmax=1))
                
                # Add colorbar with label
                cbar = plt.colorbar(cmap, ax=ax)
                cbar.set_label('Normalized Density (Log Scale)')

                # Set plot labels and styling
                ax.set_title(f'2D Color Map: {ix} vs {iy}')
                ax.set_xlabel(ix)
                ax.set_ylabel(iy)
                ax.set_xlim(xrange)
                ax.set_ylim(yrange)
                ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid lines

                # Create output filename based on input filename
                file_name = os.path.basename(file_path)
                file_base, file_ext = os.path.splitext(file_name)

                # Replace "_summarized" with beta value in filename
                if "_summarized" in file_base:
                    file_base = file_base.replace("_summarized", f"_beta_{beta_value}")
                
                # Save plot in each requested format
                for suf in opt["format"]:
                    plot_filename = f"2Dhistogram_{file_base}_{ix}_vs_{iy}.{suf}"
                    plt.savefig(os.path.join(opt["output_dir"], plot_filename))
                    if not opt["progress"]:
                        print(f"Histogram created: {plot_filename}")

                # Clean up plot objects
                plt.clf()
                plt.close()

        except Exception as e:
            print(f"Error occurred while processing {file_path}: {e}")
            err += 1

    # Print summary message
    if err == 0:
        print(f"All histograms have been saved in {opt['output_dir']}.")
    else:
        print("ERROR: Some histograms have not been created because of errors.")

if __name__ == "__main__":
    main()
