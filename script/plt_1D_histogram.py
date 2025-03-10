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
                # Get min and max ranges if available
                if "min_list" in params["algorithm"]["param"]:
                    min_list = params["algorithm"]["param"]["min_list"]
                    max_list = params["algorithm"]["param"]["max_list"]
                    # Create range list for each variable
                    config["range"] = [[float(xmin), float(xmax)] for xmin, xmax in zip(min_list, max_list)]
            # Get field labels if available
            if "label_list" in params["algorithm"]:
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
        Sorted list of file paths matching the pattern.
    """
    return sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.startswith("result_T") and f.endswith("_summarized.txt")])

def main():
    """
    Main function to generate 1D histograms from data files.
    """
    # Parse command line options and configuration files
    opt = parse_options()
    show_options(opt)

    # Extract key configuration parameters
    columns = opt["columns"]
    weight_column = opt["weight_column"]
    bins = opt["bins"]
    field_list = opt["field_list"]

    # Determine histogram range and axis sharing behavior
    hist_range = opt["range"]
    if hist_range is None:
        auto_axis = True      # Automatically determine axis range
        share_axis = False    # Don't share axes between subplots
    else:
        auto_axis = False     # Use specified range
        if isinstance(hist_range, list):
            if isinstance(hist_range[0], list):
                share_axis = False    # Different range for each variable
            else:
                share_axis = True     # Same range for all variables
        else:
            raise ValueError("unknown data type for range parameter: {}".format(type(hist_range)))

    field_id = None    # Will map field names to column indices
    z_columns = None   # Will store column indices to plot

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

            # Extract beta value in scientific notation (e.g., 1.01e-5)
            beta_value = f"{data[0, 0]:.3e}"

            # Get weights from the specified column
            weights = data[:, weight_column]

            # Normalize weights so they sum to 1
            normalized_weights = weights / np.sum(weights)

            # Set up column labels if not provided
            if not field_list:
                # Assume standard format: beta, fx, x1, .., xn, weight
                field_list = ["beta", "fx"] + [f"x{i-1}" for i in range(2, ncols-1)] + ["weight"]
            
            # Create mapping from field names to column indices
            if not field_id:
                field_id = {s: id for id, s in enumerate(field_list)}

            # Determine which columns to plot
            if not z_columns:
                if columns:
                    # Use columns specified by name
                    z_columns = [field_id[col] for col in columns]
                else:
                    # Use all data columns (excluding beta, fx, and weight)
                    z_columns = [i for i in range(2, ncols-1)]
            num_variables = len(z_columns)

            # Create figure with subplots for each variable
            fig, axes = plt.subplots(num_variables, 1, figsize=(10, 4 * num_variables), sharex=share_axis)
            if num_variables == 1:
                axes = [axes]  # Ensure axes is a list even if there is only one variable

            # Plot histogram for each variable
            for i, col_index in enumerate(z_columns):
                z = data[:, col_index]  # Get data for this variable
                
                # Determine range for histogram
                if auto_axis:
                    hrange = None  # Let matplotlib determine range automatically
                else:
                    # Use specified range (either shared or per-variable)
                    hrange = hist_range if share_axis else hist_range[col_index-2]  # Adjust index for range list

                # Create histogram with normalized weights
                axes[i].hist(z, bins=bins, range=hrange, weights=normalized_weights)
                axes[i].set_title(f'{field_list[col_index]} 1D Histogram')
                axes[i].set_ylabel(f'{field_list[col_index]}')
                axes[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid lines
                axes[i].set_xlim(hrange)  # Set x-axis limits

            # Set x-label for bottom subplot if provided
            if opt["xlabel"]:
                axes[-1].set_xlabel(opt["xlabel"])

            plt.tight_layout()

            # Create output filename based on input filename
            file_name = os.path.basename(file_path)
            file_base, file_ext = os.path.splitext(file_name)

            # Replace "_summarized" with beta value in filename
            if "_summarized" in file_base:
                file_base = file_base.replace("_summarized", f"_beta_{beta_value}")
                
            # Save plot in each requested format
            for suf in opt["format"]:
                plot_filename = f"1Dhistogram_{file_base}.{suf}"
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
