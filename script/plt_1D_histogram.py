# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import argparse
import tomli
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except:
    tqdm = None

def parse_options():
    # options with defaults
    config = {
        "columns": None,
        "weight_column": -1,
        "bins": 60,
        "range": None,
        "data_dir": ".",
        "output_dir": ".",
        "format": ["png"],
        "progress": False,
        "xlabel": None,
        "field_list": [],
        "input_files": [],
    }

    # parse command line arguments
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

    # read input parameter file used for calculation
    if args.params:
        with open(args.params, "rb") as f:
            params = tomli.load(f)
        # pickup
        if "algorithm" in params:
            if "param" in params["algorithm"]:
                if "min_list" in params["algorithm"]["param"]:
                    min_list = params["algorithm"]["param"]["min_list"]
                    max_list = params["algorithm"]["param"]["max_list"]
                    config["range"] = [[float(xmin), float(xmax)] for xmin, xmax in zip(min_list, max_list)]
            if "label_list" in params["algorithm"]:
                config["field_list"] = ["beta", "fx"] + params["algorithm"]["label_list"] + ["weight"]

    # read config file in TOML format if specified
    if args.config:
        with open(args.config, "rb") as f:
            config.update(tomli.load(f))
        # adjustments
        if isinstance(config["format"], str):
            config["format"] = config["format"].split(",")

    # store options specified in command-line
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
    for k, v in opt.items():
        print(f"{k} = {v}")

def find_files(data_dir):
    return sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("result_T") and f.endswith("_summarized.txt")])

def main():
    opt = parse_options()
    show_options(opt)

    columns = opt["columns"]
    weight_column = opt["weight_column"]
    bins = opt["bins"]
    field_list = opt["field_list"]

    hist_range = opt["range"]
    if hist_range is None:
        auto_axis = True
        share_axis = False
    else:
        auto_axis = False
        if isinstance(hist_range, list):
            if isinstance(hist_range[0], list):
                share_axis = False
            else:
                share_axis = True
        else:
            raise ValueError("unknown data type for range parameter: {}".format(type(hist_range)))

    field_id = None
    z_columns = None

    os.makedirs(opt["output_dir"], exist_ok=True)

    file_list = opt["input_files"] if opt["input_files"] else find_files(opt["data_dir"])
    if tqdm and opt["progress"]:
        file_list = tqdm(file_list)

    err = 0
    for file_path in file_list:
        try:
            # Load the file using np.loadtxt
            data = np.loadtxt(file_path, delimiter=None)
            ndata, ncols = data.shape

            # Extract beta value in scientific notation (e.g., 1.01e-5)
            beta_value = f"{data[0, 0]:.3e}"

            # Get weights
            weights = data[:, weight_column]

            # Normalize weights
            normalized_weights = weights / np.sum(weights)

            # column labels
            if not field_list:
                # assume standard format: beta, fx, x1, .., xn, weight
                field_list = ["beta", "fx"] + [f"x{i-1}" for i in range(2, ncols-1)] + ["weight"]
            if not field_id:
                field_id = { s: id for id, s in enumerate(field_list) }

            # columns to draw histograms
            if z_columns:
                pass
            else:
                if columns:
                    # selected columns
                    z_columns = [field_id[col] for col in columns]
                else:
                    # all columns
                    z_columns = [i for i in range(2, ncols-1)]
            num_variables = len(z_columns)

            # Plot histograms
            fig, axes = plt.subplots(num_variables, 1, figsize=(10, 4 * num_variables), sharex=share_axis)
            if num_variables == 1:
                axes = [axes]  # Ensure axes is a list even if there is only one variable

            for i, col_index in enumerate(z_columns):
                z = data[:, col_index]
                if auto_axis:
                    hrange = None
                else:
                    hrange = hist_range if share_axis else hist_range[col_index-2] #XXX index

                axes[i].hist(z, bins=bins, range=hrange, weights=normalized_weights)
                axes[i].set_title(f'{field_list[col_index]} 1D Histogram')
                axes[i].set_ylabel(f'{field_list[col_index]}')
                axes[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid lines
                axes[i].set_xlim(hrange)

            if opt["xlabel"]:
                axes[-1].set_xlabel(opt["xlabel"])

            plt.tight_layout()

            # Modify filename format to "1Dhistograms_T0_beta_1.000e+00.png"
            file_name = os.path.basename(file_path)
            file_base, file_ext = os.path.splitext(file_name)

            if "_summarized" in file_base:
                file_base = file_base.replace("_summarized", f"_beta_{beta_value}")
                
            for suf in opt["format"]:
                plot_filename = f"1Dhistogram_{file_base}.{suf}"
                plt.savefig(os.path.join(opt["output_dir"], plot_filename))
                if not opt["progress"]:
                    print(f"Histogram created: {plot_filename}")

            plt.clf()
            plt.close()

        except Exception as e:
            print(f"Error occurred while processing {file_path}: {e}")
            err += 1

    if err == 0:
        print(f"All histograms have been saved in {opt['output_dir']}.")
    else:
        print("ERROR: Some histograms have not been created because of errors.")

if __name__ == "__main__":
    main()
