# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import glob
import numpy as np
import argparse

try:
    from tqdm import tqdm
except:
    tqdm = None

def read_toml(input_filename):
    """
    Reads the TOML configuration file and extracts required parameters.

    Parameters
    ----------
    input_filename : str
        Path to the TOML configuration file.

    Returns
    -------
    dict
        Contains 'replica_per_proc' and 'output_dir' extracted from the config.
    """
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
        if tomllib.__version__ < "1.2.0":
            raise ImportError("tomli 1.2.0 or later required")

    with open(input_filename, "rb") as fp:
        dict_toml = tomllib.load(fp)

    replica_per_proc = dict_toml["algorithm"]["pamc"]["nreplica_per_proc"]
    output_dir = dict_toml["base"]["output_dir"]

    return {
        "replica_per_proc": replica_per_proc,
        "output_dir": output_dir
    }

def extract_columns(file_path, export_dir, replica_per_proc):
    """
    Extracts and processes specific data from a file.

    Processes the last 'replica_per_proc' rows from the file:
      - removes the 1st and 2nd columns,
      - replaces the 3rd column (T) with its reciprocal (beta), and
      - appends results to an output file, adding a header if the file is newly created.

    Parameters
    ----------
    file_path : str
        Path to the input data file.
    export_dir : str
        Directory where the output file will be saved.
    replica_per_proc : int
        Number of rows to extract from the end of the file.

    Note
    ----
    format of input data file:
        step walker_id T(or beta) fx x1 ... xN weight ancestor
    format of output file:
        beta fx x1 ... xN weight
    """
    # Prepare output file path
    file_base, file_ext = os.path.splitext(os.path.basename(file_path))
    output_filename = file_base + "_summarized" + file_ext
    output_file_path = os.path.join(export_dir, output_filename)

    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(output_file_path)

    # Read input data file
    with open(file_path, "r") as f:
        lines_all = f.readlines()
        headers = [line.strip() for line in lines_all if line.startswith("#")]
        lines = [line.strip() for line in lines_all if not line.startswith("#")]

    input_as_beta = None
    for line in headers:
        if "beta" in line:
            input_as_beta = True
            break
        if "T" in line:
            input_as_beta = False
            break
    if input_as_beta is None:
        print("Warning: assume input as T")
        input_as_beta = False

    if replica_per_proc > 0:
        extracted_data = lines[-replica_per_proc:]
    else:
        steps = set([int(line.split()[0]) for line in lines])
        last_step = np.sort(list(steps))[-1]
        extracted_data = [line for line in lines if int(line.split()[0]) == last_step]

    # Process data
    new_lines = []
    for line in extracted_data:
        parts = line.split()
        try:
            val = float(parts[2])
            if input_as_beta:
                beta = val
            elif val != 0.0:
                beta = 1.0 / val
            else:
                beta = 0.0
        except ValueError:
            beta = 0.0
        new_lines.append(" ".join(["{:.6f}".format(beta)] + parts[3:-1]) + "\n")

    # Write data
    with open(output_file_path, "a") as f:
        if not file_exists:
            f.write("# beta fx z1 z2 z3 weight\n")
        f.writelines(new_lines)


def main():
    """
    Main function to parse arguments and execute data extraction.
    """
    parser = argparse.ArgumentParser(description="Extract data from data files based on TOML config.")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the TOML configuration file.")
    parser.add_argument("-n", "--nreplica", type=int, help="Number of replica per process.")
    parser.add_argument("-d", "--data_directory", type=str, help="Path to the data directory.")
    parser.add_argument("-o", "--export_directory", type=str, default="summarized", help="Path to the export directory.")
    parser.add_argument("--progress", action="store_true", help="Show progress bar.")

    args = parser.parse_args()

    replica_per_proc = 0
    output_dir = "."
    export_dir = "."

    # Read TOML configuration
    if args.input_file:
        toml_config = read_toml(args.input_file)
        replica_per_proc = toml_config["replica_per_proc"]
        output_dir = toml_config["output_dir"]

    # Overwrite options by commandline args
    if args.nreplica:
        replica_per_proc = args.nreplica
    if args.data_directory:
        output_dir = args.data_directory

    # Prepare output directory
    os.makedirs(args.export_directory, exist_ok=True)

    # Find and process input files
    file_pattern = os.path.join(output_dir, "*", "result_T*.txt")
    input_files = sorted(glob.glob(file_pattern))

    if tqdm and args.progress:
        input_files = tqdm(input_files)

    # Extract data from each file
    for file_path in input_files:
        try:
            extract_columns(file_path, args.export_directory, replica_per_proc)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    main()
