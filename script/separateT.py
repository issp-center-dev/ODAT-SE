#!/usr/bin/env python3

# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This script separates MCMC data files into multiple files based on temperature values.
It reads MCMC result files and splits them into separate files for each temperature.
"""

import os
import sys
import argparse
import glob
from collections import OrderedDict
try:
    from tqdm import tqdm  # Import tqdm for progress bar if available
except ImportError:
    tqdm = None  # Set tqdm to None if not available

# Upper bound on simultaneously open output files, to stay well under typical
# file-descriptor soft limits (e.g. 256 on macOS) regardless of numT.
MAX_OPEN_FILES = 64

def do_separate(filename: str) -> None:
    """
    Separate a single MCMC data file into multiple files, one per temperature.

    Lines are routed to an output file keyed by the temperature value in the
    3rd column (index 2). Unlike a simple "split whenever the value changes
    from the previous line", this groups all lines with the same temperature
    together even when temperatures are interleaved (as in exchange MC), so
    each temperature gets exactly one file. Temperatures are indexed in order
    of first appearance.

    Parameters
    ----------
    filename : str
        Path to the MCMC data file to be separated.

    Returns
    -------
    None
        Files are written to disk with naming pattern: original_filename_T{index}.ext
    """
    file_base, file_ext = os.path.splitext(filename)

    header = []                # header lines (comments starting with #)
    seen: dict = {}            # temperature value (str) -> file index
    # Bounded LRU pool of open handles: keeping one open file per distinct
    # temperature would exhaust the file-descriptor limit for large numT
    # (e.g. hundreds of temperatures vs. the default 256 macOS soft limit).
    # A temperature evicted from the pool is reopened in append mode later.
    open_files: "OrderedDict[str, object]" = OrderedDict()

    def _writer(temperature):
        if temperature in open_files:
            open_files.move_to_end(temperature)  # mark most-recently used
            return open_files[temperature]
        if len(open_files) >= MAX_OPEN_FILES:
            _, victim = open_files.popitem(last=False)  # close least-recent
            victim.close()
        if temperature in seen:
            # reopen an existing file; header already written on first open
            new_file = file_base + f"_T{seen[temperature]}" + file_ext
            out = open(new_file, "a")
        else:
            # first time this temperature is seen: index by appearance order,
            # create the file and write the header collected so far (all
            # comment lines precede the data in result.txt).
            index = len(seen)
            seen[temperature] = index
            new_file = file_base + f"_T{index}" + file_ext
            out = open(new_file, "w")
            out.writelines(header)
            out.write(f"# T (or beta) = {temperature}\n")
        open_files[temperature] = out
        return out

    with open(filename, "r") as fp:
        try:
            for line in fp:
                # Preserve header lines (comments)
                if line.startswith("#"):
                    header.append(line)
                    continue

                # Split line into columns; skip blank / malformed lines
                items = line.split()
                if len(items) < 3:
                    continue

                temperature = items[2]  # temperature is in the 3rd column
                _writer(temperature).write(line)
        finally:
            # Always close every open output file, even on error.
            for out in open_files.values():
                out.close()

def main() -> None:
    """
    Main function to parse arguments and process MCMC data files.
    
    Command line arguments:
    -d/--data_dir: Directory containing MCMC data
    -t/--file_type: File type of MCMC data (default: result.txt)
    --progress: Show progress bar
    input_files: Optional list of specific files to process
    
    Returns
    -------
    None
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Separate MCMC data file by temperature values")
    parser.add_argument("-d", "--data_dir", type=str, help="Directory of MCMC data")
    parser.add_argument("-t", "--file_type", type=str, default="result.txt", help="File type of MCMC data")
    parser.add_argument("--progress", action="store_true", default=False, help="Show progress bar.")
    parser.add_argument("input_files", nargs="*", help="Files to extract in combined format.")

    args = parser.parse_args()

    # Determine input files from arguments
    if args.input_files:
        # Use explicitly provided files
        input_files = args.input_files
    elif args.data_dir:
        # Find files matching pattern in data directory
        file_pattern = os.path.join(args.data_dir, "*", args.file_type)
        input_files = sorted(glob.glob(file_pattern))
    else:
        input_files = []

    # Nothing to do: warn rather than exiting silently with success.
    if not input_files:
        parser.error("no input files: specify input files or --data_dir")

    # Add progress bar if requested and tqdm is available
    if tqdm and args.progress:
        input_files = tqdm(input_files)

    # Process each input file
    for input_file in input_files:
        # Print progress message if not using progress bar
        if not args.progress or not tqdm:
            print("processing file {}...".format(input_file))

        # Separate the file by temperature
        do_separate(input_file)


if __name__ == "__main__":
    main()
