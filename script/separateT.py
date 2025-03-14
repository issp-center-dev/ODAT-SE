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
try:
    from tqdm import tqdm  # Import tqdm for progress bar if available
except:
    tqdm = None  # Set tqdm to None if not available

def do_separate(filename: str) -> None:
    """
    Separate a single MCMC data file into multiple files based on temperature values.
    
    Parameters
    ----------
    filename : str
        Path to the MCMC data file to be separated.
    
    Returns
    -------
    None
        Files are written to disk with naming pattern: original_filename_T{index}.ext
    """
    with open(filename, "r") as fp:
        header = []  # Store header lines (comments starting with #)
        buf = []     # Buffer to store lines for current temperature
        index = 0    # Index to track temperature number
        current = None  # Current temperature value
        
        # Process the file line by line
        for line in fp:
            # Preserve header lines (comments)
            if line.startswith("#"):
                header.append(line)
                continue
            
            # Split line into columns
            items = line.strip().split()
            
            # Initialize current temperature if this is the first data line
            if current is None:
                current = items[2]  # Temperature is in the 3rd column (index 2)
            
            # If temperature changes, write the buffered data and reset
            if items[2] != current:
                do_write(filename, index, buf, header)
                current = items[2]  # Update to new temperature
                index += 1          # Increment temperature index
                buf = []            # Clear buffer for new temperature
            
            # Add current line to buffer
            buf.append(line)
            
        # Write the last temperature data if buffer is not empty
        if buf:
            do_write(filename, index, buf, header)

def do_write(filename: str, index: int, buf: list, header: list) -> None:
    """
    Write a separated temperature file.
    
    Parameters
    ----------
    filename : str
        Original filename used to create the new filename.
    index : int
        Temperature index used in the new filename.
    buf : list
        List of data lines to write to the file.
    header : list
        List of header lines to include at the top of the file.
    
    Returns
    -------
    None
        File is written to disk.
    """
    # Create new filename with temperature index
    file_base, file_ext = os.path.splitext(filename)
    new_file = file_base + f"_T{index}" + file_ext
    
    # Write header and data to the new file
    with open(new_file, "w") as fp:
        fp.writelines(header)
        fp.writelines(buf)

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
        input_files = glob.glob(file_pattern)
    else:
        input_files = []

    # Add progress bar if requested and tqdm is available
    if tqdm and args.progress:
        input_files = tqdm(input_files)

    # Process each input file
    for input_file in input_files:
        dir_name = os.path.dirname(input_file)

        # Print progress message if not using progress bar
        if not args.progress or not tqdm:
            print("processing file {}...".format(input_file))

        # Separate the file by temperature
        do_separate(input_file)


if __name__ == "__main__":
    main()
