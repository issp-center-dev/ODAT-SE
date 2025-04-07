# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import argparse
import glob
try:
    from tqdm import tqdm  # Import progress bar library if available
except:
    tqdm = None  # Set to None if import fails


def extract_tag(tag: str, file_input: str, file_output: str) -> None:
    """
    Extract lines starting with a specific tag from input file and write to output file.

    Parameters
    ----------
    tag : str
        The tag to search for at the start of lines
    file_input : str
        Path to the input file to read from
    file_output : str
        Path to the output file to write extracted lines to

    Returns
    -------
    None
    """
    tag_text = "<" + tag + "> "  # Construct the full tag pattern to match
    with open(file_input, "r") as fread, open(file_output, "w") as fwrite:
        for line in fread:
            if line.startswith(tag_text):  # Check if line starts with tag
                ll = line.replace(tag_text, "")  # Remove tag from line
                fwrite.write(ll)  # Write processed line to output file
                
def main():
    """
    Main function to handle command line arguments and process files.
    Extracts tagged lines from combined output files.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Extract lines with a specified tag from combined output")
    parser.add_argument("-d", "--data_dir", type=str, help="Directory of MCMC data")
    parser.add_argument("-t", "--tag", type=str, required=True, help="Tag to extract lines from file.")
    parser.add_argument("--progress", action="store_true", default=False, help="Show progress bar.")
    parser.add_argument("input_files", nargs="*", help="Files to extract in combined format.")
    
    args = parser.parse_args()

    tag = args.tag

    # Determine input files - either from command line arguments or by searching directory
    if args.input_files:
        input_files = args.input_files
    elif args.data_dir:
        # Search for combined.txt files in subdirectories of data_dir
        file_pattern = os.path.join(args.data_dir, "*", "combined.txt")
        input_files = glob.glob(file_pattern)
    else:
        input_files = []

    # Add progress bar if requested and tqdm is available
    if tqdm and args.progress:
        input_files = tqdm(input_files)
    
    # Process each input file
    for input_file in input_files:
        dir_name = os.path.dirname(input_file)
        output_file = os.path.join(dir_name, tag)

        # Print progress message if not using progress bar
        if not args.progress or not tqdm:
            print("extract {} from {}".format(output_file, input_file))

        extract_tag(tag, input_file, output_file)

if __name__ == "__main__":
    main()
