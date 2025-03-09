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
    from tqdm import tqdm
except:
    tqdm = None


def extract_tag(tag, file_input, file_output):
    tag_text = "<" + tag + "> "
    with open(file_input, "r") as fread, open(file_output, "w") as fwrite:
        for line in fread:
            if line.startswith(tag_text):
                ll = line.replace(tag_text, "")
                fwrite.write(ll)
                
def main():
    parser = argparse.ArgumentParser(description="Extract lines with a specified tag from combined output")
    parser.add_argument("-d", "--data_dir", type=str, help="Directory of MCMC data")
    parser.add_argument("-t", "--tag", type=str, required=True, help="Tag to extract lines from file.")
    parser.add_argument("--progress", action="store_true", default=False, help="Show progress bar.")
    parser.add_argument("input_files", nargs="*", help="Files to extract in combined format.")
    
    args = parser.parse_args()

    tag = args.tag

    if args.input_files:
        input_files = args.input_files
    elif args.data_dir:
        file_pattern = os.path.join(args.data_dir, "*", "combined.txt")
        input_files = glob.glob(file_pattern)
    else:
        input_files = []

    if tqdm and args.progress:
        input_files = tqdm(input_files)
    
    for input_file in input_files:
        dir_name = os.path.dirname(input_file)
        output_file = os.path.join(dir_name, tag)

        if not args.progress or not tqdm:
            print("extract {} from {}".format(output_file, input_file))

        extract_tag(tag, input_file, output_file)

if __name__ == "__main__":
    main()
