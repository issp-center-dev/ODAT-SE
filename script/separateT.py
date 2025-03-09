#!/usr/bin/env python3

# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
import sys
import argparse
import glob
try:
    from tqdm import tqdm
except:
    tqdm = None

def do_separate(filename):
    with open(filename, "r") as fp:
        header = []
        buf = []
        index = 0
        current = None

        for line in fp:
            if line.startswith("#"):
                header.append(line)
                continue

            items = line.strip().split()

            if current is None:
                current = items[2]

            if items[2] != current:
                do_write(filename, index, buf, header)
                current = items[2]
                index += 1
                buf = []

            buf.append(line)
        if buf:
            do_write(filename, index, buf, header)

def do_write(filename, index, buf, header):
    file_base, file_ext = os.path.splitext(filename)
    new_file = file_base + f"_T{index}" + file_ext
    with open(new_file, "w") as fp:
        fp.writelines(header)
        fp.writelines(buf)

def main():
    parser = argparse.ArgumentParser(description="Separate MCMC data file ")
    parser.add_argument("-d", "--data_dir", type=str, help="Directory of MCMC data")
    parser.add_argument("-t", "--file_type", type=str, default="result.txt", help="File type of MCMC data")
    parser.add_argument("--progress", action="store_true", default=False, help="Show progress bar.")
    parser.add_argument("input_files", nargs="*", help="Files to extract in combined format.")

    args = parser.parse_args()

    if args.input_files:
        input_files = args.input_files
    elif args.data_dir:
        file_pattern = os.path.join(args.data_dir, "*", args.file_type)
        input_files = glob.glob(file_pattern)
    else:
        input_files = []

    if tqdm and args.progress:
        input_files = tqdm(input_files)

    for input_file in input_files:
        dir_name = os.path.dirname(input_file)

        if not args.progress or not tqdm:
            print("processing file {}...".format(input_file))

        do_separate(input_file)


if __name__ == "__main__":
    main()
