#!/usr/bin/env python3

# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Import necessary libraries
from sys import argv  # For command line arguments
from collections import namedtuple  # For creating named tuples
from typing import List, Dict  # For type hints
from pathlib import Path  # For easier file path manipulation

# Named tuple to represent a result entry
# - rank: Process rank
# - step: Simulation step
# - fx: Objective function value
# - xs: List of parameter values
Entry = namedtuple("Entry", ["rank", "step", "fx", "xs"])


def load_best(filename: Path) -> Dict[str, str]:
    """
    Load results in "key=value" format from the specified file
    
    Parameters
    ----------
    filename : Path
        Path to the file to read
        
    Returns
    -------
    Dict[str, str]
        Dictionary containing key-value pairs
    """
    res = {}
    with open(filename) as f:
        for line in f:
            words = line.split("=")
            res[words[0].strip()] = words[1].strip()
    return res


# Set output directory: use command line argument or default to current directory
output_dir = Path("." if len(argv) == 1 else argv[1])
# Load number of processes from best_result.txt
nprocs: int = int(load_best(output_dir / "best_result.txt")["nprocs"])

# Prepare data structures
Ts: List[float] = []  # List of temperature values
labels: List[str] = []  # List of parameter labels
dim: int = 0  # Dimension of parameters
results: Dict[float, List[Entry]] = {}  # Results entries organized by temperature

# Collect temperature (T) values from each process's result file
for rank in range(nprocs):
    with open(output_dir / str(rank) / "result.txt") as f:
        # Extract label information from the first line
        line = f.readline()
        labels = line.split()[4:]  # Skip the first 4 items

        # Extract temperature information from the second line
        line = f.readline()
        words = line.split()
        T = float(words[2])  # Temperature is in the 3rd column
        Ts.append(T)
        results[T] = []  # Initialize result list for this temperature
        dim = len(words) - 4  # Calculate parameter dimension

# Read data from each process's result file and organize by temperature
for rank in range(nprocs):
    with open(output_dir / str(rank) / "result.txt") as f:
        f.readline()  # Skip header line
        for line in f:
            words = line.split()
            step = int(words[0])  # Step number
            T = float(words[2])   # Temperature value
            fx = float(words[3])  # Objective function value
            res = [float(words[i + 4]) for i in range(dim)]  # Parameter values
            results[T].append(Entry(rank=rank, step=step, fx=fx, xs=res))

# Sort by step number for each temperature
for T in Ts:
    results[T].sort(key=lambda entry: entry.step)

# Output results to separate files for each temperature
for i, T in enumerate(Ts):
    with open(output_dir / f"result_T{i}.txt", "w") as f:
        # Write header information
        f.write(f"# T = {T}\n")
        f.write("# step rank fx")
        for label in labels:
            f.write(f" {label}")
        f.write("\n")
        # Write data for each entry
        for entry in results[T]:
            f.write(f"{entry.step} ")
            f.write(f"{entry.rank} ")
            f.write(f"{entry.fx} ")
            for x in entry.xs:
                f.write(f"{x} ")
            f.write("\n")
