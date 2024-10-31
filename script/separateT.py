#!/usr/bin/env python3

# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


from sys import argv
from collections import namedtuple
from typing import List, Dict
from pathlib import Path

Entry = namedtuple("Entry", ["rank", "step", "fx", "xs"])


def load_best(filename: Path) -> Dict[str, str]:
    res = {}
    with open(filename) as f:
        for line in f:
            words = line.split("=")
            res[words[0].strip()] = words[1].strip()
    return res


output_dir = Path("." if len(argv) == 1 else argv[1])
nprocs: int = int(load_best(output_dir / "best_result.txt")["nprocs"])

Ts: List[float] = []
labels: List[str] = []
dim: int = 0
results: Dict[float, List[Entry]] = {}

for rank in range(nprocs):
    with open(output_dir / str(rank) / "result.txt") as f:
        line = f.readline()
        labels = line.split()[4:]

        line = f.readline()
        words = line.split()
        T = float(words[2])
        Ts.append(T)
        results[T] = []
        dim = len(words) - 4

for rank in range(nprocs):
    with open(output_dir / str(rank) / "result.txt") as f:
        f.readline()
        for line in f:
            words = line.split()
            step = int(words[0])
            T = float(words[2])
            fx = float(words[3])
            res = [float(words[i + 4]) for i in range(dim)]
            results[T].append(Entry(rank=rank, step=step, fx=fx, xs=res))

for T in Ts:
    results[T].sort(key=lambda entry: entry.step)

for i, T in enumerate(Ts):
    with open(output_dir / f"result_T{i}.txt", "w") as f:
        f.write(f"# T = {T}\n")
        f.write("# step rank fx")
        for label in labels:
            f.write(f" {label}")
        f.write("\n")
        for entry in results[T]:
            f.write(f"{entry.step} ")
            f.write(f"{entry.rank} ")
            f.write(f"{entry.fx} ")
            for x in entry.xs:
                f.write(f"{x} ")
            f.write("\n")
