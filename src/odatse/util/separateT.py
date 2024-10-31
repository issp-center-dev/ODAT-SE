# 2DMAT -- Data-analysis software of quantum beam diffraction experiments for 2D material structure
# Copyright (C) 2020- The University of Tokyo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from typing import Dict, List, Optional
import pathlib
from os import PathLike
from collections import namedtuple

import numpy as np

from odatse import mpi

Entry = namedtuple("Entry", ["step", "walker", "fx", "xs"])


def separateT(
    Ts: np.ndarray,
    nwalkers: int,
    output_dir: PathLike,
    comm: Optional[mpi.Comm],
    use_beta: bool,
    buffer_size: int = 10000,
) -> None:
    """
    Separates and processes temperature data for quantum beam diffraction experiments.

    Parameters
    ----------
    Ts : np.ndarray
        Array of temperature values.
    nwalkers : int
        Number of walkers.
    output_dir : PathLike
        Directory to store the output files.
    comm : mpi.Comm, optional
        MPI communicator for parallel processing.
    use_beta : bool
        Flag to determine if beta values are used instead of temperature.
    buffer_size : int, optional
        Size of the buffer for reading input data. Default is 10000.
    """
    if comm is None:
        mpisize = 1
        mpirank = 0
    else:
        mpisize = comm.size
        mpirank = comm.rank
    buffer_size = int(np.ceil(buffer_size / nwalkers)) * nwalkers
    output_dir = pathlib.Path(output_dir)
    proc_dir = output_dir / str(mpirank)

    T2idx = {T: i for i, T in enumerate(Ts)}
    T2rank = {}
    results = []
    for rank, Ts_local in enumerate(np.array_split(Ts, mpisize)):
        d: Dict[str, List[Entry]] = {}
        for T in Ts_local:
            T2rank[str(T)] = rank
            d[str(T)] = []
        results.append(d)

    # write file header
    for T in Ts[mpirank * nwalkers : (mpirank + 1) * nwalkers]:
        idx = T2idx[T]
        with open(output_dir / f"result_T{idx}.txt", "w") as f_out:
            if use_beta:
                f_out.write(f"# beta = {T}\n")
            else:
                f_out.write(f"# T = {T}\n")

    f_in = open(proc_dir / "result.txt")
    EOF = False
    while not EOF:
        for i in range(len(results)):
            for key in results[i].keys():
                results[i][key] = []
        for _ in range(buffer_size):
            line = f_in.readline()
            if line == "":
                EOF = True
                break
            line = line.split("#")[0].strip()
            if len(line) == 0:
                continue
            words = line.split()
            step = int(words[0])
            walker = mpirank * nwalkers + int(words[1])
            Tstr = words[2]
            fx = words[3]
            xs = words[4:]
            entry = Entry(step=step, walker=walker, fx=fx, xs=xs)
            rank = T2rank[Tstr]
            results[rank][Tstr].append(entry)
        if mpisize > 1:
            results2 = comm.alltoall(results)
        else:
            results2 = results
        d = results2[0]
        for i in range(1, len(results2)):
            for key in d.keys():
                d[key].extend(results2[i][key])
        for T in Ts[mpirank * nwalkers : (mpirank + 1) * nwalkers]:
            idx = T2idx[T]
            d[str(T)].sort(key=lambda e: e.step)
            with open(output_dir / f"result_T{idx}.txt", "a") as f_out:
                for e in d[str(T)]:
                    f_out.write(f"{e.step} ")
                    f_out.write(f"{e.walker} ")
                    f_out.write(f"{e.fx} ")
                    for x in e.xs:
                        f_out.write(f"{x} ")
                    f_out.write("\n")
