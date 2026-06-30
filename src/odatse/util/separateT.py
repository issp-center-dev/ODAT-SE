# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import pathlib
from os import PathLike
from collections import namedtuple

import numpy as np

if TYPE_CHECKING:
    from mpi4py import MPI

# step and walker are stored as ints; Tstr, fx, xs are kept as raw strings so
# that exact serialised values are round-tripped without floating-point loss.
Entry = namedtuple("Entry", ["step", "walker", "Tstr", "fx", "xs"])


# ---------------------------------------------------------------------------
# separateT – private helpers
# ---------------------------------------------------------------------------

def _parse_result_line(line: str, mpirank: int, nwalkers: int) -> Optional[Entry]:
    """Parse one line from ``result.txt`` into an :class:`Entry`.

    Returns ``None`` for blank or comment-only lines.
    The *walker* field is remapped from a rank-local index to a global index.
    *Tstr*, *fx*, and *xs* are kept as raw strings to preserve exact values.
    """
    line = line.split("#")[0].strip()
    if not line:
        return None
    words = line.split()
    return Entry(
        step=int(words[0]),
        walker=mpirank * nwalkers + int(words[1]),
        Tstr=words[2],
        fx=words[3],
        xs=words[4:],
    )


def _distribute_entries(
    entries: list[Entry],
    T2rank: dict[str, int],
    results: list[dict[str, list[Entry]]],
) -> None:
    """Append each entry into the destination-rank bucket it belongs to."""
    for entry in entries:
        results[T2rank[entry.Tstr]][entry.Tstr].append(entry)


def _merge_results(results2: list[dict[str, list[Entry]]]) -> dict[str, list[Entry]]:
    """Merge per-source dicts received from alltoall into a single dict.

    ``results2[0]`` is mutated in-place and returned; the caller owns the object
    (alltoall in mpi4py returns fresh objects, and in the serial path this is the
    same dict that will be cleared at the start of the next chunk).
    """
    merged = results2[0]
    for part in results2[1:]:
        for key in merged:
            merged[key].extend(part[key])
    return merged


def _clear_results(results: list[dict[str, list[Entry]]]) -> None:
    """Reset all entry lists to empty so the structure can be reused each chunk."""
    for d in results:
        for key in d:
            d[key] = []


def _write_entries_to_file(entries: list[Entry], filepath: pathlib.Path) -> None:
    """Sort *entries* by step number and append them to *filepath*."""
    entries.sort(key=lambda e: e.step)
    with open(filepath, "a") as f_out:
        for e in entries:
            f_out.write(f"{e.step} {e.walker} {e.fx}")
            for x in e.xs:
                f_out.write(f" {x}")
            f_out.write("\n")


# ---------------------------------------------------------------------------
# calculate_statistics – private helpers
# ---------------------------------------------------------------------------

def _read_result_T_entries(
    filepath: pathlib.Path,
) -> list[tuple[int, float, np.ndarray]]:
    """Read ``result_T<idx>.txt`` and return a list of ``(step, fx, x)`` tuples.

    Comment lines and blank lines are skipped.
    """
    samples: list[tuple[int, float, np.ndarray]] = []
    with open(filepath, "r") as f_in:
        for line in f_in:
            line = line.split("#")[0].strip()
            if not line:
                continue
            words = line.split()
            step = int(words[0])
            fx = float(words[2])
            x = np.array([float(v) for v in words[3:]])
            samples.append((step, fx, x))
    return samples


def _compute_temperature_statistics(
    samples: list[tuple[int, float, np.ndarray]],
    thermalization_steps: int,
    dbeta: float,
) -> tuple[float, float, float, float]:
    """Compute statistics for one temperature from a list of MC samples.

    Parameters
    ----------
    samples : list of (step, fx, x)
        Raw MC samples in the order they were recorded.
    thermalization_steps : int
        Steps with ``step < thermalization_steps`` are discarded as thermalisation.
    dbeta : float
        ``1/T_lower - 1/T``; pass ``0.0`` when there is no lower-temperature
        neighbour (the thermodynamic-integration contribution is then 0).

    Returns
    -------
    fx_mean : float
        Arithmetic mean of f(x) over production samples.  ``nan`` if N == 0.
    fx_error : float
        Standard error of the mean.  ``nan`` if N <= 1.
    dlogZ : float
        ``log(Z_lower / Z_this)`` via log-sum-exp thermodynamic integration.
        ``0.0`` when ``dbeta == 0.0`` or N == 0.
    acceptance : float
        Fraction of steps where x changed.  ``nan`` if N == 0.
    """
    dlogZ = 0.0
    f_base: Optional[float] = None
    fx_sum = 0.0
    fx_sum2 = 0.0
    accepted = 0
    old_x: Optional[np.ndarray] = None
    N = 0

    for step, fx, x in samples:
        if step < thermalization_steps:
            old_x = x
            continue

        N += 1
        # Guard against old_x being None on the very first production step
        # (can happen when thermalization_steps == 0).
        if old_x is not None and not np.allclose(x, old_x):
            accepted += 1
        old_x = x

        fx_sum += fx
        fx_sum2 += fx * fx

        if dbeta == 0.0:
            continue

        if f_base is None:
            # First production sample: record baseline for numerical stability.
            f_base = fx
            continue

        # log-sum-exp accumulation: log(sum_i exp(-dbeta*(fx_i - f_base)))
        # Uses the identity log(A+B) = log(A) + log(1 + exp(log(B)-log(A)))
        # with the larger term as logA for numerical stability.
        logA = dlogZ
        logB = -dbeta * (fx - f_base)
        if logB > logA:
            logA, logB = logB, logA
        dlogZ = logA + np.log1p(np.exp(logB - logA))

    if N == 0:
        return np.nan, np.nan, 0.0, np.nan

    mean = fx_sum / N
    err = np.sqrt((fx_sum2 / N - mean ** 2) / (N - 1)) if N > 1 else np.nan
    acceptance = accepted / N

    if dbeta != 0.0 and f_base is not None:
        # Normalise: log((1/N) * sum) + shift back by f_base
        dlogZ -= np.log(N)
        dlogZ += -dbeta * f_base

    return mean, err, dlogZ, acceptance


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def separateT(
    Ts: np.ndarray,
    nwalkers: int,
    output_dir: PathLike,
    comm: Optional[MPI.Comm],
    use_beta: bool,
    buffer_size: int = 10000,
) -> None:
    """
    Separates and processes temperature data for quantum beam diffraction experiments.

    Reads each rank's ``result.txt``, redistributes entries to the rank that owns
    their temperature via MPI alltoall, and writes per-temperature ``result_T*.txt``
    files.  Large files are processed in chunks of *buffer_size* lines so that
    memory usage is bounded.

    Parameters
    ----------
    Ts : np.ndarray
        Array of temperature (or beta) values, shared across all ranks.
    nwalkers : int
        Number of walkers per MPI rank.  Must satisfy ``len(Ts) == nwalkers * mpisize``.
    output_dir : PathLike
        Root output directory.  Rank *r* reads from ``output_dir/<r>/result.txt``.
    comm : MPI.Comm, optional
        MPI communicator.  Pass ``None`` for serial execution.
    use_beta : bool
        Write ``# beta = …`` headers instead of ``# T = …`` when ``True``.
    buffer_size : int, optional
        Maximum number of lines read per chunk (rounded up to a multiple of
        *nwalkers*).  Default is 10000.
    """
    if comm is None:
        mpisize = 1
        mpirank = 0
    else:
        mpisize = comm.size
        mpirank = comm.rank

    # Temperatures are used as routing/index keys (both as ``str(T)`` for the
    # per-rank buckets and as the raw value for T2idx). Duplicate values -- or
    # values whose string representation collides -- would make distinct
    # replicas share one output file and silently lose data. Separation by
    # temperature is only well-defined for distinct temperatures, so reject the
    # ambiguous case explicitly instead of producing wrong output.
    if len(set(map(str, Ts))) != len(Ts):
        raise ValueError(
            "separateT requires distinct temperature/beta values, "
            f"but got duplicates in {list(Ts)}"
        )

    # Round up so that each buffer covers whole walker-groups.
    buffer_size = int(np.ceil(buffer_size / nwalkers)) * nwalkers
    output_dir = pathlib.Path(output_dir)
    proc_dir = output_dir / str(mpirank)

    T2idx = {T: i for i, T in enumerate(Ts)}

    # Build per-rank routing table and empty bucket structure.
    T2rank: dict[str, int] = {}
    results: list[dict[str, list[Entry]]] = []
    for rank, Ts_local in enumerate(np.array_split(Ts, mpisize)):
        d: dict[str, list[Entry]] = {}
        for T in Ts_local:
            T2rank[str(T)] = rank
            d[str(T)] = []
        results.append(d)

    local_Ts = Ts[mpirank * nwalkers : (mpirank + 1) * nwalkers]

    # Write per-temperature output file headers.
    label = "beta" if use_beta else "T"
    for T in local_Ts:
        with open(output_dir / f"result_T{T2idx[T]}.txt", "w") as f_out:
            f_out.write(f"# {label} = {T}\n")

    # Read result.txt in chunks, redistribute each chunk, and write to output.
    with open(proc_dir / "result.txt") as f_in:
        while True:
            _clear_results(results)
            entries: list[Entry] = []
            reached_eof = False

            for _ in range(buffer_size):
                raw = f_in.readline()
                if raw == "":
                    reached_eof = True
                    break
                entry = _parse_result_line(raw, mpirank, nwalkers)
                if entry is not None:
                    entries.append(entry)

            _distribute_entries(entries, T2rank, results)
            results2 = comm.alltoall(results) if mpisize > 1 else results
            merged = _merge_results(results2)

            for T in local_Ts:
                _write_entries_to_file(
                    merged[str(T)], output_dir / f"result_T{T2idx[T]}.txt"
                )

            if reached_eof:
                break


def calculate_statistics_from_separated_files(
    Ts: np.ndarray,
    output_dir: PathLike,
    thermalization_steps: int,
    comm: Optional[MPI.Comm],
) -> None:
    """
    Calculate and save statistical quantities (means and errors of f(x) and partition function) from separated files generated by separateT.

    This function reads the separated files, ``result_T<Tindex>.txt`` in ``output_dir``,
    generated by separateT.
    The output file is ``fx.txt`` in ``output_dir``. The format is described as a header as follows:

    .. code-block::

        # $1: 1/T
        # $2: mean of f(x)
        # $3: standard error of f(x)
        # $4: number of replicas [Not used for exchange MC]
        # $5: log(Z/Z0)
        # $6: acceptance ratio

    Parameters
    ----------
    Ts : np.ndarray
        Array of temperature values.
    output_dir : PathLike
        Directory to store the output files.
    thermalization_steps : int
        Number of steps to discard for thermalization.
    comm : MPI.Comm, optional
        MPI communicator for parallel processing.
    """
    if comm is None:
        mpisize = 1
        mpirank = 0
    else:
        mpisize = comm.size
        mpirank = comm.rank
    output_dir = pathlib.Path(output_dir)

    numT = len(Ts)
    T_is_ascending = Ts[0] < Ts[1] if numT > 1 else True
    local_Tindices = np.array_split(np.arange(numT), mpisize)[mpirank]

    fx_means = np.zeros(numT)
    fx_errors = np.zeros(numT)
    dlogZs = np.zeros(numT)
    acceptances = np.zeros(numT)

    for Tindex in local_Tindices:
        T = Ts[Tindex]

        if T_is_ascending:
            lowerTindex = Tindex - 1
        else:
            lowerTindex = Tindex + 1
        if 0 <= lowerTindex < numT:
            lowerT = Ts[lowerTindex]
            dbeta = 1.0 / lowerT - 1.0 / T
        else:
            lowerTindex = -1
            dbeta = 0.0

        samples = _read_result_T_entries(output_dir / f"result_T{Tindex}.txt")
        mean, err, dlogZ, acceptance = _compute_temperature_statistics(
            samples, thermalization_steps, dbeta
        )

        fx_means[Tindex] = mean
        fx_errors[Tindex] = err
        acceptances[Tindex] = acceptance
        if lowerTindex >= 0:
            dlogZs[lowerTindex] = dlogZ

    if comm is not None and mpisize > 1:
        buffer = np.zeros(numT)
        comm.Allreduce(dlogZs, buffer)
        dlogZs[:] = buffer

        comm.Allreduce(fx_means, buffer)
        fx_means[:] = buffer

        comm.Allreduce(fx_errors, buffer)
        fx_errors[:] = buffer

        comm.Allreduce(acceptances, buffer)
        acceptances[:] = buffer

    if mpirank == 0:
        dlogZ_acc = 0.0
        with open(output_dir / "fx.txt", "w") as f_out:
            f_out.write("# $1: 1/T\n")
            f_out.write("# $2: mean of f(x)\n")
            f_out.write("# $3: standard error of f(x)\n")
            f_out.write("# $4: number of replicas [Not used for exchange MC]\n")
            f_out.write("# $5: log(Z/Z0)\n")
            f_out.write("# $6: acceptance ratio\n")
            Tindices = np.arange(numT)
            if T_is_ascending:
                Tindices = Tindices[::-1]
            for Tindex in Tindices:
                dlogZ_acc += dlogZs[Tindex]
                T = Ts[Tindex]
                f_out.write(f"{1/T}")
                f_out.write(f" {fx_means[Tindex]}")
                f_out.write(f" {fx_errors[Tindex]}")
                f_out.write(f" {numT}")
                f_out.write(f" {dlogZ_acc}")
                f_out.write(f" {acceptances[Tindex]}")
                f_out.write("\n")
