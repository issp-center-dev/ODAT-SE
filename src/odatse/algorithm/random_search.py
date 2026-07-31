# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, Optional

from pathlib import Path
from io import open
import numpy as np
import os
import time

import odatse

from .mapper_mpi_base import Algorithm as MapperMPIAlgorithm
from ._iterator import RandomIterator, ListIterator


class Algorithm(MapperMPIAlgorithm):
    """
    Algorithm class that evaluates the objective function at random points.
    Inherits from odatse.algorithm.mapper_mpi_base.Algorithm.
    """
    mesh_list: list[Union[int, float]]

    # --cont extends the run with additional points. Supported for the
    # "random" mode and for the nested quasi-random sequences (sobol,
    # halton); rejected for latin (see _check_continue).
    _continuable: bool = True

    def __init__(self,
                 info: odatse.Info,
                 runner: Optional[odatse.Runner] = None,
                 run_mode: str = "initial",
    ) -> None:
        """
        Initialize the Algorithm instance.

        Parameters
        ----------
        info : Info
            Information object containing algorithm parameters.
        runner : Runner
            Optional runner object for submitting tasks.
        run_mode : str
            Mode to run the algorithm, defaults to "initial".
        """
        super().__init__(info=info, runner=runner, run_mode=run_mode)

        info_mode = info.algorithm.get("mode", None)
        if info_mode is None:
            mode = "random"
        else:
            mode = info_mode.get("mode", "random")

        info_param = info.algorithm.get("param", {})

        self._point_mode = mode
        self._sequence = info_mode.get("sequence", "sobol") if mode == "quasi-random" else None
        # scipy QMC engine used to generate the sequence (algorithm rank 0
        # only, None elsewhere). Kept on the instance and included in the
        # checkpoint so that --cont can draw further points of the same
        # sequence.
        self._sampler = None
        self._num_points = None

        if odatse.mpi.run_on_algorithm():
            if mode == "random":
                iter = self._random_iterator(info_param, self.rng)
            elif mode == "quasi-random":
                seq = self._sequence
                seed = info.algorithm.get("seed", None)
                iter = self._quasi_random_iterator(info_param, seq, seed)
            else:
                raise ValueError("ERROR: algorithm.mode.mode = {} is not supported".format(mode))
            # delayed setup
            self._iter = iter
        else:
            self._iter = None

    def _random_iterator(self, info_param, rng):
        """
        Setup the grid based on min, max, and num lists.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the grid.
        """
        if "min_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.min_list is not defined in the input")
        min_list = info_param["min_list"]

        if "max_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.max_list is not defined in the input")
        max_list = info_param["max_list"]

        if "num_points" not in info_param:
            raise ValueError("ERROR: algorithm.param.num_points is not defined in the input")
        num_points = info_param["num_points"]

        if len(min_list) != len(max_list):
            raise ValueError("ERROR: lengths of min_list and max_list do not match")
        if num_points <= 0:
            raise ValueError("ERROR: num_points must be positive")

        self._min_list = min_list
        self._max_list = max_list
        self._num_points = num_points

        return RandomIterator(min_list, max_list, num_points, rng)

    def _quasi_random_iterator(self, info_param, seq, seed=None):
        """
        Setup a quasi-random (low-discrepancy) point sequence.

        Parameters
        ----------
        info_param
            Dictionary containing parameters for setting up the points.
        seq : str
            Sequence type: "sobol", "halton", or "latin".
        seed : int, optional
            Seed for the scrambling of the sequence. The sequence is
            generated on the algorithm-rank-0 process only, so a single
            integer makes the whole point set reproducible independently
            of the MPI configuration. If None, the scrambling differs
            from run to run.
        """
        from scipy.stats import qmc

        if "min_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.min_list is not defined in the input")
        min_list = info_param["min_list"]

        if "max_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.max_list is not defined in the input")
        max_list = info_param["max_list"]

        if "num_points" not in info_param:
            raise ValueError("ERROR: algorithm.param.num_points is not defined in the input")
        num_points = info_param["num_points"]

        if len(min_list) != len(max_list):
            raise ValueError("ERROR: lengths of min_list and max_list do not match")
        if num_points <= 0:
            raise ValueError("ERROR: num_points must be positive")

        self._min_list = min_list
        self._max_list = max_list
        self._num_points = num_points

        if odatse.mpi.algrank() == 0:
            d = len(min_list)

            if seq == "sobol":
                sampler = qmc.Sobol(d, scramble=True, optimization=None, seed=seed)
            elif seq == "halton":
                sampler = qmc.Halton(d, scramble=True, optimization=None, seed=seed)
            elif seq == "latin":
                sampler = qmc.LatinHypercube(d, scramble=True, strength=1, optimization=None, seed=seed)
            else:
                raise ValueError("unknown sequence type {}".format(seq))

            # generate samples on rank 0 all at once
            idx = np.arange(num_points)
            sample = sampler.random(n=num_points)
            #print("discrepancy=", qmc.discrepancy(sample))
            sample = qmc.scale(sample, min_list, max_list)

            data = [[i, *x] for i, x in zip(idx, sample)]
            self._sampler = sampler
        else:
            data = None

        return ListIterator(data)

    def __getstate__(self) -> dict:
        """Return a checkpoint snapshot including the sequence generator.

        Extends the mapper snapshot with the scipy QMC engine (algorithm
        rank 0 only, None elsewhere). The engine state advances as points
        are drawn, so a restored engine continues the sequence exactly
        where the previous run left off, which makes --cont possible even
        without an explicit seed.
        """
        state = super().__getstate__()
        state["sampler"] = self._sampler
        return state

    def _apply_state(self, data: dict, mode: str = "resume", restore_rng: bool = True) -> None:
        """Restore algorithm state; in continue mode also extend the run.

        For ``mode="continue"`` the point set is extended to the num_points
        of the new input: the previously evaluated points are kept (restored
        from the checkpoint) and only the additional points are evaluated.
        This is supported for the "random" mode and for the sobol/halton
        quasi-random sequences, whose point sets are nested (the first N
        points of a longer sequence are exactly the N points of the shorter
        one). It is rejected for latin, whose design is not nested.
        """
        if mode == "continue":
            self._check_continue(data)
        super()._apply_state(data, mode=mode, restore_rng=restore_rng)
        self._sampler = data.get("sampler", None)
        if mode == "continue":
            self._extend_points(data)

    def _check_continue(self, data: dict) -> None:
        """Validate that the new input is a legal extension of the old run."""
        prev = data["info"]
        prev_mode_tbl = prev.get("mode", None) or {}
        prev_mode = prev_mode_tbl.get("mode", "random")
        if prev_mode != self._point_mode:
            raise RuntimeError(
                "cannot continue: algorithm.mode.mode changed from {} to {}".format(
                    prev_mode, self._point_mode))
        if self._point_mode == "quasi-random":
            if self._sequence == "latin":
                raise RuntimeError(
                    "continue mode is not supported for the latin sequence: "
                    "a Latin hypercube design is not nested (the design for a "
                    "larger number of points does not contain the design for "
                    "a smaller one), so the previous evaluations cannot be "
                    "reused. Start a new run instead.")
            prev_seq = prev_mode_tbl.get("sequence", "sobol")
            if prev_seq != self._sequence:
                raise RuntimeError(
                    "cannot continue: algorithm.mode.sequence changed from {} to {}".format(
                        prev_seq, self._sequence))
        prev_param = prev.get("param", {})
        if list(prev_param.get("min_list", [])) != list(self._min_list) \
           or list(prev_param.get("max_list", [])) != list(self._max_list):
            raise RuntimeError(
                "cannot continue: algorithm.param.min_list/max_list changed "
                "from the previous run")
        prev_n = prev_param.get("num_points")
        if prev_n is None or self._num_points < prev_n:
            raise RuntimeError(
                "cannot continue: num_points ({}) is smaller than in the "
                "previous run ({})".format(self._num_points, prev_n))

    def _extend_points(self, data: dict) -> None:
        """Extend the iterator with the additional points (continue mode)."""
        prev_n = data["info"]["param"]["num_points"]
        add = self._num_points - prev_n
        if add == 0:
            return

        if self._point_mode == "random":
            self._iter._extend(self._num_points)
            return

        # quasi-random: draw the additional points from the restored engine
        # on rank 0 and distribute them. The availability of the engine is
        # agreed on collectively first, so that no rank enters the scatter
        # in _extend() when rank 0 has to fail.
        ok = self._sampler is not None if odatse.mpi.algrank() == 0 else None
        if odatse.mpi.algsize() > 1:
            ok = odatse.mpi.algcomm().bcast(ok, root=0)
        if not ok:
            raise RuntimeError(
                "cannot continue: the checkpoint does not contain the state "
                "of the sequence generator (it may have been created by an "
                "older version of ODAT-SE)")

        if odatse.mpi.algrank() == 0:
            from scipy.stats import qmc
            sample = self._sampler.random(n=add)
            sample = qmc.scale(sample, self._min_list, self._max_list)
            ext = [[i, *x] for i, x in zip(range(prev_n, prev_n + add), sample)]
        else:
            ext = None
        self._iter._extend(ext)
