# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import IO, Optional, TYPE_CHECKING

import numpy as np
from scipy.linalg import lu, solve_triangular
import odatse
import odatse.domain

if TYPE_CHECKING:
    from mpi4py import MPI


def maxvol(
    mat: np.ndarray, tol: float = 1.001, max_it: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    n, r = mat.shape
    P, L, U = lu(mat, check_finite=False, p_indices=True)
    row_idxs = P.argsort()[:r]
    Q = solve_triangular(U, mat.T, trans=1, check_finite=False)
    B = solve_triangular(L[:r, :], Q, trans=1, lower=True, check_finite=False).T
    for _ in range(max_it):
        i, j = np.divmod(np.abs(B).argmax(), r)
        if np.abs(B[i, j]) <= tol:
            break
        Bi = B[i, :].copy()  # to avoid changing B
        Bi[j] -= 1
        B -= np.outer(B[:, j], Bi) / B[i, j]
        row_idxs[j] = i
    return row_idxs, B


def find_rows(z: np.ndarray, tol: float = 1.001, max_it: int = 1000) -> np.ndarray:
    q, r = np.linalg.qr(z)  # QR decompose to deal with singular z
    if q.shape[0] <= q.shape[1]:  # bypass if matrix is wide and not tall
        row_idxs = np.arange(q.shape[0])
    else:
        row_idxs = maxvol(q, tol, max_it)[0]
    # reference implementation forces row idx of max entry in z to be in row list
    max_idx = np.divmod(np.abs(z).argmax(), z.shape[1])
    if max_idx[0] not in row_idxs:
        row_idxs[-1] = max_idx[0]
    return row_idxs


def update_right(
    grid: np.ndarray,
    cur_poi: Optional[np.ndarray],
    cur_n_points: int,
    cur_rank: int,
    idx_list: np.ndarray,
) -> np.ndarray:
    w2 = np.kron(
        grid, np.ones((cur_rank if cur_poi is not None else 1, 1), dtype=int)
    )
    if cur_poi is not None:  # do nothing if at edge
        w1 = np.kron(np.ones((cur_n_points, 1), dtype=int), cur_poi)
        w2 = np.concatenate([w1, w2], axis=1)
    return w2[idx_list, :]


def update_left(
    grid: np.ndarray,
    next_poi: Optional[np.ndarray],
    cur_n_points: int,
    next_rank: int,
    idx_list: np.ndarray,
) -> np.ndarray:
    w1 = np.kron(
        np.ones((next_rank if next_poi is not None else 1, 1), dtype=int), grid
    )
    if next_poi is not None:  # do nothing if at edge
        w2 = np.kron(next_poi, np.ones((cur_n_points, 1), dtype=int))
        w1 = np.concatenate([w1, w2], axis=1)
    return w1[idx_list, :]


def fuse_pois(
    grid: np.ndarray,
    cur_poi: Optional[np.ndarray],
    next_poi: Optional[np.ndarray],
    cur_rank: int,
    cur_n_points: int,
    next_rank: int,
) -> np.ndarray:
    w2 = np.kron(
        np.kron(
            np.ones((next_rank if next_poi is not None else 1, 1), dtype=int), grid
        ),
        np.ones((cur_rank if cur_poi is not None else 1, 1), dtype=int),
    )
    if cur_poi is not None:
        w1 = np.kron(np.ones((cur_n_points * next_rank, 1), dtype=int), cur_poi)
        w2 = np.concatenate([w1, w2], axis=1)
    if next_poi is not None:
        w3 = np.kron(next_poi, np.ones((cur_rank * cur_n_points, 1), dtype=int))
        w2 = np.concatenate([w2, w3], axis=1)
    return w2


class Algorithm(odatse.algorithm.AlgorithmBase):
    """
    Tensor-Train Optimization (TTOpt) algorithm implementation.

    This class implements the TTOpt algorithm. It is an optimization method
    that searches for function minima by considering an implicit tensor-train
    decomposition of a large tensor whose elements correspond to evaluation
    of the target function at various points in the parameter space.

    Attributes
    ----------
    bounds : np.ndarray
        Lower and upper bounds along each dimension.
        It is constructed from a Domain object based on the min_list and max_list parameters.
    p_points: np.ndarray
        The number of coarse modes in the parameter space for each dimension (default: [2]*n_dim).
        Each dimension is effectively discretized into p^q uniformly-spaced points. It is recommended to have p=2 along each dimension.
    q_points: np.ndarray
        The number of fine submodes in the parameter space for each dimension (default: [1]*n_dim).
        Each dimension is effectively discretized into p^q uniformly-spaced points. This parameter determines the fineness of each mode.
    r_max: int
        The maximum rank of the implicit tensor-train decomposition (default: 4).
    max_f_eval: int
        The maximum number of function evaluations (default: 10000).
    maxvol_tol: float
        The tolerance needed for early stopping in the maxvol subroutine.
        Meaningful values are larger than 1 (default: 1.001).
    maxvol_max_it: int
        The maximum number of maxvol iterations each time the maxvol subroutine is called (default: 1000).
    save_eval_history: bool
        If True, stream ``ttopt_eval_history.txt`` under the output directory while running (rank 0
        only; default: True). Rows are flushed when the in-memory buffer reaches
        ``eval_history_buffer_rows`` points.
    eval_history_buffer_rows: int
        Number of evaluation rows to buffer in memory before appending to disk (default: 256).
    init_points: Optional[np.ndarray]
        Initial points in physical coordinates to evaluate at the start of optimization. (default: None).
    xopt: Optional[np.ndarray]
        The optimal solution.
    fopt: Optional[float]
        The optimal function value.
    """

    bounds: np.ndarray
    p_points: np.ndarray
    q_points: np.ndarray
    r_max: int
    max_f_eval: int
    maxvol_tol: float
    maxvol_max_it: int
    save_eval_history: bool
    eval_history_buffer_rows: int
    init_points: np.ndarray

    xopt: Optional[np.ndarray]
    fopt: Optional[float]

    f_eval_count: int
    f_eval_count_history: list[int]
    xopt_history: list[np.ndarray]
    fopt_history: list[float]

    n_points: np.ndarray
    n_q_points: np.ndarray
    n_q_dims: int
    grids: list[np.ndarray]
    tt_ranks: np.ndarray
    poi: list[Optional[np.ndarray]]

    def __init__(
        self,
        info: odatse.Info,
        runner: Optional[odatse.Runner] = None,
        domain: Optional[odatse.domain.Region] = None,
        run_mode: str = "initial",
        mpicomm: Optional["MPI.Comm"] = None,
    ) -> None:
        """
        Initialize the Algorithm class.

        Parameters
        ----------
        info : odatse.Info
            Information object containing algorithm parameters.
        runner : odatse.Runner, optional
            Runner object for executing the algorithm.
        run_mode : str, optional
            Mode to run the algorithm in, by default "initial".
        mpicomm : MPI.Comm
            MPI communicator to use for parallelization.
            If not provided, the default MPI communicator (MPI.COMM_WORLD) will be used if mpi4py is installed.
        """

        super().__init__(info=info, runner=runner, run_mode=run_mode, mpicomm=mpicomm)

        if self.runner is None:
            raise ValueError("The TTOpt algorithm requires a runner instance.")

        if domain and isinstance(domain, odatse.domain.Region):
            self.domain = domain
        else:
            self.domain = odatse.domain.Region(info)

        self.min_list = self.domain.min_list
        self.max_list = self.domain.max_list
        self.bounds = np.vstack([self.min_list, self.max_list]).T

        info_ttopt: dict = info.algorithm.get("ttopt", {})

        p_points = info_ttopt.get("p_points", 2)
        if isinstance(p_points, int):
            p_points = [p_points] * self.dimension

        q_points = info_ttopt.get("q_points", 1)
        if isinstance(q_points, int):
            q_points = [q_points] * self.dimension

        self.p_points = np.asarray(p_points, dtype=int)
        self.q_points = np.asarray(q_points, dtype=int)

        if self.p_points.shape[0] != self.dimension:
            raise ValueError("ttopt.p_points must match the problem dimension.")
        if self.q_points.shape[0] != self.dimension:
            raise ValueError("ttopt.q_points must match the problem dimension.")

        self.r_max = int(info_ttopt.get("r_max", 4))
        self.max_f_eval = int(info_ttopt.get("max_f_eval", 10000))
        self.maxvol_tol = float(info_ttopt.get("maxvol_tol", 1.001))
        self.maxvol_max_it = int(info_ttopt.get("maxvol_max_it", 1000))
        self.save_eval_history = bool(info_ttopt.get("save_eval_history", True))
        _buf_rows = int(info_ttopt.get("eval_history_buffer_rows", 256))
        self.eval_history_buffer_rows = max(1, _buf_rows)
        self.init_points = np.unique(
            np.asarray(info_ttopt.get("init_points", []), dtype=float), axis=0
        )

        self.xopt = None
        self.fopt = np.inf

        # memoize known points
        self.cache = {}
        self.cache_hits = 0

        self._show_parameters()

    def _prepare(self) -> None:
        assert self.bounds.shape[0] == self.p_points.shape[0] == self.q_points.shape[0]
        self.n_points = (
            self.p_points**self.q_points
        )  # number of points in full discretized interval
        self.n_q_points = np.asarray(
            [
                self.p_points[i]
                for i in range(self.bounds.shape[0])
                for _ in range(self.q_points[i])
            ]
        )
        self.n_q_dims = self.n_q_points.shape[0]  # number of quantized dims
        self.grids = [
            np.reshape(np.arange(self.n_q_points[i], dtype=int), (-1, 1))
            for i in range(self.n_q_dims)
        ]
        self.tt_ranks = np.ones(self.n_q_dims + 1, dtype=int)
        for i in range(1, self.n_q_dims):
            self.tt_ranks[i] = np.min(
                [self.tt_ranks[i - 1] * self.n_q_points[i - 1], self.r_max]
            )  # rank is min(L*C,r_max)

        self.f_eval_count = 0
        self.f_eval_count_history = []
        self.xopt_history = []
        self.fopt_history = []
        if self.mpirank == 0 and getattr(self, "_eval_hist_file", None) is not None:
            self._eval_hist_file.close()
        self._eval_hist_file: Optional[IO[str]] = None
        self._eval_hist_next_row = 1
        self._eval_hist_pending_x: list[np.ndarray] = []
        self._eval_hist_pending_f: list[np.ndarray] = []
        self._eval_hist_pending_nrows = 0

        # process init_points
        if self.init_points.shape[0] > 0:
            n_init_points = self.init_points.shape[0]
            n_dims = self.bounds.shape[0]

            # convert to indices
            init_indices = np.zeros((n_init_points, n_dims), dtype=float)
            for dim in range(n_dims):
                # inverse of coordinate conversion in f_eval
                t = (self.init_points[:, dim] - self.bounds[dim, 0]) / (
                    self.bounds[dim, 1] - self.bounds[dim, 0]
                )
                init_indices[:, dim] = t * (self.n_points[dim] - 1)
                init_indices[:, dim] = np.round(init_indices[:, dim])

            # convert to multi-indices
            self.init_q_pois = np.zeros((n_init_points, self.n_q_dims), dtype=int)
            start = 0
            for dim in range(n_dims):
                n_idx = init_indices[:, dim].astype(int)
                n_idx = np.clip(n_idx, 0, self.n_points[dim] - 1)
                for i in range(n_init_points):
                    multi_idx = np.unravel_index(
                        n_idx[i],
                        self.n_q_points[start : start + self.q_points[dim]],
                        order="F",
                    )
                    self.init_q_pois[i, start : start + self.q_points[dim]] = multi_idx
                start += self.q_points[dim]

            # evaluate early to get values for z
            f_vals_init, self.xopt, self.fopt = self.f_eval(
                self.bounds,
                self.p_points,
                self.q_points,
                self.n_points,
                self.n_q_points,
                self.init_q_pois,
                self.xopt,
                self.fopt,
                self.runner,
            )
            self.f_eval_count += len(self.init_q_pois)
            self.f_eval_count_history.append(self.f_eval_count)
            self.xopt_history.append(self.xopt)
            self.fopt_history.append(self.fopt)
        else:
            self.init_q_pois = np.zeros((0, self.n_q_dims), dtype=int)
            f_vals_init = np.zeros(0)

        self.poi = [None for _ in range(self.n_q_dims + 1)]
        for i in range(self.n_q_dims - 1):
            if self.init_q_pois.shape[0] > 0: # if init_points > 0
                # fill z, an mps tensor with size (L, C, R), with known values from init_q_pois
                z = np.zeros(
                    (self.tt_ranks[i], self.n_q_points[i], self.tt_ranks[i + 1])
                )
                # p/l/c/r are the point, left, center, right indices respectively
                for p in range(self.init_q_pois.shape[0]):
                    if i == 0:
                        l = 0
                    else:
                        matches = np.all(self.poi[i] == self.init_q_pois[p, :i], axis=1)
                        if not np.any(matches):
                            continue
                        l = np.where(matches)[0][0]

                    c = self.init_q_pois[p, i]
                    r = (
                        p % self.tt_ranks[i + 1]
                    )  # in case there are more points than ranks
                    z[l, c, r] = max(z[l, c, r], np.exp(self.fopt - f_vals_init[p]))
                # fallback if all elements are 0
                if np.sum(z) == 0:
                    z = self.rng.normal(size=z.shape)
            else:
                z = self.rng.normal(
                    size=(self.tt_ranks[i], self.n_q_points[i], self.tt_ranks[i + 1])
                )  # matrix is (L*C,R)

            z = np.reshape(
                z,
                (self.tt_ranks[i] * self.n_q_points[i], self.tt_ranks[i + 1]),
                order="F",
            )
            if self.mpisize > 1:  # make state uniform across all ranks
                assert self.mpicomm is not None
                z = self.mpicomm.bcast(z, root=0)
            row_idxs = find_rows(z, self.maxvol_tol, self.maxvol_max_it)
            next_poi = update_right(
                self.grids[i],
                self.poi[i],
                self.n_q_points[i],
                self.tt_ranks[i],
                row_idxs,
            )
            self.tt_ranks[i + 1] = next_poi.shape[0]
            self.poi[i + 1] = next_poi
        if self.mpirank == 0:
            with open(self.output_dir / "ttopt_hyperparameters.txt", "w") as f:
                f.write(f"nprocs = {self.mpisize}\n")
                f.write(f"bounds = {self.bounds.tolist()}\n")
                f.write(f"p_points = {self.p_points}\n")
                f.write(f"q_points = {self.q_points}\n")
                f.write(f"r_max = {self.r_max}\n")
                f.write(f"max_f_eval = {self.max_f_eval}\n")
                f.write(f"maxvol_tol = {self.maxvol_tol}\n")
                f.write(f"maxvol_max_it = {self.maxvol_max_it}\n")
                f.write(f"save_eval_history = {self.save_eval_history}\n")
                f.write(f"eval_history_buffer_rows = {self.eval_history_buffer_rows}\n")

    def _run(self) -> None:
        run = self.runner
        sweeps = [
            (True, list(range(self.n_q_dims - 1, -1, -1))),
            (False, list(range(0, self.n_q_dims))),
        ]
        while True:
            for r2l, sweep_range in sweeps:
                for i in sweep_range:
                    todo_q_pois = fuse_pois(
                        self.grids[i],
                        self.poi[i],
                        self.poi[i + 1],
                        self.tt_ranks[i],
                        self.n_q_points[i],
                        self.tt_ranks[i + 1],
                    )
                    f_vals, self.xopt, self.fopt = self.f_eval(
                        self.bounds,
                        self.p_points,
                        self.q_points,
                        self.n_points,
                        self.n_q_points,
                        todo_q_pois,
                        self.xopt,
                        self.fopt,
                        run,
                    )
                    self.f_eval_count += len(todo_q_pois)
                    self.f_eval_count_history.append(self.f_eval_count)
                    self.xopt_history.append(self.xopt)
                    self.fopt_history.append(self.fopt)
                    if self.f_eval_count >= self.max_f_eval:
                        return
                    # map the values so that they are all positive, in a way that the maximal modulus element is also the minimum
                    z = np.exp(
                        self.fopt - f_vals
                    )  # reference implementation uses this smoothing function
                    # z=(np.pi/2)-np.arctan(f_vals-min_f) # paper uses this smoothing function
                    if r2l:
                        z = np.reshape(
                            z,
                            (
                                self.tt_ranks[i],
                                self.n_q_points[i] * self.tt_ranks[i + 1],
                            ),
                            order="F",
                        )
                        row_idxs = find_rows(
                            z.T, self.maxvol_tol, self.maxvol_max_it
                        )
                        if i != 0:
                            next_poi = update_left(
                                self.grids[i],
                                self.poi[i + 1],
                                self.n_q_points[i],
                                self.tt_ranks[i + 1],
                                row_idxs,
                            )
                            self.tt_ranks[i] = next_poi.shape[0]
                            self.poi[i] = next_poi
                    else:
                        z = np.reshape(
                            z,
                            (
                                self.tt_ranks[i] * self.n_q_points[i],
                                self.tt_ranks[i + 1],
                            ),
                            order="F",
                        )
                        row_idxs = find_rows(
                            z, self.maxvol_tol, self.maxvol_max_it
                        )
                        if i != self.n_q_dims - 1:
                            next_poi = update_right(
                                self.grids[i],
                                self.poi[i],
                                self.n_q_points[i],
                                self.tt_ranks[i],
                                row_idxs,
                            )
                            self.tt_ranks[i + 1] = next_poi.shape[0]
                            self.poi[i + 1] = next_poi

    def _eval_hist_append_batch(
        self, todo_pois: np.ndarray, f_vals: np.ndarray
    ) -> None:
        if not self.save_eval_history or self.mpirank != 0:
            return
        n = int(f_vals.shape[0])
        if n == 0:
            return
        self._eval_hist_pending_x.append(np.asarray(todo_pois).copy())
        self._eval_hist_pending_f.append(np.asarray(f_vals).copy())
        self._eval_hist_pending_nrows += n
        if self._eval_hist_pending_nrows >= self.eval_history_buffer_rows:
            self._eval_hist_flush(force=True)

    def _eval_hist_flush(self, *, force: bool) -> None:
        if not self.save_eval_history or self.mpirank != 0:
            return
        if self._eval_hist_pending_nrows == 0:
            return
        if not force and self._eval_hist_pending_nrows < self.eval_history_buffer_rows:
            return
        xs = np.vstack(self._eval_hist_pending_x)
        fs = np.concatenate(self._eval_hist_pending_f)
        path = self.output_dir / "ttopt_eval_history.txt"
        if self._eval_hist_file is None:
            self._eval_hist_file = open(
                path,
                "w",
                encoding="utf-8",
                newline="\n",
            )
            hf = self._eval_hist_file
            hf.write("# $1: row index (order of batches in the run)\n")
            for d in range(self.dimension):
                hf.write(f"# ${d+2}: {self.label_list[d]}\n")
            hf.write(f"# ${self.dimension+2}: f(x)\n")
        hf = self._eval_hist_file
        for row in range(xs.shape[0]):
            hf.write(f"{self._eval_hist_next_row:d}")
            self._eval_hist_next_row += 1
            for d in range(self.dimension):
                hf.write(f" {xs[row, d]:.15e}")
            hf.write(f" {fs[row]:.15e}\n")
        self._eval_hist_pending_x.clear()
        self._eval_hist_pending_f.clear()
        self._eval_hist_pending_nrows = 0

    def _eval_hist_finalize(self) -> None:
        if not self.save_eval_history or self.mpirank != 0:
            return
        self._eval_hist_flush(force=True)
        if self._eval_hist_file is not None:
            self._eval_hist_file.close()
            self._eval_hist_file = None

    def _post(self) -> dict:
        """Collect the best solution across ranks."""
        result = {
            "x": self.xopt,
            "fx": self.fopt,
        }

        opt_history = list(
            zip(self.f_eval_count_history, self.xopt_history, self.fopt_history)
        )
        if self.mpirank == 0:
            self._eval_hist_finalize()
            with open("ttopt_history.txt", "w") as f:
                f.write("# $1: count\n")
                for d in range(self.dimension):
                    f.write(f"# ${d+2}: x_opt[{d}]\n")
                f.write(f"# ${self.dimension+2}: fx_opt\n")
                for entry in opt_history:
                    f.write(f"{entry[0]:d}")
                    for x in entry[1]:
                        f.write(f" {x:.15e}")
                    f.write(f" {entry[2]:.15e}\n")

        if self.mpisize > 1:
            assert self.mpicomm is not None
            results = self.mpicomm.allgather(result)
        else:
            results = [result]

        xs = [v["x"] for v in results]
        fxs = [v["fx"] for v in results]

        idx = np.argmin(fxs)
        result = {"x": xs[idx], "fx": fxs[idx], "opt_history": opt_history}
        if self.mpirank == 0:
            print(f"Best solution: x = {result['x']}, f(x) = {result['fx']}")
            hitrate = (
                100 * self.cache_hits / self.f_eval_count
                if self.f_eval_count > 0
                else 0
            )
            print(f"Cache hitrate: {hitrate:.2f}%")
            with open("res.txt", "w") as fp:
                fp.write(f"fx = {result['fx']}\n")
                for label, val in zip(self.label_list, result["x"]):
                    fp.write(f"{label} = {val}\n")
        return result

    def f_eval(
        self,
        bounds: np.ndarray,
        ps: np.ndarray,
        qs: np.ndarray,
        n_points: np.ndarray,
        n_q_points: np.ndarray,
        todo_q_pois: np.ndarray,
        min_params: np.ndarray,
        min_f: float,
        runner,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        # convert p,q idxs to n idxs
        n_dims = bounds.shape[0]
        todo_pois = np.zeros((todo_q_pois.shape[0], n_dims), dtype=float)
        start = 0
        for dim_idx in range(n_dims):
            submat = todo_q_pois[:, start : start + qs[dim_idx]]
            todo_pois[:, dim_idx] = np.ravel_multi_index(
                submat.T, n_q_points[start : start + qs[dim_idx]], order="F"
            )
            start += qs[dim_idx]
        # convert idxs to coordinates
        todo_pois = (
            todo_pois / (n_points - 1) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        )  # scale before shift to minimize fp error
        f_vals = np.empty(todo_pois.shape[0])
        eval_idxs = []
        eval_pois = []
        for i, row in enumerate(todo_pois):
            row_tuple = tuple(row)
            if row_tuple in self.cache:
                self.cache_hits += 1
                f_vals[i] = self.cache[row_tuple]
            else:
                eval_idxs.append(i)
                eval_pois.append(row)

        # parallelization
        eval_f_vals = []
        if len(eval_idxs) > 0:
            if self.mpisize > 1:
                assert self.mpicomm is not None
                split = np.array_split(np.arange(len(eval_idxs)), self.mpisize)
                my_local_idxs = split[self.mpirank]
                my_f_vals = [runner.submit(eval_pois[k]) for k in my_local_idxs]
                all_f_vals = self.mpicomm.allgather(my_f_vals)
                eval_f_vals = [v for sublist in all_f_vals for v in sublist]
            else:
                eval_f_vals = [
                    runner.submit(eval_pois[k]) for k in range(len(eval_idxs))
                ]
            for i, idx in enumerate(eval_idxs):
                val = eval_f_vals[i]
                self.cache[tuple(eval_pois[i])] = val
                f_vals[idx] = val
        best_idxs = np.argmin(f_vals)
        best_params = todo_pois[best_idxs]
        best_f = f_vals[best_idxs]
        if best_f < min_f:
            min_params, min_f = (best_params, best_f)
        self._eval_hist_append_batch(todo_pois, f_vals)
        return f_vals, min_params, min_f
