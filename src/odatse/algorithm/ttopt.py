# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Tuple, List, Dict, Union, Optional, TYPE_CHECKING

import numpy as np
from scipy.linalg import lu, solve_triangular
import odatse
import odatse.domain

if TYPE_CHECKING:
    from mpi4py import MPI

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

    xopt: Optional[np.ndarray]
    fopt: Optional[float]

    n_points: np.ndarray
    n_q_points: np.ndarray
    n_q_dims: int
    grids: List[np.ndarray]
    tt_ranks: np.ndarray
    poi: List[Union[None, np.ndarray]]

    def __init__(
        self,
        info: odatse.Info,
        runner: odatse.Runner = None,
        domain: Union[odatse.domain.MeshGrid, odatse.domain.Region] = None,
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

        super().__init__(info=info, runner=runner, run_mode=run_mode)

        if self.runner is None:
            raise ValueError("The TTOpt algorithm requires a runner instance.")

        if domain and isinstance(domain, (odatse.domain.Region, odatse.domain.MeshGrid)):
            self.domain = domain
        else:
            self.domain = odatse.domain.Region(info)

        self.min_list = self.domain.min_list
        self.max_list = self.domain.max_list
        self.bounds = np.vstack([self.min_list, self.max_list]).T

        info_ttopt: Dict = info.algorithm.get("ttopt", {})

        self.p_points = np.asarray(info_ttopt.get("p_points", [2] * self.dimension), dtype=int)
        self.q_points = np.asarray(info_ttopt.get("q_points", [1] * self.dimension), dtype=int)
        self.r_max = int(info_ttopt.get("r_max", 4))
        self.max_f_eval = int(info_ttopt.get("max_f_eval", 10000))
        self.maxvol_tol = float(info_ttopt.get("maxvol_tol", 1.001))
        self.maxvol_max_it = int(info_ttopt.get("maxvol_max_it", 1000))

        self.xopt = None
        self.fopt = np.inf

        if self.p_points.shape[0] != self.dimension:
            raise ValueError("ttopt.p_points must match the problem dimension.")
        if self.q_points.shape[0] != self.dimension:
            raise ValueError("ttopt.q_points must match the problem dimension.")

        self._show_parameters()

    def _prepare(self) -> None:
        assert self.bounds.shape[0] == self.p_points.shape[0] == self.q_points.shape[0]
        self.n_points = self.p_points ** self.q_points # number of points in full discretized interval
        self.n_q_points = np.asarray([self.p_points[i] for i in range(self.bounds.shape[0]) for _ in range(self.q_points[i])])
        self.n_q_dims = self.n_q_points.shape[0] # number of quantized dims
        self.grids = [np.reshape(np.arange(self.n_q_points[i], dtype=int), (-1, 1)) for i in range(self.n_q_dims)]
        self.tt_ranks = np.ones(self.n_q_dims + 1, dtype=int)
        for i in range(1, self.n_q_dims):
            self.tt_ranks[i] = np.min([self.tt_ranks[i - 1] * self.n_q_points[i - 1], self.r_max]) # rank is min(L*C,r_max)
        self.poi = [None for _ in range(self.n_q_dims + 1)]
        for i in range(self.n_q_dims - 1):
            z = self.rng.normal(size=(self.tt_ranks[i], self.n_q_points[i], self.tt_ranks[i + 1])) # matrix is (L*C,R)
            z = np.reshape(z, (self.tt_ranks[i] * self.n_q_points[i], self.tt_ranks[i + 1]), order="F")
            row_idxs = Algorithm.find_rows(z, self.maxvol_tol, self.maxvol_max_it)
            self.poi[i + 1] = Algorithm.update_right(self.grids[i], self.poi[i], self.n_q_points[i], self.tt_ranks[i], row_idxs)
            self.tt_ranks[i + 1] = self.poi[i + 1].shape[0]
        self.f_eval_count = 0

    def _run(self) -> None:
        run = self.runner

        while True:
            for i in range(self.n_q_dims - 1, -1, -1): # sweep right to left, except edges at idx n_q_dims and 0
                todo_q_pois = Algorithm.fuse_pois(self.grids[i], self.poi[i], self.poi[i + 1], self.tt_ranks[i], self.n_q_points[i], self.tt_ranks[i + 1])
                f_vals, self.xopt, self.fopt = Algorithm.f_eval(self.bounds, self.p_points, self.q_points, self.n_points, self.n_q_points, todo_q_pois, self.xopt, self.fopt, run)
                self.f_eval_count += len(todo_q_pois)
                if self.f_eval_count >= self.max_f_eval:
                    return self.xopt, self.fopt
                # map the values so that they are all positive, in a way that the maximal modulus element is also the minimum
                z = np.exp(self.fopt - f_vals) # reference implementation uses this smoothing function
                # z=(np.pi/2)-np.arctan(f_vals-min_f) # paper uses this smoothing function
                # same as (and faster than) np.reshape(z,(n_q_points[i]*tt_ranks[i+1],tt_ranks[i])).T
                z = np.reshape(z, (self.tt_ranks[i], self.n_q_points[i] * self.tt_ranks[i + 1]), order="F")
                row_idxs = Algorithm.find_rows(z.T, self.maxvol_tol, self.maxvol_max_it)
                if i != 0:
                    self.poi[i] = Algorithm.update_left(self.grids[i], self.poi[i + 1], self.n_q_points[i], self.tt_ranks[i + 1], row_idxs)
                    self.tt_ranks[i] = self.poi[i].shape[0]
                # print(self.f_eval_count,self.fopt)
            for i in range(0, self.n_q_dims): # sweep left to right, except edges at idx 0 and n_q_dims
                todo_q_pois = Algorithm.fuse_pois(self.grids[i], self.poi[i], self.poi[i + 1], self.tt_ranks[i], self.n_q_points[i], self.tt_ranks[i + 1])
                f_vals, self.xopt, self.fopt = Algorithm.f_eval(self.bounds, self.p_points, self.q_points, self.n_points, self.n_q_points, todo_q_pois, self.xopt, self.fopt, run)
                self.f_eval_count += len(todo_q_pois)
                if self.f_eval_count >= self.max_f_eval:
                    return self.xopt, self.fopt
                # map the values so that they are all positive, in a way that the maximal modulus element is also the minimum
                z = np.exp(self.fopt - f_vals) # reference implementation uses this smoothing function
                # z=(np.pi/2)-np.arctan(f_vals-min_f) # paper uses this smoothing function
                # same as (and faster than) np.reshape(z,(n_q_points[i]*tt_ranks[i+1],tt_ranks[i])).T
                z = np.reshape(z, (self.tt_ranks[i] * self.n_q_points[i], self.tt_ranks[i + 1]), order="F")
                row_idxs = Algorithm.find_rows(z, self.maxvol_tol, self.maxvol_max_it)
                if i != self.n_q_dims - 1:
                    self.poi[i + 1] = Algorithm.update_right(self.grids[i], self.poi[i], self.n_q_points[i], self.tt_ranks[i], row_idxs)
                    self.tt_ranks[i + 1] = self.poi[i + 1].shape[0]
                # print(self.f_eval_count,self.fopt)

        self.xopt = min_params
        self.fopt = min_f

    def _post(self) -> Dict[str, np.ndarray]:
        """Collect the best solution across ranks."""
        result = {
            "x": self.xopt,
            "fx": self.fopt,
        }

        if self.mpisize > 1:
            results = self.mpicomm.allgather(result)
        else:
            results = [result]

        xs = [v["x"] for v in results]
        fxs = [v["fx"] for v in results]

        idx = np.argmin(fxs)
        print({"x": xs[idx], "fx": fxs[idx]})
        return {"x": xs[idx], "fx": fxs[idx]}
 
    @staticmethod
    def maxvol(mat: np.ndarray, tol: float = 1.001, max_it: int = 1000) -> Tuple[List[int], np.ndarray]:
        n, r = mat.shape
        P, L, U = lu(mat, check_finite=False, p_indices=True)
        row_idxs = P.argsort()[:r]
        Q = solve_triangular(U, mat.T, trans=1, check_finite=False)
        B = solve_triangular(L[:r, :], Q, trans=1, lower=True, check_finite=False).T
        for _ in range(max_it):
            i, j = np.divmod(np.abs(B).argmax(), r)
            if np.abs(B[i, j]) <= tol:
                break
            Bi = B[i, :].copy() # to avoid changing B
            Bi[j] -= 1
            B -= np.outer(B[:, j], Bi) / B[i, j]
            row_idxs[j] = i
        return row_idxs, B

    @staticmethod
    def find_rows(z: np.ndarray, tol: float = 1.001, max_it=1000) -> np.ndarray:
        q, r = np.linalg.qr(z) # QR decompose to deal with singular z
        if q.shape[0] <= q.shape[1]: # bypass if matrix is wide and not tall
            row_idxs = np.arange(q.shape[0])
        else:
            row_idxs, _ = Algorithm.maxvol(q, tol, max_it)
        # reference implementation forces row idx of max entry in z to be in row list
        max_idx = np.divmod(np.abs(z).argmax(), z.shape[1])
        if not max_idx[0] in row_idxs:
            row_idxs[-1] = max_idx[0]
        return row_idxs

    @staticmethod
    def update_right(grid: np.ndarray, cur_poi: np.ndarray, cur_n_points: int, cur_rank: int, idx_list: np.ndarray) -> np.ndarray:
        w2 = np.kron(grid, np.ones((cur_rank if cur_poi is not None else 1, 1), dtype=int))
        if cur_poi is not None: # do nothing if at edge
            w1 = np.kron(np.ones((cur_n_points, 1), dtype=int), cur_poi)
            w2 = np.concatenate([w1, w2], axis=1)
        return w2[idx_list, :]

    @staticmethod
    def update_left(grid: np.ndarray, next_poi: np.ndarray, cur_n_points: int, next_rank: int, idx_list: np.ndarray) -> np.ndarray:
        w1 = np.kron(np.ones((next_rank if next_poi is not None else 1, 1), dtype=int), grid)
        if next_poi is not None: # do nothing if at edge
            w2 = np.kron(next_poi, np.ones((cur_n_points, 1), dtype=int))
            w1 = np.concatenate([w1, w2], axis=1)
        return w1[idx_list, :]
    
    @staticmethod
    def fuse_pois(grid: np.ndarray, cur_poi: np.ndarray, next_poi: np.ndarray, cur_rank: int, cur_n_points: int, next_rank: int):
        w2 = np.kron(np.kron(np.ones((next_rank if next_poi is not None else 1, 1), dtype=int), grid), np.ones((cur_rank if cur_poi is not None else 1, 1), dtype=int))
        if cur_poi is not None:
            w1 = np.kron(np.ones((cur_n_points * next_rank, 1), dtype=int), cur_poi)
            w2 = np.concatenate([w1, w2], axis=1)
        if next_poi is not None:
            w3 = np.kron(next_poi, np.ones((cur_rank * cur_n_points, 1), dtype=int))
            w2 = np.concatenate([w2, w3], axis=1)
        return w2

    @staticmethod
    def f_eval(bounds: np.ndarray, ps: np.ndarray, qs: np.ndarray, n_points: np.ndarray, n_q_points: np.ndarray, todo_q_pois: np.ndarray, min_params: np.ndarray, min_f: float, runner) -> Tuple[np.ndarray, np.ndarray, float]:
        # convert p,q idxs to n idxs
        n_dims = bounds.shape[0]
        todo_pois = np.zeros((todo_q_pois.shape[0], n_dims), dtype=float)
        start = 0
        for dim_idx in range(n_dims):
            submat = todo_q_pois[:, start:start + qs[dim_idx]]
            todo_pois[:, dim_idx] = np.ravel_multi_index(submat.T, n_q_points[start:start + qs[dim_idx]], order="F")
            start += qs[dim_idx]
        # convert idxs to coordinates
        for dim_idx in range(todo_pois.shape[1]):
            t = todo_pois[:, dim_idx] / (n_points[dim_idx] - 1) * (bounds[dim_idx, 1] - bounds[dim_idx, 0]) + bounds[dim_idx, 0]  # scale before shift to minimize fp error
            todo_pois[:, dim_idx] = t
        f_vals = runner.submit(todo_pois.T)
        best_idxs = np.argmin(f_vals)
        best_params = todo_pois[best_idxs]
        best_f = f_vals[best_idxs]
        if best_f < min_f:
            min_params, min_f = (best_params, best_f)
        return f_vals, min_params, min_f