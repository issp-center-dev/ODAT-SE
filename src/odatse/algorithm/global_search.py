# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, Optional, TYPE_CHECKING
import inspect
import time

import numpy as np
from scipy.optimize import differential_evolution, shgo, dual_annealing

try:
    from scipy.optimize import direct
except ImportError:  # scipy < 1.9
    direct = None

import odatse
import odatse.domain

if TYPE_CHECKING:
    from mpi4py import MPI


class Algorithm(odatse.algorithm.AlgorithmBase):
    """
    Algorithm class for global optimization using scipy.optimize routines.

    The optimization method is selected by the ``method`` parameter in the
    ``[algorithm.global_search]`` section. Currently implemented:

    * "DE" / "differential_evolution": scipy.optimize.differential_evolution
    * "shgo": scipy.optimize.shgo
    * "direct": scipy.optimize.direct
    * "dual_annealing": scipy.optimize.dual_annealing

    All other entries of the section are passed verbatim as arguments of the
    selected scipy routine; argument names the routine does not accept abort
    before the optimization starts.

    MPI parallelization uses a master-worker layout over the algorithm
    communicator: algorithm rank 0 drives the scipy optimizer, whose
    ``workers`` hook scatters candidate points (for DE, a whole generation at
    a time) to all algorithm ranks; the other ranks run an evaluation-server
    loop, evaluating their share of the points with their own solver group.
    This composes with solver-side parallelism (``nsolve``): the total
    parallelism is algsize (points) x nsolve (per point). The direct and
    dual_annealing methods do not support parallel evaluation and run
    entirely on rank 0.
    """

    # method name aliases (case-insensitive) -> scipy routine name
    _METHOD_ALIASES = {
        "de": "differential_evolution",
        "differential_evolution": "differential_evolution",
        "shgo": "shgo",
        "direct": "direct",
        "dual_annealing": "dual_annealing",
    }

    # methods implemented so far
    _IMPLEMENTED = {"differential_evolution", "shgo", "direct", "dual_annealing"}

    # arguments of the scipy routines managed by ODAT-SE itself; rejected if
    # the user sets them in [algorithm.global_search]
    _RESERVED = {
        "func", "bounds", "args", "workers", "seed", "rng", "callback",
        "constraints", "vectorized",
    }

    # inputs
    label_list: np.ndarray
    min_list: np.ndarray
    max_list: np.ndarray
    unit_list: np.ndarray

    # optimization method and its parameters
    method: str
    opt_params: dict

    # results
    xopt: np.ndarray
    fopt: float
    itera: Optional[int]
    funcalls: Optional[int]
    success: bool
    # all local minima found (shgo only)
    xl: Optional[np.ndarray]
    funl: Optional[np.ndarray]

    iter_history: list[list[Union[int, float]]]
    fev_history: list[list[Union[int, float]]]

    def __init__(
        self,
        info: odatse.Info,
        runner: odatse.Runner = None,
        domain=None,
        run_mode: str = "initial",
    ) -> None:
        """
        Initialize the Algorithm class.

        Parameters
        ----------
        info : Info
            Information object containing algorithm settings.
        runner : Runner
            Runner object for submitting jobs.
        domain :
            Domain object defining the search space.
        run_mode : str
            Mode of running the algorithm.
        """
        super().__init__(info=info, runner=runner, run_mode=run_mode)

        if domain and isinstance(domain, odatse.domain.Region):
            self.domain = domain
        else:
            self.domain = odatse.domain.Region(info)

        self.min_list = self.domain.min_list
        self.max_list = self.domain.max_list
        self.unit_list = self.domain.unit_list

        info_gs = info.algorithm.get("global_search", {})

        method = str(info_gs.get("method", "DE"))
        key = method.lower()
        if key not in self._METHOD_ALIASES:
            raise ValueError(
                f"algorithm.global_search.method '{method}' is unknown; "
                f"available: DE (differential_evolution), shgo, direct, "
                f"dual_annealing"
            )
        self.method = self._METHOD_ALIASES[key]
        if self.method not in self._IMPLEMENTED:
            raise NotImplementedError(
                f"algorithm.global_search.method '{method}' is not implemented yet; "
                f"currently implemented: DE (differential_evolution), shgo, direct, "
                f"dual_annealing"
            )
        if self.method == "direct" and direct is None:
            raise RuntimeError(
                "algorithm.global_search.method 'direct' requires scipy >= 1.9"
            )

        # forward all remaining entries verbatim as arguments of the scipy
        # routine
        self.opt_params = {k: v for k, v in info_gs.items() if k != "method"}
        reserved = self._RESERVED & set(self.opt_params)
        if reserved:
            raise ValueError(
                "algorithm.global_search parameters {} are managed by ODAT-SE "
                "and cannot be set in the input file".format(sorted(reserved))
            )
        # validate the argument names against the signature of the installed
        # scipy before anything runs, instead of catching TypeError around
        # the optimizer call: a TypeError raised at runtime (by the solver,
        # a callback, ...) must not be misreported as an input-file mistake
        scipy_func = {
            "differential_evolution": differential_evolution,
            "shgo": shgo,
            "direct": direct,
            "dual_annealing": dual_annealing,
        }[self.method]
        accepted = set(inspect.signature(scipy_func).parameters)
        unknown = set(self.opt_params) - accepted
        if unknown:
            raise ValueError(
                "algorithm.global_search parameters {} are not accepted by "
                "scipy.optimize.{} of the installed scipy version; accepted "
                "arguments are {}".format(
                    sorted(unknown), self.method,
                    sorted(accepted - {"func", "bounds"} - self._RESERVED))
            )

        self._show_parameters()

    def _initialize(self) -> None:
        """Set up initial state for a fresh run.

        The global search does not use checkpointing, so this is a no-op.
        """
        pass

    def _prepare(self) -> None:
        pass

    def _run(self) -> None:
        """
        Run the global optimization.

        Algorithm rank 0 drives the scipy optimizer; the other algorithm
        ranks serve function evaluations until rank 0 signals completion.
        """
        run = self.runner

        min_list = self.min_list
        max_list = self.max_list
        unit_list = self.unit_list

        comm = odatse.mpi.algcomm()
        nprocs = odatse.mpi.algsize()
        rank = odatse.mpi.algrank()

        step = [0]
        fev_history = []
        iter_history = []
        # best-so-far values recorded by the workers hook, so that the
        # per-generation callback can report f without re-evaluating
        f_cache = {}

        def _f_calc(x_list: np.ndarray) -> float:
            """
            Calculate the objective function value at one point.
            """
            # check if within region; scipy keeps candidates inside bounds,
            # so this is a safety net (boundary points are legitimate)
            in_range = np.all((min_list <= x_list) & (x_list <= max_list))
            if not in_range:
                print("Warning: out of range: {}".format(x_list))
                return float("inf")

            # check if limitation satisfied
            in_limit = self.runner.limitation.judge(x_list)
            if not in_limit:
                print("Warning: variables do not satisfy the constraint formula")
                return float("inf")

            # scale into solver units on a copy; x_list is owned by scipy
            x_scaled = x_list / unit_list

            step[0] += 1
            y = run.submit(x_scaled, (step[0], 0))
            fev_history.append([step[0], *x_scaled, y])
            # cache rank-local evaluations too (e.g. local refinements that
            # bypass the workers hook), so the iteration callback can report f
            f_cache[np.asarray(x_list, dtype=float).tobytes()] = y
            return y

        def _evaluate_chunk(xs: np.ndarray):
            """Evaluate this rank's share of the points.

            Exceptions are captured and returned instead of raised, so that
            the collective communication pattern stays balanced across ranks;
            the failed points evaluate to inf.
            """
            idx = np.array_split(np.arange(len(xs)), nprocs)[rank]
            vals = []
            error = None
            for i in idx:
                try:
                    v = _f_calc(np.asarray(xs[i], dtype=float))
                except Exception as e:
                    if error is None:
                        error = e
                    v = float("inf")
                vals.append(v)
            return vals, error

        def _evaluate_points(xs: np.ndarray) -> list:
            """(rank 0) Evaluate a set of points using all algorithm ranks."""
            if nprocs > 1:
                comm.bcast((odatse.mpi.MSG_EVALUATE, xs), root=0)
            vals, error = _evaluate_chunk(xs)
            if nprocs > 1:
                gathered = comm.gather((vals, error), root=0)
                vals = [v for chunk, _ in gathered for v in chunk]
                errors = [e for _, e in gathered if e is not None]
                if errors:
                    raise errors[0]
            else:
                if error is not None:
                    raise error
            return vals

        def _workers(func, iterable):
            """Map-like hook passed to the scipy routine as workers=.

            The func argument (scipy's wrapped objective) is ignored: every
            algorithm rank evaluates with its own identical _f_calc, so the
            objective never needs to be shipped over MPI.
            """
            points = list(iterable)
            if len(points) == 0:
                # e.g. shgo maps over an evaluation pool that can be empty
                return []
            xs = np.atleast_2d(np.asarray(points, dtype=float))
            vals = _evaluate_points(xs)
            for x, v in zip(xs, vals):
                f_cache[x.tobytes()] = v
            return vals

        def _serve_evaluations() -> bool:
            """(rank > 0) Evaluate chunks of points until rank 0 signals
            completion (MSG_FINISHED) or failure (MSG_ABORT).

            Returns True on normal completion. Local evaluation errors are
            reported to rank 0 through the gather (which makes rank 0 abort
            the optimization) and re-raised here after the loop ends, so
            that the collective pattern stays balanced across ranks.
            """
            captured = None
            while True:
                msg, xs = comm.bcast(None, root=0)
                if msg != odatse.mpi.MSG_EVALUATE:
                    break
                vals, error = _evaluate_chunk(xs)
                if error is not None and captured is None:
                    captured = error
                comm.gather((vals, error), root=0)
            if captured is not None:
                raise captured
            return msg == odatse.mpi.MSG_FINISHED

        def _cb(xk, convergence=None):
            """
            Per-iteration callback for the scipy routines.

            differential_evolution calls it per generation as
            (xk, convergence); shgo and direct call it per iteration as
            (xk). The old-style signatures are used because they are
            supported by every scipy version in the supported range.
            """
            fun = f_cache.get(np.asarray(xk, dtype=float).tobytes(), float("nan"))
            row = [len(iter_history), *xk, fun]
            if convergence is not None:
                row.append(float(convergence))
            print("iteration {}: best x={}, fun={}".format(len(iter_history), xk, fun))
            iter_history.append(row)

        def _cb_da(x, f, context):
            """
            Callback for dual_annealing, invoked each time a new best
            minimum is found, as (x, f, context) with context 0 (found
            during annealing), 1 (found during local search) or 2 (found
            in the dual annealing process). f comes with the callback, so
            no f_cache lookup is needed.
            """
            row = [len(iter_history), *x, float(f), int(context)]
            print("minimum {}: x={}, fun={}, context={}".format(
                len(iter_history), x, f, context))
            iter_history.append(row)

        params = dict(self.opt_params)
        if self.method == "differential_evolution":
            # deferred updating evaluates a whole generation at a time, which
            # the parallel evaluation requires; it is also scipy's own
            # fallback when workers is set, so make it the default to keep
            # serial and parallel runs identical (a user-specified value
            # still takes precedence)
            params.setdefault("updating", "deferred")

        bounds = list(zip(min_list, max_list))

        # inject the MPI map only when there are ranks to distribute to:
        # passing workers= unconditionally would make even serial runs
        # require a scipy version that supports the keyword (shgo gained it
        # in 1.11). Serial DE results stay identical either way because
        # updating='deferred' evaluates the population in the same order as
        # the workers hook does.
        workers_kwargs = {"workers": _workers} if nprocs > 1 else {}

        time_sta = time.perf_counter()
        if rank == 0:
            # argument names were validated against the scipy signature in
            # __init__, so a TypeError here is a genuine runtime failure and
            # propagates unchanged (issue #76)
            try:
                if self.method == "differential_evolution":
                    optres = differential_evolution(
                        _f_calc,
                        bounds,
                        # self.rng is a RandomState; the seed path accepts
                        # it across all supported scipy versions, while
                        # the new rng= argument of scipy >= 1.15 does not
                        seed=self.rng,
                        callback=_cb,
                        **workers_kwargs,
                        **params,
                    )
                elif self.method == "shgo":
                    # shgo is deterministic and takes no seed; workers
                    # parallelizes the sampling-phase evaluations, while
                    # the local refinements run serially through _f_calc
                    optres = shgo(
                        _f_calc,
                        bounds,
                        callback=_cb,
                        **workers_kwargs,
                        **params,
                    )
                elif self.method == "direct":
                    # direct is deterministic and does not support
                    # parallel evaluation; it runs entirely on rank 0
                    # while the other ranks stay idle in the server loop
                    if nprocs > 1:
                        print("Warning: method 'direct' does not support "
                              "parallel evaluation; algorithm ranks > 0 "
                              "stay idle")
                    optres = direct(
                        _f_calc,
                        bounds,
                        callback=_cb,
                        **params,
                    )
                elif self.method == "dual_annealing":
                    # dual_annealing runs a single sequential annealing
                    # chain and does not support parallel evaluation; it
                    # runs entirely on rank 0 while the other ranks stay
                    # idle in the server loop
                    if nprocs > 1:
                        print("Warning: method 'dual_annealing' does not "
                              "support parallel evaluation; algorithm ranks "
                              "> 0 stay idle")
                    optres = dual_annealing(
                        _f_calc,
                        bounds,
                        # as for differential_evolution: the seed path
                        # accepts the RandomState across all supported
                        # scipy versions, while the new rng= argument of
                        # scipy >= 1.15 does not
                        seed=self.rng,
                        callback=_cb_da,
                        **params,
                    )
                else:  # pragma: no cover - guarded in __init__
                    raise RuntimeError(f"method {self.method} not implemented")
            except BaseException:
                # release the evaluation servers before propagating, so that
                # every rank reaches the consensus collective in run()
                if nprocs > 1:
                    comm.bcast((odatse.mpi.MSG_ABORT, None), root=0)
                raise
            if nprocs > 1:
                comm.bcast((odatse.mpi.MSG_FINISHED, None), root=0)
            result = (
                np.asarray(optres.x),
                float(optres.fun),
                getattr(optres, "nit", None),
                getattr(optres, "nfev", None),
                bool(optres.success),
                # shgo also reports all local minima found
                getattr(optres, "xl", None),
                getattr(optres, "funl", None),
            )
        else:
            finished = _serve_evaluations()
            if not finished:
                # rank 0 aborted before broadcasting the result; skip the
                # result broadcast (rank 0 is not participating in it) and
                # let the consensus in run() report the failure
                raise odatse.mpi.OtherAlgorithmProcessError()
            result = None

        if nprocs > 1:
            result = comm.bcast(result, root=0)
        (self.xopt, self.fopt, self.itera, self.funcalls, self.success,
         self.xl, self.funl) = result

        time_end = time.perf_counter()
        self.timer["run"]["global_search"] = time_end - time_sta

        self.iter_history = iter_history
        self.fev_history = fev_history

        self._output_results()

    def _output_results(self):
        """
        Output the results of the optimization to files.

        Every algorithm rank writes the history of its own function
        evaluations; the iteration history and the result summary exist only
        on rank 0, which drove the optimizer.
        """
        label_list = self.label_list

        with open("History_FunctionCall.txt", "w") as fp:
            fp.write("#No " + " ".join(label_list) + "\n")
            for v in self.fev_history:
                fp.write(" ".join(map(str, v)) + "\n")

        if odatse.mpi.algrank() == 0:
            if self.method == "differential_evolution":
                iter_file, iter_header = "GenerationData.txt", "#gen {} R-factor convergence\n"
            elif self.method == "dual_annealing":
                # rows are recorded when a new best minimum is found, not
                # per iteration
                iter_file, iter_header = "MinimumData.txt", "#no {} R-factor context\n"
            else:
                iter_file, iter_header = "IterationData.txt", "#iter {} R-factor\n"
            with open(iter_file, "w") as fp:
                fp.write(iter_header.format(" ".join(label_list)))
                for v in self.iter_history:
                    fp.write(" ".join(map(str, v)) + "\n")

            if self.xl is not None and self.funl is not None:
                with open("LocalMinimaData.txt", "w") as fp:
                    fp.write("#no " + " ".join(label_list) + " R-factor\n")
                    for i, (x, f) in enumerate(zip(self.xl, self.funl)):
                        fp.write(str(i) + " " + " ".join(map(str, x)) + " " + str(f) + "\n")

            with open("res.txt", "w") as fp:
                fp.write(f"fx = {self.fopt}\n")
                for x, y in zip(label_list, self.xopt):
                    fp.write(f"{x} = {y}\n")
                if self.itera is not None:
                    fp.write(f"iterations = {self.itera}\n")
                if self.funcalls is not None:
                    fp.write(f"function_evaluations = {self.funcalls}\n")

    def _post(self):
        """
        Post-process the results after optimization.
        """
        if odatse.mpi.algrank() == 0:
            label_list = self.label_list
            with open("res.txt", "w") as fp:
                fp.write(f"fx = {self.fopt}\n")
                for x, y in zip(label_list, self.xopt):
                    fp.write(f"{x} = {y}\n")

        return {"x": self.xopt, "fx": self.fopt}
