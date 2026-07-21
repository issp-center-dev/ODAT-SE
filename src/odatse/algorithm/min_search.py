# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, Optional, TYPE_CHECKING
import time
import warnings

import numpy as np
import scipy
from scipy.optimize import minimize, basinhopping, OptimizeResult, OptimizeWarning

import odatse
import odatse.domain
from odatse.util.version import parse_version

if TYPE_CHECKING:
    from mpi4py import MPI


class _ClippedRandomDisplacement:
    """Random displacement for basinhopping, clipped to the search region.

    The default take_step of scipy's basinhopping may propose points outside
    [min_list, max_list], which would only waste solver evaluations on the
    inf-penalty. Clipping keeps every hop inside the region. The ``stepsize``
    attribute is exposed so that basinhopping's adaptive stepsize adjustment
    keeps working.
    """

    def __init__(self, rng, stepsize, min_list, max_list):
        self.rng = rng
        self.stepsize = stepsize
        self.min_list = min_list
        self.max_list = max_list

    def __call__(self, x):
        x = x + self.rng.uniform(-self.stepsize, self.stepsize, np.shape(x))
        return np.clip(x, self.min_list, self.max_list)


class Algorithm(odatse.algorithm.AlgorithmBase):
    """
    Algorithm class for performing minimization using scipy.optimize.minimize.

    The optimization method is selected by the ``method`` parameter in the
    ``[algorithm.minimize]`` section (default: "Nelder-Mead"). All other
    entries of the section except ODAT-SE-specific keys are passed through
    to scipy.optimize.minimize as its ``options`` argument.

    Setting ``basinhopping`` (a boolean, or a ``[algorithm.minimize.basinhopping]``
    table whose entries are passed to scipy.optimize.basinhopping) switches to
    global optimization by basin hopping, with the configured method serving
    as the local minimizer.
    """

    # methods for which ODAT-SE passes bounds= to scipy.optimize.minimize.
    # Nelder-Mead is deliberately excluded to keep the legacy behavior of
    # returning +inf for out-of-range points unchanged.
    _BOUNDS_METHODS = {"powell", "l-bfgs-b", "tnc", "slsqp", "trust-constr", "cobyla", "cobyqa"}

    # keys of [algorithm.minimize] consumed by ODAT-SE itself, i.e. not
    # forwarded to scipy.optimize.minimize as options
    _ODATSE_KEYS = {"method", "initial_scale_list", "basinhopping"}

    # basinhopping arguments managed by ODAT-SE itself; rejected if the user
    # sets them in [algorithm.minimize.basinhopping]
    _BH_RESERVED = {"minimizer_kwargs", "take_step", "accept_test", "callback", "seed", "rng"}

    # inputs
    label_list: np.ndarray
    initial_list: np.ndarray
    min_list: np.ndarray
    max_list: np.ndarray
    unit_list: np.ndarray

    # optimization method and its options
    method: str
    minimize_options: dict
    # None: plain minimize; dict (possibly empty): basinhopping parameters
    basinhopping_params: Optional[dict]

    # hyperparameters of Nelder-Mead
    initial_simplex_list: list[list[float]]

    # results
    xopt: np.ndarray
    fopt: float
    itera: Optional[int]
    funcalls: Optional[int]
    allvecs: Optional[list[np.ndarray]]

    iter_history: list[list[Union[int, float]]]
    fev_history: list[list[Union[int, float]]]
    hop_history: list[list[Union[int, float]]]

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

        if odatse.mpi.run_on_algorithm():
            self.domain.initialize(rng=self.rng, limitation=runner.limitation, num_walkers=odatse.mpi.algsize())
            self.initial_list = self.domain.initial_list[odatse.mpi.algrank()]
        else:
            self.initial_list = []

        info_minimize = info.algorithm.get("minimize", {})
        self.method = str(info_minimize.get("method", "Nelder-Mead"))
        self.initial_scale_list = info_minimize.get(
            "initial_scale_list", [0.25] * self.dimension
        )

        # basinhopping = true enables scipy.optimize.basinhopping with its
        # default parameters; a [algorithm.minimize.basinhopping] table both
        # enables it and forwards its entries as basinhopping arguments
        bh = info_minimize.get("basinhopping", False)
        if bh is False or bh is None:
            self.basinhopping_params = None
        elif bh is True:
            self.basinhopping_params = {}
        elif isinstance(bh, dict):
            self.basinhopping_params = dict(bh)
        else:
            raise ValueError(
                "algorithm.minimize.basinhopping must be a boolean or a table, "
                f"not {type(bh).__name__}"
            )
        if self.basinhopping_params is not None:
            reserved = self._BH_RESERVED & set(self.basinhopping_params)
            if reserved:
                raise ValueError(
                    "algorithm.minimize.basinhopping parameters {} are managed "
                    "by ODAT-SE and cannot be set in the input file".format(sorted(reserved))
                )

        # forward all remaining entries verbatim to scipy.optimize.minimize
        # as its options argument; unknown option names are detected by scipy
        # and turned into an error in _run() before the optimization starts
        self.minimize_options = {
            k: v for k, v in info_minimize.items() if k not in self._ODATSE_KEYS
        }

        self._show_parameters()

    def _initialize(self) -> None:
        """Set up initial state for a fresh run.

        Nelder-Mead does not use checkpointing, so this is a no-op.
        The simplex initialisation is done in ``_prepare()``.
        """
        pass

    def _run(self) -> None:
        """
        Run the minimization algorithm.
        """
        run = self.runner

        min_list = self.min_list
        max_list = self.max_list
        unit_list = self.unit_list
        label_list = self.label_list

        step = [0]
        iter_history = []
        fev_history = []

        # evaluate the initial point in solver units, as _f_calc does
        f0 = run.submit(np.asarray(self.initial_list) / unit_list, (0, 0))
        iter_history.append([*self.initial_list, f0])

        if parse_version(scipy.__version__) >= (1, 11, 0):
            def _cb(intermediate_result):
                """
                Callback function for scipy.optimize.minimize.

                The parameter must be named intermediate_result so that scipy
                passes an OptimizeResult where supported. Methods that do not
                support the new-style callback (e.g. COBYLA, SLSQP, TNC) still
                pass the raw parameter vector, so handle both.
                """
                if isinstance(intermediate_result, OptimizeResult):
                    x = intermediate_result.x
                    fun = intermediate_result.fun
                else:
                    x = intermediate_result
                    fun = _f_calc(x, 1)
                print("eval: x={}, fun={}".format(x, fun))
                iter_history.append([*x, fun])
        else:
            def _cb(x):
                """
                Callback function for scipy.optimize.minimize.
                """
                fun = _f_calc(x, 1)
                print("eval: x={}, fun={}".format(x, fun))
                iter_history.append([*x, fun])

        # for methods that support it, let scipy keep the search within the
        # region via bounds=. the range check in _f_calc then allows points
        # exactly on the boundary, which such methods evaluate legitimately.
        # On older scipy where the method predates bounds support (e.g.
        # Powell < 1.5, COBYLA < 1.11), scipy itself ignores bounds= with a
        # RuntimeWarning ("Method X cannot handle bounds."); that warning is
        # not escalated by the OptimizeWarning filter below, and the range
        # check in _f_calc remains as the inf-penalty safety net.
        use_bounds = self.method.lower() in self._BOUNDS_METHODS

        def _f_calc(x_list: np.ndarray, iset) -> float:
            """
            Calculate the objective function value.

            Parameters
            ----------
            x_list : np.ndarray
                List of variables.
            iset :
                Set index.

            Returns
            -------
            float
                Objective function value.
            """
            # check if within region; kept as a safety net even when bounds=
            # is passed to minimize
            if use_bounds:
                in_range = np.all((min_list <= x_list) & (x_list <= max_list))
            else:
                in_range = np.all((min_list < x_list) & (x_list < max_list))
            if not in_range:
                print("Warning: out of range: {}".format(x_list))
                return float("inf")

            # check if limitation satisfied
            in_limit = self.runner.limitation.judge(x_list)
            if not in_limit:
                print("Warning: variables do not satisfy the constraint formula")
                return float("inf")

            # Scale into solver units on a *copy*: x_list is the array owned by
            # scipy's optimizer (and, for scipy < 1.11, the same array passed
            # to the callback). Dividing it in place corrupts the optimizer's
            # simplex bookkeeping and, on old scipy, double-scales x in _cb.
            x_scaled = x_list / unit_list

            step[0] += 1
            args = (step[0], iset)
            y = run.submit(x_scaled, args)
            if iset == 0:
                fev_history.append([step[0], *x_scaled, y])
            return y

        use_basinhopping = self.basinhopping_params is not None

        options = dict(self.minimize_options)
        if self.method.lower() == "nelder-mead":
            # keep the historical defaults of the Nelder-Mead implementation;
            # user-specified values in [algorithm.minimize] take precedence
            options.setdefault("xatol", 0.0001)
            options.setdefault("fatol", 0.0001)
            options.setdefault("maxiter", 10000)
            options.setdefault("maxfev", 100000)
            if not use_basinhopping:
                # a fixed initial simplex makes scipy ignore its x0 argument,
                # which would restart every basinhopping hop from the same
                # simplex; only usable for a single local optimization
                options.setdefault("initial_simplex", self.initial_simplex_list)
                options.setdefault("return_all", True)
        if use_basinhopping:
            # per-hop convergence messages of the local minimizer are noisy;
            # progress is reported per hop by basinhopping itself
            options.setdefault("disp", False)
        else:
            options.setdefault("disp", True)

        minimize_kwargs = {}
        if use_bounds:
            minimize_kwargs["bounds"] = list(zip(min_list, max_list))

        hop_history = []

        def _bh_cb(x, f, accept):
            """
            Per-hop callback function for scipy.optimize.basinhopping.
            """
            print("hop: x={}, fun={}, accept={}".format(x, f, accept))
            hop_history.append([len(hop_history), *x, f, int(accept)])

        time_sta = time.perf_counter()
        try:
            with warnings.catch_warnings():
                # scipy only warns on option names the method does not accept
                # and silently ignores them; promote the warning to an error
                # so that e.g. a misspelled tolerance aborts immediately
                # instead of running a lengthy optimization with defaults
                warnings.filterwarnings(
                    "error", message="Unknown solver options", category=OptimizeWarning
                )
                if use_basinhopping:
                    bh_params = dict(self.basinhopping_params)
                    bh_params.setdefault("disp", True)
                    take_step = _ClippedRandomDisplacement(
                        self.rng, bh_params.pop("stepsize", 0.5), min_list, max_list
                    )
                    try:
                        optres = basinhopping(
                            _f_calc,
                            self.initial_list,
                            minimizer_kwargs={
                                "method": self.method,
                                "args": (0,),
                                "options": options,
                                "callback": _cb,
                                **minimize_kwargs,
                            },
                            take_step=take_step,
                            callback=_bh_cb,
                            # self.rng is a RandomState; the deprecated seed
                            # path accepts it on scipy >= 1.15 while rng= does
                            # not, and older scipy has only seed
                            seed=self.rng,
                            **bh_params,
                        )
                    except TypeError as e:
                        raise RuntimeError(
                            f"{e}: check the [algorithm.minimize.basinhopping] "
                            f"section of the input file against the arguments "
                            f"accepted by scipy.optimize.basinhopping"
                        ) from e
                else:
                    optres = minimize(
                        _f_calc,
                        self.initial_list,
                        method=self.method,
                        args=(0,),
                        options=options,
                        callback=_cb,
                        **minimize_kwargs,
                    )
        except OptimizeWarning as w:
            raise RuntimeError(
                f"{w}: check the [algorithm.minimize] section of the input file "
                f"against the options accepted by scipy.optimize.minimize "
                f"for method '{self.method}'"
            ) from w

        self.xopt = optres.x
        self.fopt = optres.fun
        self.itera = getattr(optres, "nit", None)
        self.funcalls = getattr(optres, "nfev", None)
        self.allvecs = getattr(optres, "allvecs", None)
        time_end = time.perf_counter()
        self.timer["run"]["min_search"] = time_end - time_sta

        self.iter_history = iter_history
        self.fev_history = fev_history
        self.hop_history = hop_history

        self._output_results()

        if odatse.mpi.run_on_algorithm():
            if odatse.mpi.algsize() > 1:
                odatse.mpi.algcomm().barrier()

    def _prepare(self):
        """
        Prepare the initial simplex for the Nelder-Mead algorithm.

        The simplex is only passed to scipy when method is Nelder-Mead;
        for other methods it is built but unused.
        """
        # make initial simplex
        #   [ v0, v0+a_1*e_1, v0+a_2*e_2, ... v0+a_d*e_d ]
        # where a = ( a_1 a_2 a_3 ... a_d ) and e_k is a unit vector along k-axis
        v = np.array(self.initial_list)
        a = np.array(self.initial_scale_list)
        self.initial_simplex_list = np.vstack((v, v + np.diag(a)))

    def _output_results(self):
        """
        Output the results of the minimization to files.
        """
        label_list = self.label_list

        with open("SimplexData.txt", "w") as fp:
            fp.write("#step " + " ".join(label_list) + " R-factor\n")
            for i, v in enumerate(self.iter_history):
                fp.write(str(i) + " " + " ".join(map(str,v)) + "\n")

        with open("History_FunctionCall.txt", "w") as fp:
            fp.write("#No " + " ".join(label_list) + "\n")
            for i, v in enumerate(self.fev_history):
                fp.write(" ".join(map(str,v)) + "\n")

        if self.hop_history:
            with open("BasinHoppingData.txt", "w") as fp:
                fp.write("#hop " + " ".join(label_list) + " R-factor accept\n")
                for v in self.hop_history:
                    fp.write(" ".join(map(str, v)) + "\n")

        with open("res.txt", "w") as fp:
            fp.write(f"fx = {self.fopt}\n")
            for x, y in zip(label_list, self.xopt):
                fp.write(f"{x} = {y}\n")
            # some methods (e.g. COBYLA) do not report these quantities
            if self.itera is not None:
                fp.write(f"iterations = {self.itera}\n")
            if self.funcalls is not None:
                fp.write(f"function_evaluations = {self.funcalls}\n")

    def _post(self):
        """
        Post-process the results after minimization.
        """
        result = {
            "x": self.xopt,
            "fx": self.fopt,
            "x0": self.initial_list,
        }

        if odatse.mpi.algsize() > 1:
            results = odatse.mpi.algcomm().allgather(result)
        else:
            results = [result]

        xs = [v["x"] for v in results]
        fxs = [v["fx"] for v in results]
        x0s = [v["x0"] for v in results]

        idx = np.argmin(fxs)

        if odatse.mpi.algrank() == 0:
            label_list = self.label_list
            with open("res.txt", "w") as fp:
                fp.write(f"fx = {fxs[idx]}\n")
                for x, y in zip(label_list, xs[idx]):
                    fp.write(f"{x} = {y}\n")
                if len(results) > 1:
                    fp.write(f"index = {idx}\n")
                    for x, y in zip(label_list, x0s[idx]):
                        fp.write(f"initial {x} = {y}\n")

        return {"x": xs[idx], "fx": fxs[idx], "x0": x0s[idx]}
