# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List, Union
import time

import numpy as np
import scipy
from scipy.optimize import minimize

import odatse
import odatse.domain


class Algorithm(odatse.algorithm.AlgorithmBase):
    """
    Algorithm class for performing minimization using the Nelder-Mead method.
    """

    # inputs
    label_list: np.ndarray
    initial_list: np.ndarray
    min_list: np.ndarray
    max_list: np.ndarray
    unit_list: np.ndarray

    # hyperparameters of Nelder-Mead
    initial_simplex_list: List[List[float]]
    xtol: float
    ftol: float

    # results
    xopt: np.ndarray
    fopt: float
    itera: int
    funcalls: int
    allvecs: List[np.ndarray]

    iter_history: List[List[Union[int,float]]]
    fev_history: List[List[Union[int,float]]]

    def __init__(self, info: odatse.Info,
                 runner: odatse.Runner = None,
                 domain = None,
                 run_mode: str = "initial"
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

        self.domain.initialize(rng=self.rng, limitation=runner.limitation, num_walkers=self.mpisize)
        self.initial_list = self.domain.initial_list[self.mpirank]

        info_minimize = info.algorithm.get("minimize", {})
        self.initial_scale_list = info_minimize.get(
            "initial_scale_list", [0.25] * self.dimension
        )
        self.xtol = info_minimize.get("xatol", 0.0001)
        self.ftol = info_minimize.get("fatol", 0.0001)
        self.maxiter = info_minimize.get("maxiter", 10000)
        self.maxfev = info_minimize.get("maxfev", 100000)

        self._show_parameters()

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

        f0 = run.submit(self.initial_list, (0, 0))
        iter_history.append([*self.initial_list, f0])

        scipy_version = [int(s) for s in scipy.__version__.split('.')]

        if scipy_version[0] >= 1 and scipy_version[1] >= 11:
            def _cb(intermediate_result):
                """
                Callback function for scipy.optimize.minimize.
                """
                x = intermediate_result.x
                fun = intermediate_result.fun
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
            # check if within region -> boundary option in minimize
            # note: 'bounds' option supported in scipy >= 1.7.0
            in_range = np.all((min_list < x_list) & (x_list < max_list))
            if not in_range:
                print("Warning: out of range: {}".format(x_list))
                return float("inf")

            # check if limitation satisfied
            in_limit = self.runner.limitation.judge(x_list)
            if not in_limit:
                print("Warning: variables do not satisfy the constraint formula")
                return float("inf")

            x_list /= unit_list

            step[0] += 1
            args = (step[0], iset)
            y = run.submit(x_list, args)
            if iset == 0:
                fev_history.append([step[0], *x_list, y])
            return y

        time_sta = time.perf_counter()
        optres = minimize(
            _f_calc,
            self.initial_list,
            method="Nelder-Mead",
            args=(0,),
            # bounds=[(a,b) for a,b in zip(min_list, max_list)],
            options={
                "xatol": self.xtol,
                "fatol": self.ftol,
                "return_all": True,
                "disp": True,
                "maxiter": self.maxiter,
                "maxfev": self.maxfev,
                "initial_simplex": self.initial_simplex_list,
            },
            callback=_cb,
        )

        self.xopt = optres.x
        self.fopt = optres.fun
        self.itera = optres.nit
        self.funcalls = optres.nfev
        self.allvecs = optres.allvecs
        time_end = time.perf_counter()
        self.timer["run"]["min_search"] = time_end - time_sta

        self.iter_history = iter_history
        self.fev_history = fev_history

        self._output_results()

        if self.mpisize > 1:
            self.mpicomm.barrier()

    def _prepare(self):
        """
        Prepare the initial simplex for the Nelder-Mead algorithm.
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

        with open("res.txt", "w") as fp:
            fp.write(f"fx = {self.fopt}\n")
            for x, y in zip(label_list, self.xopt):
                fp.write(f"{x} = {y}\n")
            fp.write(f"iterations = {self.itera}\n")
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

        if self.mpisize > 1:
            results = self.mpicomm.allgather(result)
        else:
            results = [result]

        xs = [v["x"] for v in results]
        fxs = [v["fx"] for v in results]
        x0s = [v["x0"] for v in results]

        idx = np.argmin(fxs)

        if self.mpirank == 0:
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
