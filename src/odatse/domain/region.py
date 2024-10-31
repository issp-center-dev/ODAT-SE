# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import List, Dict, Union, Any

from pathlib import Path
import numpy as np

import odatse
from ._domain import DomainBase

class Region(DomainBase):
    """
    A class to represent a region in the domain.

    Attributes
    ----------
    min_list : np.array
        Minimum values for each dimension.
    max_list : np.array
        Maximum values for each dimension.
    unit_list : np.array
        Unit values for each dimension.
    initial_list : np.array
        Initial values for each dimension.
    """

    min_list: np.array
    max_list: np.array
    unit_list: np.array
    initial_list: np.array

    def __init__(self, info: odatse.Info = None, *, param: Dict[str, Any] = None):
        """
        Initialize the Region object.

        Parameters
        ----------
        info : odatse.Info, optional
            Information object containing algorithm parameters.
        param : dict, optional
            Dictionary containing algorithm parameters.
        """
        super().__init__(info)

        if info:
            if "param" in info.algorithm:
                self._setup(info.algorithm["param"])
            else:
                raise ValueError("ERROR: algorithm.param not defined")
        elif param:
            self._setup(param)
        else:
            pass

    def _setup(self, info_param):
        """
        Setup the region with the given parameters.

        Parameters
        ----------
        info_param : dict
            Dictionary containing the parameters for the region.
        """
        if "min_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.min_list is not defined in the input")
        min_list = np.array(info_param["min_list"])

        if "max_list" not in info_param:
            raise ValueError("ERROR: algorithm.param.max_list is not defined in the input")
        max_list = np.array(info_param["max_list"])

        if len(min_list) != len(max_list):
            raise ValueError("ERROR: lengths of min_list and max_list do not match")

        self.dimension = len(min_list)

        unit_list = np.array(info_param.get("unit_list", [1.0] * self.dimension))

        self.min_list = min_list
        self.max_list = max_list
        self.unit_list = unit_list

        initial_list = np.array(info_param.get("initial_list", []))
        if initial_list.ndim == 1:
            initial_list = initial_list.reshape(1, -1)

        if initial_list.size > 0:
            if initial_list.shape[1] != self.dimension:
                raise ValueError("ERROR: dimension of initial_list is incorrect")
            self.num_walkers = initial_list.shape[0]
        else:
            self.num_walkers = 0

        self.initial_list = initial_list

    def initialize(self, rng=np.random, limitation=odatse.util.limitation.Unlimited(), num_walkers: int = 1):
        """
        Initialize the region with random values or predefined initial values.

        Parameters
        ----------
        rng : numpy.random, optional
            Random number generator.
        limitation : odatse.util.limitation, optional
            Limitation object to judge the validity of the values.
        num_walkers : int, optional
            Number of walkers to initialize.
        """
        if num_walkers > self.num_walkers:
            self.num_walkers = num_walkers

        if self.initial_list.size > 0 and self.initial_list.shape[0] >= num_walkers:
            pass
        else:
            self._init_random(rng=rng, limitation=limitation)

    def _init_random(self, rng=np.random, limitation=odatse.util.limitation.Unlimited(), max_count=100):
        """
        Initialize the region with random values within the specified limits.

        Parameters
        ----------
        rng : numpy.random, optional
            Random number generator.
        limitation : odatse.util.limitation, optional
            Limitation object to judge the validity of the values.
        max_count : int, optional
            Maximum number of trials to generate valid values.
        """
        initial_list = np.zeros((self.num_walkers, self.dimension), dtype=float)
        is_ok = np.full(self.num_walkers, False)

        if self.initial_list.size > 0:
            nitem = min(self.num_walkers, self.initial_list.shape[0])
            initial_list[0:nitem] = self.initial_list[0:nitem]
            is_ok[0:nitem] = True

        count = 0
        while (not np.all(is_ok)):
            count += 1
            initial_list[~is_ok] = self.min_list + (self.max_list - self.min_list) * rng.rand(np.count_nonzero(~is_ok), self.dimension)
            is_ok = np.array([limitation.judge(v) for v in initial_list])
            if count >= max_count:
                raise RuntimeError("ERROR: init_random: trial count exceeds {}".format(max_count))
        self.initial_list = initial_list

if __name__ == "__main__":
    reg = Region(param={
        "min_list": [0.0, 0.0, 0.0],
        "max_list": [1.0, 1.0, 1.0],
        "initial_list": [[0.1, 0.2, 0.3],
                         [0.2, 0.3, 0.1],
                         [0.3, 0.1, 0.2],
                         [0.2, 0.1, 0.3],
                         [0.1, 0.3, 0.2],
                         ],
    })

    #lim = odatse.util.limitation.Unlimited()
    lim = odatse.util.limitation.Inequality(a=np.array([[1,0,0],[-1,-1,-1]]),b=np.array([0,1]),is_limitary=True)
    
    reg.initialize(np.random, lim, 8)
    
    print(reg.min_list, reg.max_list, reg.unit_list, reg.initial_list, reg.num_walkers)
