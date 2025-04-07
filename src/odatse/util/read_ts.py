# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import typing
from typing import TextIO, Union, List, Tuple

import numpy as np

def read_Ts(info: dict, numT: int = None) -> Tuple[bool, np.ndarray]:
    """
    Read temperature or inverse-temperature values from the provided info dictionary.

    Parameters
    ----------
    info : dict
        Dictionary containing temperature or inverse-temperature parameters.
    numT : int, optional
        Number of temperature or inverse-temperature values to generate (default is None).

    Returns
    -------
    as_beta : bool
        True if using inverse-temperature, False if using temperature.
    betas : np.ndarray
        Sequence of inverse-temperature values.

    Raises
    ------
    ValueError
        If numT is not specified, or if both Tmin/Tmax and bmin/bmax are defined, or if neither are defined,
        or if bmin/bmax or Tmin/Tmax values are invalid.
    RuntimeError
        If the mode is unknown (neither set_T nor set_b).
    """
    if numT is None:
        raise ValueError("read_Ts: numT is not specified")

    Tmin = info.get("Tmin", None)
    Tmax = info.get("Tmax", None)
    bmin = info.get("bmin", None)
    bmax = info.get("bmax", None)
    logscale = info.get("Tlogspace", True)

    if "Tinvspace" in info:
        raise ValueError("Tinvspace is deprecated. Use bmax/bmin instead.")

    set_b = (bmin is not None or bmax is not None)
    set_T = (Tmin is not None or Tmax is not None)

    if set_b and set_T:
        raise ValueError("both Tmin/Tmax and bmin/bmax are defined")
    if (not set_b) and (not set_T):
        raise ValueError("neither Tmin/Tmax nor bmin/bmax are defined")

    if set_b:
        if bmin is None or bmax is None:
            raise ValueError("bmin and bmax must be set")

        input_as_beta = True
        if not np.isreal(bmin) or bmin < 0.0:
            raise ValueError("bmin must be zero or a positive real number")
        if not np.isreal(bmax) or bmax < 0.0:
            raise ValueError("bmin must be zero or a positive real number")
        if bmin > bmax:
            raise ValueError("bmin must be smaller than or equal to bmax")

        if logscale:
            if bmin == 0.0:
                raise ValueError("bmin must be greater than 0.0 when Tlogspace is True")
            betas = np.logspace(start=np.log10(bmin), stop=np.log10(bmax), num=numT)
        else:
            betas = np.linspace(start=bmin, stop=bmax, num=numT)

    elif set_T:
        if Tmin is None or Tmax is None:
            raise ValueError("Tmin and Tmax must be set")

        input_as_beta = False
        if not np.isreal(Tmin) or Tmin <= 0.0:
            raise ValueError("Tmin must be a positive real number")
        if not np.isreal(Tmax) or Tmax <= 0.0:
            raise ValueError("Tmax must be a positive real number")
        if Tmin > Tmax:
            raise ValueError("Tmin must be smaller than or equal to Tmax")

        if logscale:
            Ts = np.logspace(start=np.log10(Tmin), stop=np.log10(Tmax), num=numT)
        else:
            Ts = np.linspace(start=Tmin, stop=Tmax, num=numT)

        betas = 1.0 / Ts
    else:
        raise RuntimeError("read_Ts: unknown mode: not set_T nor set_b")

    return input_as_beta, betas
