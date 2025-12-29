# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE-KKR -- KKR solver module for ODAT-SE
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
ODAT-SE KKR Solver Module.

This module provides the AkaiKKR solver integration for ODAT-SE,
including parameter configuration from TOML and retry functionality
for convergence failures.

Examples
--------
>>> from odatse_kkr import (
...     ConvergenceError,
...     RetryConfig,
...     apply_kkr_parameters_from_config,
...     check_convergence,
...     load_input_file,
...     run_with_retry,
... )
"""

from .exceptions import ConvergenceError
from .input import (
    apply_kkr_parameters_from_config,
    load_input_file,
)
from .retry import (
    RetryConfig,
    check_convergence,
    run_with_retry,
)

__version__ = "0.1.0"

__all__ = [
    "ConvergenceError",
    "RetryConfig",
    "apply_kkr_parameters_from_config",
    "check_convergence",
    "load_input_file",
    "run_with_retry",
]

