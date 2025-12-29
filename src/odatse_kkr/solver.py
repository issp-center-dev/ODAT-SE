# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE-KKR -- KKR solver module for ODAT-SE
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
AkaiKKR solver implementation for ODAT-SE.

This module provides a solver class that integrates AkaiKKR with ODAT-SE,
including KKR parameter configuration and automatic retry on convergence failures.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import odatse
from odatse.solver import SolverBase

from .exceptions import ConvergenceError
from .input import (
    KKRInputData,
    apply_kkr_parameters_from_config,
    generate_input_content,
    load_input_file,
    write_input_file,
)
from .retry import RetryConfig, check_convergence, run_with_retry

logger = logging.getLogger(__name__)


class Solver(SolverBase):
    """
    AkaiKKR solver for ODAT-SE.

    This solver integrates AkaiKKR with the ODAT-SE framework, providing:
    - Configuration of KKR parameters from TOML files
    - Automatic retry on convergence failures with different ewidth values
    - Template-based input file generation

    Parameters
    ----------
    info : odatse.Info
        Information object containing solver configuration.

    Attributes
    ----------
    base_input_data : KKRInputData
        Base input data loaded from template file.
    retry_config : Optional[RetryConfig]
        Configuration for retry on convergence failure.
    command_template : str
        Command template for executing AkaiKKR.
    command_timeout : Optional[float]
        Timeout for command execution in seconds.

    Examples
    --------
    >>> import odatse
    >>> info = odatse.Info.from_file("input.toml")
    >>> solver = Solver(info)
    >>> result = solver.evaluate(x=np.array([1.0, 2.0]), args=(0, 0))
    """

    _name: str = "odatse-kkr"

    def __init__(self, info: odatse.Info) -> None:
        """
        Initialize the AkaiKKR solver.

        Parameters
        ----------
        info : odatse.Info
            Information object containing solver configuration.

        Raises
        ------
        FileNotFoundError
            If template input file is not found.
        ValueError
            If required configuration is missing.
        """
        super().__init__(info)
        self._name = "odatse-kkr"

        # Get solver configuration
        solver_config = info.solver
        config = info.config if hasattr(info, "config") else {}

        # Parse configuration
        self._parse_config(solver_config, config)

        # Load base input data and apply KKR parameters from config
        self.base_input_data = load_input_file(self.template_input)
        self.base_input_data = apply_kkr_parameters_from_config(
            self.base_input_data, config
        )

        # Retry configuration for convergence failures
        self.retry_config = RetryConfig.from_config(config)
        if self.retry_config:
            logger.info(
                f"Retry on convergence failure enabled: "
                f"ewidth_list={self.retry_config.ewidth_list}"
            )

        logger.info(f"Initialized {self._name} solver")

    def _parse_config(
        self,
        solver_config: Dict[str, Any],
        config: Dict[str, Any],
    ) -> None:
        """
        Parse solver configuration from info object.

        Parameters
        ----------
        solver_config : Dict[str, Any]
            Solver-specific configuration from [solver] section.
        config : Dict[str, Any]
            Full configuration dictionary.
        """
        # Template input file
        template_path = solver_config.get("template_input", "template.in")
        self.template_input = self.root_dir / Path(template_path)

        if not self.template_input.exists():
            raise FileNotFoundError(
                f"Template input file not found: {self.template_input}"
            )

        # Command template for execution
        self.command_template = solver_config.get(
            "command_template",
            "specx < {input} > {output}",
        )

        # Command timeout
        self.command_timeout = solver_config.get("command_timeout")

        # Input/output file names
        self.input_filename = solver_config.get("input_file", "input.in")
        self.output_filename = solver_config.get("output_file", "output.log")

        # Variable list for template substitution
        self.variable_list = solver_config.get("variable_list", [])

        # Reference data file
        self.reference_file = solver_config.get("reference_file")
        if self.reference_file:
            self.reference_file = self.root_dir / Path(self.reference_file)
            self.reference_data = self._load_reference_data(self.reference_file)
        else:
            self.reference_data = None

        # Additional solver options
        self.remove_work_dir = solver_config.get("remove_work_dir", False)
        self.use_tmpdir = solver_config.get("use_tmpdir", False)
        self.solver_data_dir = solver_config.get("solver_data")

    def _load_reference_data(
        self,
        file_path: Path,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load reference data from file.

        Parameters
        ----------
        file_path : Path
            Path to reference data file.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            Tuple of (x, y) arrays, or None if file not found.
        """
        if not file_path.exists():
            logger.warning(f"Reference file not found: {file_path}")
            return None

        try:
            data = np.loadtxt(file_path, unpack=True)
            return (data[0], data[1])
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            return None

    def evaluate(
        self,
        x: np.ndarray,
        args: Tuple = (),
        nprocs: int = 1,
        nthreads: int = 1,
    ) -> float:
        """
        Evaluate the solver for given parameters.

        This method:
        1. Creates a working directory for the calculation
        2. Generates input file from template with parameter substitution
        3. Runs AkaiKKR with automatic retry on convergence failure
        4. Parses output and calculates R-factor

        Parameters
        ----------
        x : np.ndarray
            Parameter values for this evaluation.
        args : Tuple, optional
            Additional arguments (step, iset). Defaults to ().
        nprocs : int, optional
            Number of processes to use. Defaults to 1.
        nthreads : int, optional
            Number of threads to use. Defaults to 1.

        Returns
        -------
        float
            R-factor or other objective value.
            Returns inf if calculation fails.

        Raises
        ------
        ConvergenceError
            If calculation fails to converge after all retries
            (only if retry_config is set and all attempts fail).
        """
        # Create work directory name
        if args:
            work_dir_name = "Log{:08d}_{:08d}".format(*args[:2])
        else:
            work_dir_name = "Log_default"

        calc_dir = self.proc_dir / work_dir_name
        calc_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy solver data if specified
            if self.solver_data_dir:
                src_dir = self.root_dir / self.solver_data_dir
                if src_dir.exists():
                    shutil.copytree(src_dir, calc_dir, dirs_exist_ok=True)

            # Generate input file
            input_path = calc_dir / self.input_filename
            output_path = calc_dir / self.output_filename

            # Create variable mapping
            variables = dict(zip(self.variable_list, x))

            # Write input file with parameter substitution
            write_input_file(self.base_input_data, input_path, variables)

            # Prepare command
            command = self.command_template.format(
                input=self.input_filename,
                output=self.output_filename,
            )

            # Prepare environment
            env = {
                "OMP_NUM_THREADS": str(nthreads),
            }

            # Define retry callback
            def on_retry(attempt: int, new_ewidth: float) -> None:
                logger.info(f"Retry {attempt}: ewidth={new_ewidth}, record=2nd")

            # Run with retry
            try:
                attempts = run_with_retry(
                    command,
                    work_dir=calc_dir,
                    input_path=input_path,
                    output_path=output_path,
                    env=env,
                    timeout=self.command_timeout,
                    retry_config=self.retry_config,
                    on_retry=on_retry,
                )
                if attempts > 1:
                    logger.info(f"Converged after {attempts} attempt(s)")
            except ConvergenceError as e:
                logger.error(f"Convergence failed: {e}")
                raise

            # Calculate result
            result = self._calculate_result(output_path, x)

        except ConvergenceError:
            raise
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            result = float("inf")

        finally:
            # Clean up if requested
            if self.remove_work_dir and calc_dir.exists():
                shutil.rmtree(calc_dir, ignore_errors=True)

        return result

    def _calculate_result(
        self,
        output_path: Path,
        x: np.ndarray,
    ) -> float:
        """
        Calculate objective value from output.

        Parameters
        ----------
        output_path : Path
            Path to output file.
        x : np.ndarray
            Input parameters.

        Returns
        -------
        float
            Calculated objective value (R-factor).
        """
        # Parse output and calculate R-factor
        calc_data = self._parse_output(output_path)
        if calc_data is None:
            return float("inf")

        if self.reference_data is None:
            # If no reference data, return some extracted value
            logger.warning("No reference data, returning 0.0")
            return 0.0

        # Calculate R-factor
        r_factor = self._calc_r_factor(calc_data, self.reference_data)
        return r_factor

    def _parse_output(
        self,
        output_path: Path,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Parse output file to extract calculated data.

        Parameters
        ----------
        output_path : Path
            Path to output file.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            Tuple of (x, y) arrays, or None if parsing fails.
        """
        if not output_path.exists():
            logger.warning(f"Output file not found: {output_path}")
            return None

        try:
            # This is a placeholder - actual implementation depends on
            # AkaiKKR output format
            data = np.loadtxt(output_path, unpack=True)
            return (data[0], data[1])
        except Exception as e:
            logger.debug(f"Failed to parse output: {e}")
            return None

    def _calc_r_factor(
        self,
        calc: Tuple[np.ndarray, np.ndarray],
        ref: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        """
        Calculate R-factor between calculated and reference data.

        Parameters
        ----------
        calc : Tuple[np.ndarray, np.ndarray]
            Calculated data (x, y).
        ref : Tuple[np.ndarray, np.ndarray]
            Reference data (x, y).

        Returns
        -------
        float
            R-factor value.
        """
        if calc is None or ref is None:
            return float("inf")

        n_points = len(calc[1])
        if n_points == 0:
            return float("inf")

        # Simple mean squared error
        r_factor = (1.0 / n_points) * np.sum(np.square(calc[1] - ref[1]))
        return r_factor

