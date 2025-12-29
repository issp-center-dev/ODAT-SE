# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE-KKR -- KKR solver module for ODAT-SE
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Retry functionality for AkaiKKR convergence failures.

This module provides configuration and execution of automatic retries
when KKR calculations fail to converge.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .exceptions import ConvergenceError

logger = logging.getLogger(__name__)

# Pattern to detect convergence failure in output
CONVERGENCE_FAILURE_PATTERN = re.compile(r"no\s+convergence", re.IGNORECASE)


class RetryConfig:
    """
    Configuration for automatic retry on convergence failure.

    This class holds the configuration for retrying KKR calculations
    when convergence fails, including the list of ewidth values to try.

    Parameters
    ----------
    ewidth_list : List[float]
        List of ewidth values to try in order.
    max_retries : Optional[int]
        Maximum number of retries. Defaults to len(ewidth_list) - 1.
    change_record_on_retry : bool
        Whether to change record from "init" to "2nd" on retry.

    Attributes
    ----------
    ewidth_list : List[float]
        List of ewidth values to try.
    max_retries : int
        Maximum number of retries.
    change_record_on_retry : bool
        Whether to change record on retry.

    Examples
    --------
    >>> config = RetryConfig(
    ...     ewidth_list=[2.0, 2.5, 3.0, 3.5],
    ...     change_record_on_retry=True,
    ... )
    >>> print(config.max_retries)
    3
    """

    def __init__(
        self,
        ewidth_list: List[float],
        max_retries: Optional[int] = None,
        change_record_on_retry: bool = True,
    ) -> None:
        """
        Initialize RetryConfig.

        Parameters
        ----------
        ewidth_list : List[float]
            List of ewidth values to try in order.
        max_retries : Optional[int]
            Maximum number of retries. Defaults to len(ewidth_list) - 1.
        change_record_on_retry : bool
            Whether to change record from "init" to "2nd" on retry.
        """
        if not ewidth_list:
            raise ValueError("ewidth_list must not be empty")

        self.ewidth_list = list(ewidth_list)
        self.max_retries = (
            max_retries if max_retries is not None else len(ewidth_list) - 1
        )
        self.change_record_on_retry = change_record_on_retry

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["RetryConfig"]:
        """
        Create RetryConfig from a configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing [kkr.retry] section.

        Returns
        -------
        Optional[RetryConfig]
            RetryConfig instance if [kkr.retry] section exists with
            ewidth_list, None otherwise.

        Examples
        --------
        >>> config = {
        ...     "kkr": {
        ...         "retry": {
        ...             "ewidth_list": [2.0, 2.5, 3.0],
        ...             "change_record_on_retry": True,
        ...         }
        ...     }
        ... }
        >>> retry_config = RetryConfig.from_config(config)
        """
        kkr_config = config.get("kkr", {})
        retry_config = kkr_config.get("retry", {})

        ewidth_list = retry_config.get("ewidth_list")
        if not ewidth_list:
            return None

        return cls(
            ewidth_list=ewidth_list,
            max_retries=retry_config.get("max_retries"),
            change_record_on_retry=retry_config.get("change_record_on_retry", True),
        )

    def get_ewidth(self, attempt: int) -> Optional[float]:
        """
        Get the ewidth value for a given attempt number.

        Parameters
        ----------
        attempt : int
            Attempt number (0-indexed).

        Returns
        -------
        Optional[float]
            ewidth value for this attempt, or None if no more values.
        """
        if attempt < len(self.ewidth_list):
            return self.ewidth_list[attempt]
        return None

    def __repr__(self) -> str:
        """Return string representation of RetryConfig."""
        return (
            f"RetryConfig(ewidth_list={self.ewidth_list}, "
            f"max_retries={self.max_retries}, "
            f"change_record_on_retry={self.change_record_on_retry})"
        )


def check_convergence(output_path: Union[str, Path]) -> bool:
    """
    Check if the KKR calculation converged by examining output.

    Parameters
    ----------
    output_path : Union[str, Path]
        Path to the output file to check.

    Returns
    -------
    bool
        True if calculation converged, False if "no convergence" found.

    Examples
    --------
    >>> converged = check_convergence("output.log")
    >>> if not converged:
    ...     print("Calculation did not converge")
    """
    output_path = Path(output_path)

    if not output_path.exists():
        logger.warning(f"Output file not found: {output_path}")
        return False

    try:
        with open(output_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read output file {output_path}: {e}")
        return False

    # Check for "no convergence" pattern
    if CONVERGENCE_FAILURE_PATTERN.search(content):
        logger.debug(f"Convergence failure detected in {output_path}")
        return False

    logger.debug(f"Calculation converged: {output_path}")
    return True


def _update_input_for_retry(
    input_path: Union[str, Path],
    new_ewidth: float,
    change_record: bool = True,
) -> None:
    """
    Update input file for retry attempt.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to the input file to modify.
    new_ewidth : float
        New ewidth value to set.
    change_record : bool
        Whether to change record from "init" to "2nd".
    """
    input_path = Path(input_path)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise IOError(f"Failed to read input file {input_path}: {e}") from e

    # Update ewidth value
    content = re.sub(
        r"(ewidth\s*=\s*)[\d.]+",
        rf"\g<1>{new_ewidth}",
        content,
        flags=re.IGNORECASE,
    )

    # Update record from "init" to "2nd" if requested
    if change_record:
        content = re.sub(
            r'(record\s*=\s*)["\']?init["\']?',
            r'\g<1>2nd',
            content,
            flags=re.IGNORECASE,
        )

    try:
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Failed to write input file {input_path}: {e}") from e

    logger.debug(f"Updated input file {input_path}: ewidth={new_ewidth}")


def run_with_retry(
    command_template: Union[str, List[str]],
    *,
    work_dir: Optional[Union[str, Path]] = None,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    retry_config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, float], None]] = None,
) -> int:
    """
    Run AkaiKKR command with automatic retry on convergence failure.

    This function executes the KKR solver and automatically retries
    with different ewidth values if convergence fails.

    Parameters
    ----------
    command_template : Union[str, List[str]]
        Command to execute. Can be a string or list of arguments.
    work_dir : Optional[Union[str, Path]]
        Working directory for command execution.
    input_path : Union[str, Path]
        Path to the input file (for retry modification).
    output_path : Union[str, Path]
        Path to the output file (for convergence checking).
    env : Optional[Dict[str, str]]
        Environment variables for the subprocess.
    timeout : Optional[float]
        Timeout in seconds for each execution attempt.
    retry_config : Optional[RetryConfig]
        Retry configuration. If None, no retries are performed.
    on_retry : Optional[Callable[[int, float], None]]
        Callback function called before each retry with (attempt, new_ewidth).

    Returns
    -------
    int
        Number of attempts required for convergence (1 = first attempt succeeded).

    Raises
    ------
    ConvergenceError
        If calculation fails to converge after all retry attempts.
    subprocess.TimeoutExpired
        If command times out.
    subprocess.CalledProcessError
        If command returns non-zero exit code.

    Examples
    --------
    >>> retry_config = RetryConfig(ewidth_list=[2.0, 2.5, 3.0])
    >>> def on_retry(attempt, new_ewidth):
    ...     print(f"Retry {attempt}: ewidth={new_ewidth}")
    >>> attempts = run_with_retry(
    ...     "specx < input.in > output.log",
    ...     work_dir="calc",
    ...     input_path="calc/input.in",
    ...     output_path="calc/output.log",
    ...     retry_config=retry_config,
    ...     on_retry=on_retry,
    ... )
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    work_dir = Path(work_dir) if work_dir else None

    # Prepare command
    if isinstance(command_template, str):
        shell = True
        command = command_template
    else:
        shell = False
        command = command_template

    # Prepare environment
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    # Determine max attempts
    if retry_config:
        max_attempts = len(retry_config.ewidth_list)
    else:
        max_attempts = 1

    ewidth_tried: List[float] = []

    for attempt in range(max_attempts):
        # Update input file for retry (except first attempt)
        if attempt > 0 and retry_config:
            new_ewidth = retry_config.get_ewidth(attempt)
            if new_ewidth is None:
                break

            ewidth_tried.append(new_ewidth)

            if on_retry:
                on_retry(attempt, new_ewidth)

            _update_input_for_retry(
                input_path,
                new_ewidth,
                change_record=retry_config.change_record_on_retry,
            )

        # Execute command
        logger.debug(f"Executing command (attempt {attempt + 1}): {command}")

        try:
            result = subprocess.run(
                command,
                shell=shell,
                cwd=work_dir,
                env=full_env,
                timeout=timeout,
                capture_output=True,
                text=True,
            )

            # Write stdout to output file if using shell command
            if shell and result.stdout:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n--- STDERR ---\n")
                        f.write(result.stderr)

            # Check for non-zero exit (but don't fail yet - convergence check matters)
            if result.returncode != 0:
                logger.warning(
                    f"Command returned non-zero exit code: {result.returncode}"
                )

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            raise

        # Check convergence
        if check_convergence(output_path):
            logger.debug(f"Converged on attempt {attempt + 1}")
            return attempt + 1

        # First attempt failed - track the ewidth if available
        if attempt == 0 and retry_config:
            initial_ewidth = retry_config.get_ewidth(0)
            if initial_ewidth is not None:
                ewidth_tried.append(initial_ewidth)

        # Log retry or failure
        if attempt < max_attempts - 1:
            logger.info(f"Convergence failed on attempt {attempt + 1}, will retry")
        else:
            logger.error(f"Convergence failed after {max_attempts} attempts")

    # All attempts failed
    raise ConvergenceError(
        message="Calculation did not converge after all retry attempts",
        ewidth_tried=ewidth_tried,
        attempts=max_attempts,
    )

