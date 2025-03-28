# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
Utility functions and classes for solver operations.

This module provides utilities for running external solvers, managing working
directories, and handling solver paths.
"""

from typing import Dict, List, Tuple
from pathlib import Path

import os

#-- delay import
# import subprocess
# from tempfile import TemporaryDirectory


def run_by_subprocess(command: List[str]) -> None:
    """
    Run a command using subprocess with output redirection.

    This function executes the given command and redirects both stdout and stderr
    to a file named 'stdout'.

    Parameters
    ----------
    command : List[str]
        Command to run as a list of strings, where the first element is the command
        and subsequent elements are arguments.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the command returns a non-zero exit status.
    IOError
        If the stdout file cannot be opened or written to.

    Examples
    --------
    >>> run_by_subprocess(['ls', '-l'])
    >>> run_by_subprocess(['python', 'script.py', '--arg', 'value'])
    """
    import subprocess
    try:
        with open("stdout", "w") as fi:
            subprocess.run(
                command,
                stdout=fi,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except subprocess.CalledProcessError as err:
        msg = "subprocess failed: {}".format(err)
        print(msg)
        raise RuntimeError(msg) from err


def set_solver_path(solver_name: str, root_dir: Path = ".") -> Path:
    """
    Search for a solver executable and return its full path.

    This function searches for the specified solver executable in the given root
    directory and system PATH. If solver_name includes a directory path, it is
    interpreted relative to root_dir.

    Parameters
    ----------
    solver_name : str
        Name or path of solver executable.
    root_dir : Path, optional
        Root directory for relative paths, by default current directory.

    Returns
    -------
    Path
        Full path to the solver executable.

    Raises
    ------
    RuntimeError
        If the solver executable is not found or is not executable.

    Notes
    -----
    The function uses the PATH environment variable to search for the executable
    if solver_name does not include a directory path.

    Examples
    --------
    >>> set_solver_path('mysolver')
    Path('/usr/local/bin/mysolver')
    >>> set_solver_path('solvers/custom_solver', Path('/opt'))
    Path('/opt/solvers/custom_solver')
    """
    if os.path.dirname(solver_name) != "":
        solver_path = root_dir / Path(solver_name).expanduser()
    else:
        for p in [root_dir] + os.environ["PATH"].split(":"):
            solver_path = os.path.join(p, solver_name)
            if os.access(solver_path, mode=os.X_OK):
                break
    if not os.access(solver_path, mode=os.X_OK):
        raise RuntimeError(f"ERROR: solver ({solver_name}) is not found")
    return solver_path


class Workdir:
    """
    A context manager for handling working directories.

    This class provides functionality to temporarily change the working directory
    and optionally clean up afterwards. It supports both named directories and
    temporary directories, with optional automatic cleanup.

    Attributes
    ----------
    work_dir : str or None
        Path to the working directory.
    remove_work_dir : bool
        Whether to remove the working directory on exit.
    use_tmpdir : bool
        Whether to create and use a temporary directory.
    owd : List[str]
        Stack of original working directories.
    tmpdir : TemporaryDirectory, optional
        Temporary directory object when use_tmpdir is True.

    Examples
    --------
    >>> # Using a specific directory
    >>> with Workdir('my_work_dir'):
    ...     # Do work in my_work_dir
    ...     pass

    >>> # Using a temporary directory that gets cleaned up
    >>> with Workdir(use_tmpdir=True):
    ...     # Do work in temporary directory
    ...     pass

    >>> # Using a directory that gets removed afterwards
    >>> with Workdir('temp_dir', remove=True):
    ...     # Do work in temp_dir
    ...     pass  # Directory will be removed after
    """

    def __init__(self, work_dir=None, *, remove=False, use_tmpdir=False):
        """
        Initialize the Workdir context manager.

        Parameters
        ----------
        work_dir : str or None, optional
            Path to the working directory. If None and use_tmpdir is False,
            no directory change occurs.
        remove : bool, optional
            Whether to remove the work directory on exit, by default False.
        use_tmpdir : bool, optional
            Whether to create and use a temporary directory, by default False.

        Notes
        -----
        The TMPDIR environment variable can be used to specify where temporary
        directories should be created when use_tmpdir is True.
        """
        self.work_dir = work_dir
        self.remove_work_dir = remove
        self.use_tmpdir = use_tmpdir

        if work_dir is None:
            self.remove_work_dir = False

        self.owd = []

    def __enter__(self):
        """
        Enter the context, changing to the working directory.

        Returns
        -------
        Workdir
            The Workdir instance.
        """
        if self.use_tmpdir:
            from tempfile import TemporaryDirectory
            self.tmpdir = TemporaryDirectory()
            self.owd.append(os.getcwd())
            os.chdir(self.tmpdir.name)
        elif self.work_dir is not None:
            os.makedirs(self.work_dir, exist_ok=True)
            self.owd.append(os.getcwd())
            os.chdir(self.work_dir)
        return self

    def __exit__(self, ex_type, ex_value, tb):
        """
        Exit the context, restoring the original directory and cleaning up.

        Parameters
        ----------
        ex_type : type
            The type of the exception that occurred, if any.
        ex_value : Exception
            The exception instance that occurred, if any.
        tb : traceback
            The traceback of the exception that occurred, if any.

        Returns
        -------
        bool
            True if no exception occurred, False otherwise.
        """
        if self.owd:
            owd = self.owd.pop()
            os.chdir(owd)

        if not self.use_tmpdir:
            if self.remove_work_dir:
                import shutil
                def rmtree_error_handler(function, path, excinfo):
                    print(f"WARNING: Failed to remove a working directory, {path}")
                shutil.rmtree(self.work_dir, onerror=rmtree_error_handler)

        assert self.owd == []
        return ex_type is None

