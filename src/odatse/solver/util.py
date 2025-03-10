# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Dict, List, Tuple
from pathlib import Path

import os

#-- delay import
# import subprocess
# from tempfile import TemporaryDirectory


def run_by_subprocess(command: List[str]) -> None:
    """
    Runs a command using subprocess.

    Parameters
    ----------
    command : List[str]
        Command to run.
    """
    import subprocess
    with open("stdout", "w") as fi:
        subprocess.run(
            command,
            stdout=fi,
            stderr=subprocess.STDOUT,
            check=True,
        )

def set_solver_path(solver_name: str, root_dir: Path = ".") -> Path:
    """
    Search for the solver executable and returns the path to the solver.

    Parameters
    ----------
    solver_name : str
        Name or path of solver executable.
    root_dir : Path
        Root directory for relative paths.

    Environment variables
    ---------------------
    PATH
        Command search paths.

    Returns
    -------
    Path
        Full path to the solver executable.
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
    Managing work directory

    enters into the work directory on entry, and leaves from it on exit.
    available in "with" clause.
    """
    def __init__(self, work_dir=None, *, remove=False, use_tmpdir=False):
        """
        Initialize the Workdir class.

        Parameters
        ----------
        work_dir : str
            Name of work directory
        remove : bool
            Flag whether to remove the work directory on exit.
        use_tmpdir : bool
            Flag whether to create and use a temporal directory in /tmp.

        Environment variables
        ---------------------
        TMPDIR
            Directory in which temporal directories are created.
            implicitly used by the tmpfile module.
        """
        self.work_dir = work_dir
        self.remove_work_dir = remove
        self.use_tmpdir = use_tmpdir

        if work_dir is None:
            self.remove_work_dir = False

        self.owd = []

    def __enter__(self):
        if self.use_tmpdir:
            from tempfile import TemporaryDirectory
            self.tmpdir = TemporaryDirectory()  # to keep alive
            self.owd.append(os.getcwd())
            #print("Workdir: enter in tmpdir {}".format(self.tmpdir.name))
            os.chdir(self.tmpdir.name)
        elif self.work_dir is not None:
            os.makedirs(self.work_dir, exist_ok=True)
            self.owd.append(os.getcwd())
            #print("Workdir: enter in workdir {}".format(self.work_dir))
            os.chdir(self.work_dir)
        else:
            print("Workdir: do nothing")
            pass
        return self

    def __exit__(self, ex_type, ex_value, tb):
        if self.owd:
            owd = self.owd.pop()
            #print("Workdir: go back to {}".format(owd))
            os.chdir(owd)

        if not self.use_tmpdir:
            if self.remove_work_dir:
                import shutil
                def rmtree_error_handler(function, path, excinfo):
                    print(f"WARNING: Failed to remove a working directory, {path}")
                #print("Workdir: remove directory: {}".format(self.work_dir))
                shutil.rmtree(self.work_dir, onerror=rmtree_error_handler)

        assert self.owd == []
        return ex_type is None

