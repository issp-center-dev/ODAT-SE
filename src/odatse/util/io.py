# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Reading files in a parallel run: one process reads, the others receive.

A solver or algorithm often reads a file: reference data or an input
template in its constructor, a file written by an external program during
``evaluate()``. In a parallel run the natural way, one process reads and
broadcasts, hangs the job when the read fails on that process: it raises
before the broadcast while the others wait in it. The functions here do it
safely. In a serial run (no mpi4py, ``ODATSE_NOMPI=1``, a single process)
they are plain reads.

::

    import odatse.util.io as oio

    class MySolver(odatse.solver.SolverBase):
        def __init__(self, info):
            super().__init__(info)
            self.reference = oio.loadtxt(info.solver["reference_path"])
            self.template = oio.read_text(info.solver["template_path"])

        def evaluate(self, xs, args):
            ...
            result = oio.loadtxt("result.dat", scope="solver")

``scope`` selects who receives the data (and which processes must make the
call, in the same order, since the read and the distribution are a
collective):

- ``"job"`` (default): every process of the job; for data read once, in a
  constructor.
- ``"solver"``: the controller and the workers of one solver group; for a
  file read inside ``evaluate()``.
- ``"algorithm"``: the algorithm ranks only (the framework uses it for the
  mesh and the neighbor list); returns ``None`` on a solver worker.

If the read fails, every process of the scope raises the same
``odatse.exception.LoadError`` (an ``InputError``), the reading process with
the original exception as ``__cause__``. When every process reads its own
copy of a file there is nothing to distribute and these functions are not
needed.
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from odatse import mpi
from . import toml as _toml

__all__ = ["load", "read_text", "read_bytes", "load_toml", "loadtxt", "load_json"]

_SCOPES = ("job", "algorithm", "solver")


def load(loader: Callable[[], Any], *, scope: str = "job", root: int = 0,
         what: Optional[str] = None, distribute: bool = True) -> Any:
    """Call ``loader()`` on the root process of the scope and hand its result
    to every process of the scope.

    Parameters
    ----------
    loader : callable
        Called without arguments on the root process only; returns the data
        (anything that can be pickled).
    scope : {"job", "algorithm", "solver"}
        Who receives the data: every process of the job, the algorithm ranks,
        or the controller and workers of one solver group. ``"algorithm"`` and
        ``"solver"`` need ``odatse.mpi.setup()`` to have been called.
    root : int
        Rank, within the scope's communicator, of the process that reads.
    what : str, optional
        Name of the data for error messages (``"reference data"``, a path).
    distribute : bool
        If False, only the outcome of the read is shared: the data is
        returned on the root and ``None`` elsewhere (for data the root will
        scatter itself).

    Returns
    -------
    The data on every process of the scope (``None`` on a solver worker with
    ``scope="algorithm"``, which is not a member).

    Raises
    ------
    odatse.exception.LoadError
        on every process of the scope, when ``loader()`` raised on the root
        or its result could not be pickled. The root attaches the original
        exception as ``__cause__``.

    Notes
    -----
    Collective over the scope: every process of it must call ``load()`` in
    the same order. In a serial run, or with the non-MPI stub, this is a
    plain ``loader()`` call (wrapped in ``LoadError`` on failure).
    """
    if scope not in _SCOPES:
        raise ValueError(f"scope must be one of {_SCOPES}, got {scope!r}")
    if scope == "algorithm" and not mpi.run_on_algorithm():
        return None   # a solver worker is not a member of the algorithm layer
    comm = {"job": mpi.comm, "algorithm": mpi.algcomm, "solver": mpi.solcomm}[scope]()
    return mpi._distribute(loader, comm, root=root, what=what or "data", distribute=distribute)


def read_text(path, *, encoding: str = "utf-8", scope: str = "job", root: int = 0) -> str:
    """The text of the file at ``path``, read on the root process of the scope."""
    path = Path(path)
    return load(lambda: path.read_text(encoding=encoding), scope=scope, root=root, what=f"file {path}")


def read_bytes(path, *, scope: str = "job", root: int = 0) -> bytes:
    """The bytes of the file at ``path``, read on the root process of the scope."""
    path = Path(path)
    return load(lambda: path.read_bytes(), scope=scope, root=root, what=f"file {path}")


def load_toml(path, *, scope: str = "job", root: int = 0):
    """The TOML document at ``path`` as a dict (``odatse.util.toml.load``)."""
    path = Path(path)
    return load(lambda: _toml.load(str(path)), scope=scope, root=root, what=f"input file {path}")


def loadtxt(path, *, scope: str = "job", root: int = 0, **kwargs) -> np.ndarray:
    """The array in the text file at ``path`` (``numpy.loadtxt(path, **kwargs)``)."""
    path = Path(path)
    return load(lambda: np.loadtxt(path, **kwargs), scope=scope, root=root, what=f"file {path}")


def load_json(path, *, encoding: str = "utf-8", scope: str = "job", root: int = 0):
    """The JSON document at ``path``."""
    path = Path(path)
    return load(lambda: json.loads(path.read_text(encoding=encoding)), scope=scope, root=root,
                what=f"file {path}")
