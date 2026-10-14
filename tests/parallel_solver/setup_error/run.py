"""Run the odatse command with a failure injected into the set-up.

FAIL_WHERE (``solver`` or ``runner``) selects the constructor that raises
odatse.exception.InputError, ``makedirs`` makes os.makedirs raise
PermissionError; FAIL_RANK is the global rank on which it happens. Without
FAIL_WHERE the command runs unchanged. With DRIVER set, that script is run
(with the remaining arguments) instead of the odatse command.
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

import odatse
from odatse import exception
from odatse.solver import analytical

where = os.environ.get("FAIL_WHERE")
if where is not None:
    fail_rank = int(os.environ["FAIL_RANK"])
    if where == "makedirs":
        original_makedirs = os.makedirs

        def makedirs(*args, **kwargs):
            if odatse.mpi.rank() == fail_rank:
                raise PermissionError("injected makedirs failure")
            return original_makedirs(*args, **kwargs)

        os.makedirs = makedirs
    else:
        cls = {"solver": analytical.Solver, "runner": odatse.Runner}[where]
        original = cls.__init__

        def __init__(self, *args, **kwargs):
            if odatse.mpi.rank() == fail_rank:
                raise exception.InputError(f"injected {where} failure")
            original(self, *args, **kwargs)

        cls.__init__ = __init__

driver = os.environ.get("DRIVER")
if driver is not None:
    sys.argv = [driver] + sys.argv[1:]
    runpy.run_path(driver, run_name="__main__")
else:
    odatse._main.cli(sys.argv[1:])
