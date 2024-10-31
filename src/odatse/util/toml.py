# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import MutableMapping, Any

from ..mpi import rank

USE_TOML = False
OLD_TOMLI_API = False

try:
    import tomli

    if tomli.__version__ < "1.2.0":
        OLD_TOMLI_API = True
except ImportError:
    try:
        import toml

        USE_TOML = True
        if rank() == 0:
            print("WARNING: tomli is not found and toml is found.")
            print("         use of toml package is left for compatibility.")
            print("         please install tomli package.")
            print("HINT: python3 -m pip install tomli")
            print()
    except ImportError:
        if rank() == 0:
            print("ERROR: tomli is not found")
            print("HINT: python3 -m pip install tomli")
        raise


def load(path: str) -> MutableMapping[str, Any]:
    """read TOML file

    Parameters
    ----------
    path: str
        File path to an input TOML file

    Returns
    -------
    toml_dict: MutableMapping[str, Any]
        Dictionary representing TOML file

    """
    if USE_TOML:
        return toml.load(path)
    else:
        if OLD_TOMLI_API:
            with open(path, "r") as f:
                return tomli.load(f)
        else:
            with open(path, "rb") as f:
                return tomli.load(f)
