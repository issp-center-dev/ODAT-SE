# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import MutableMapping, Any

from ..mpi import rank

try:
    import tomllib
    # print("tomllib is available")
except ImportError:
    tomllib = None
    try:
        import tomli
        if tomli.__version__ < "1.2.0":
            use_old_tomli_api = True
            # print("tomli old api is available")
        else:
            use_old_tomli_api = False
            # print("tomli new api is available")
    except ImportError:
        tomli = None
        try:
            import toml
            if rank() == 0:
                # print("toml is available")
                print("WARNING: use of toml package is left for compatibility.")
                print("         please use tomli package instead.")
                print("HINT: python3 -m pip install tomli")
                print()
        except ImportError:
            print("ERROR: toml parser library is not available.")
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
    if tomllib:
        with open(path, "rb") as f:
            return tomllib.load(f)
    elif tomli:
        if use_old_tomli_api:
            with open(path, "r") as f:
                return tomli.load(f)
        else:
            with open(path, "rb") as f:
                return tomli.load(f)
    elif toml:
        return toml.load(path)
