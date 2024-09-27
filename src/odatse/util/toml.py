# 2DMAT -- Data-analysis software of quantum beam diffraction experiments for 2D material structure
# Copyright (C) 2020- The University of Tokyo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

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
