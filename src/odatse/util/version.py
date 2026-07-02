# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import re


def parse_version(version: str, ncomponents: int = 3) -> tuple:
    """Parse a dotted version string into a tuple of integers.

    Intended for numeric comparison of package versions, where a plain
    string comparison is wrong (e.g. "1.10.0" < "1.2.0") and a bare
    int() per component crashes on pre-release suffixes (e.g. "1.16.0rc1").

    Only the leading digits of each component are used; a component
    without leading digits and everything after it are ignored. The
    result is zero-padded to ncomponents elements.

    Parameters
    ----------
    version : str
        Dotted version string, e.g. "1.16.0rc1".
    ncomponents : int
        Number of components in the returned tuple.

    Returns
    -------
    tuple
        Tuple of ncomponents integers, e.g. (1, 16, 0).
    """
    comps = []
    for s in version.split(".")[:ncomponents]:
        m = re.match(r"[0-9]+", s)
        if m is None:
            break
        comps.append(int(m.group()))
    comps.extend([0] * (ncomponents - len(comps)))
    return tuple(comps)
