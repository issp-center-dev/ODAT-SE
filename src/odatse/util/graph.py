# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


from typing import List

import collections
import numpy as np


def is_connected(nnlist: List[List[int]]) -> bool:
    """
    Check if the graph represented by the neighbor list is connected.

    Parameters
    ----------
    nnlist : List[List[int]]
        A list of lists where each sublist represents the neighbors of a node.

    Returns
    -------
    bool
        True if the graph is connected, False otherwise.
    """
    nnodes = len(nnlist)
    visited = np.full(nnodes, False)
    nvisited = 1
    visited[0] = True
    stack = collections.deque([0])
    while len(stack) > 0:
        node = stack.pop()
        neighbors = [n for n in nnlist[node] if not visited[n]]
        visited[neighbors] = True
        stack.extend(neighbors)
        nvisited += len(neighbors)

    return nvisited == nnodes


def is_bidirectional(nnlist: List[List[int]]) -> bool:
    """
    Check if the graph represented by the neighbor list is bidirectional.

    Parameters
    ----------
    nnlist : List[List[int]]
        A list of lists where each sublist represents the neighbors of a node.

    Returns
    -------
    bool
        True if the graph is bidirectional, False otherwise.
    """
    for i in range(len(nnlist)):
        for j in nnlist[i]:
            if i not in nnlist[j]:
                return False
    return True

if __name__ == "__main__":
    filename = "./neighborlist.txt"
    nnlist = []
    with open(filename) as f:
        for line in f:
            words = line.split()
            nn = [int(w) for w in words[1:]]
            nnlist.append(nn)
    print(is_connected(nnlist))
