"""Shared QUBO problem-instance definitions.

Both ``run_qubo_test.py`` (which feeds the instances to ODAT-SE) and
``qubo_mats.py`` (which dumps them as ``.mat`` files for the MATLAB
reference solver) import from here, so the two paths solve exactly the
same set of instances.
"""

import numpy as np
import networkx as nx
import qubogen

# Problem types and the dimensions (number of variables) to generate.
functions = {
    "qubo_maxcut": [50, 100, 200],
    "qubo_mvc": [50, 100, 200],
    "qubo_qkp": [50, 100, 200],
}

# Number of random instances generated per (problem type, dimension).
n_instances = 10

# Edge probability of the random graphs used by the graph-based problems.
graph_density = 0.5


def qubo_maxcut(n: int, p: float, seed: int) -> np.ndarray:
    g = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
    g = qubogen.Graph(edges=g.edges, n_nodes=n)
    return qubogen.qubo_max_cut(g)


def qubo_mvc(n: int, p: float, seed: int) -> np.ndarray:
    g = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
    g = qubogen.Graph(edges=g.edges, n_nodes=n)
    return qubogen.qubo_mvc(g)


def qubo_qkp(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    v = np.diag(rng.random(n)) / 3
    a = rng.random(n)
    return qubogen.qubo_qkp(v, a, np.mean(a))


def generate_instances(func_name: str, dim: int) -> list:
    """Return the list of QUBO matrices for a problem type and dimension."""
    if func_name == "qubo_maxcut":
        return [qubo_maxcut(dim, graph_density, i) for i in range(n_instances)]
    elif func_name == "qubo_mvc":
        return [qubo_mvc(dim, graph_density, i) for i in range(n_instances)]
    elif func_name == "qubo_qkp":
        return [qubo_qkp(dim, i) for i in range(n_instances)]
    else:
        raise ValueError("Unknown function name: %s" % func_name)
