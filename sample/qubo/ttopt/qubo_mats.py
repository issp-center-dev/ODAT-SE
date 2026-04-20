import sys, os, time
import numpy as np
import scipy.io as spio
import networkx as nx
import qubogen

def qubo_maxcut(n: int, p: float, seed: int) -> np.ndarray:
    g=nx.fast_gnp_random_graph(n=n,p=p,seed=seed)
    g=qubogen.Graph(edges=g.edges,n_nodes=n)
    return qubogen.qubo_max_cut(g)

def qubo_mvc(n: int, p: float, seed: int) -> np.ndarray:
    g=nx.fast_gnp_random_graph(n=n,p=p,seed=seed)
    g=qubogen.Graph(edges=g.edges,n_nodes=n)
    return qubogen.qubo_mvc(g)

def qubo_qkp(n: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed=seed)
    v=np.diag(rng.random(n))/3
    a=rng.random(n)
    return qubogen.qubo_qkp(v,a,np.mean(a))

output_dir = "output_qubo_mats"
os.makedirs(output_dir, exist_ok=True)

functions = {
    "qubo_maxcut": [50, 100, 200],
    "qubo_mvc": [50, 100, 200],
    "qubo_qkp": [50, 100, 200],
}

for func_name, dims in functions.items():
    for dim in dims:
        if func_name == "qubo_maxcut":
            q_mats=[qubo_maxcut(dim,0.5,i) for i in range(10)]
        elif func_name == "qubo_mvc":
            q_mats=[qubo_mvc(dim,0.5,i) for i in range(10)]
        elif func_name == "qubo_qkp":
            q_mats=[qubo_qkp(dim,i) for i in range(10)]
        else:
            raise ValueError("Unknown function name: %s" % func_name)
        for i in range(len(q_mats)):
            spio.savemat(os.path.join(output_dir, f"{func_name}_{dim}_{i}.mat"), {"q_mat": q_mats[i]})
