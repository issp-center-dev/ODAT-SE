import sys, os, time
import odatse
from odatse.algorithm import choose_algorithm
import numpy as np
import pandas as pd
import networkx as nx
import qubogen
np.set_printoptions(threshold=np.inf)

class QUBOSolver(odatse.solver.SolverBase):
    def __init__(self, info, **kwargs):
        super().__init__(info)

        self.q_mat = kwargs.get("q_mat", None)

        self.opt_x = None
        self.opt_fx = np.inf

    def evaluate(self, xs, args, nprocs=1, nthreads=1):
        res = qubo(xs, self.q_mat)
        return res

def qubo(x: np.ndarray, q_mat: np.ndarray) -> np.ndarray:
    return np.einsum("i...,ij,j...",x,q_mat,x)

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

n_trials = 10

functions = {
    "qubo_maxcut": [50, 100, 200],
    "qubo_mvc": [50, 100, 200],
    "qubo_qkp": [50, 100, 200],
}

dfs = []
for func_name, dims in functions.items():
    for dim in dims:
        min_list = [0] * dim
        max_list = [1] * dim
        p_points = [2] * dim
        q_points = [1] * dim

        if func_name == "qubo_maxcut":
            q_mats=[qubo_maxcut(dim,0.5,i) for i in range(10)]
        elif func_name == "qubo_mvc":
            q_mats=[qubo_mvc(dim,0.5,i) for i in range(10)]
        elif func_name == "qubo_qkp":
            q_mats=[qubo_qkp(dim,i) for i in range(10)]
        else:
            raise ValueError("Unknown function name: %s" % func_name)
        for i in range(len(q_mats)):
            df=pd.DataFrame(columns=["f","dim","instance","min_params","min_f","time"])
            for j in range(n_trials):
                output_dir = "output/output_%s_d%d_i%d/trial%d" % (func_name, dim, i, j)
                os.makedirs(output_dir, exist_ok=True)
                # generate input file
                toml_content = f"""[base]
dimension = {dim}
output_dir = "{output_dir}"

[solver]
name = "custom"
function_name = "{func_name}"

[algorithm]
name = "ttopt"
seed = {12345+j}

[algorithm.param]
max_list = {max_list}
min_list = {min_list}

[algorithm.ttopt]
p_points = {p_points}
q_points = {q_points}
r_max = 4
max_f_eval = 100000
"""
                toml_filename = f"input_{func_name}.toml"
                if odatse.mpi.rank() == 0:
                    with open(toml_filename, "w") as f:
                        f.write(toml_content)
                sys.argv = ["script.py", toml_filename, "--init"]
                info, run_mode = odatse.initialize()
                output_dir = info.base.get("output_dir", "./output")
                os.makedirs(output_dir, exist_ok=True)
                solver = QUBOSolver(info, q_mat = q_mats[i])
                runner = odatse.Runner(solver, info)
                alg_module = choose_algorithm(info.algorithm["name"])
                alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
                time0 = time.time()
                result = alg.main()
                time1 = time.time()
                elapsed_time = time1 - time0
                df.loc[len(df)]=[func_name,dim,i,result["x"],result["fx"],elapsed_time]
                df=df.sort_index()

                # cleanup
                if odatse.mpi.rank() == 0:
                    if os.path.exists(toml_filename):
                        os.remove(toml_filename)
    
            dfs.append(df)
df=pd.concat(dfs)
if odatse.mpi.rank() == 0:
    print("QUBO problems")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df[["f","dim","instance","min_f","time"]].groupby(["f","dim","instance"]).agg(["mean","std","min"]))
    df.to_csv("qubo_results.csv", index=False)
    df[["f","dim","instance","min_f","time"]].groupby(["f","dim","instance"]).agg(["mean","std","min"]).to_csv("qubo_results_agg.csv")
