import os
import scipy.io as spio

from qubo_instances import functions, generate_instances

output_dir = "output_qubo_mats"
os.makedirs(output_dir, exist_ok=True)

for func_name, dims in functions.items():
    for dim in dims:
        q_mats = generate_instances(func_name, dim)
        for i in range(len(q_mats)):
            spio.savemat(os.path.join(output_dir, f"{func_name}_{dim}_{i}.mat"), {"q_mat": q_mats[i]})
