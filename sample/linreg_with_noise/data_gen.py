import sys
import numpy as np
import matplotlib.pyplot as plt


def read_parameters(file_name):
    dict = {}
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            words = line.split("=")
            if len(words) != 2:
                sys.exit(f"Error: invalid line: {line}")
            key = words[0].strip()
            value = words[1].strip()
            if key == "a":
                value = value.rstrip(",")
                value = np.array([float(v) for v in value.split(",")])
            elif key == "noise_levels":
                value = value.rstrip(",")
                value = [float(v) for v in value.split(",")]
            elif key == "n":
                value = int(value)
            elif key == "seed":
                value = int(value)
            elif key == "datafile_prefix":
                value = value.strip()
            else:
                value = float(value)
            dict[key] = value
    return dict


parameter_file = sys.argv[1] if len(sys.argv) > 1 else "parameters.txt"

parameters = read_parameters(parameter_file)

np.random.seed(parameters.get("seed", 42))

n_samples = parameters.get("n", 50)
x_min = parameters.get("x_min", 0)
x_max = parameters.get("x_max", 10)
x = np.linspace(x_min, x_max, n_samples)
a = parameters.get("a", np.array([2.5]))
noise_levels = parameters.get("noise_levels", np.array([1.0, 3.0, 5.0]))
datafile_prefix = parameters.get("datafile_prefix", "data")

dim = len(a)
X = np.zeros((n_samples, dim))
X[:, 0] = x.copy()
for i in range(1, dim):
    X[:, i] = X[:, i - 1] * x
y_true = X @ a


file_names = []
for noise in noise_levels:
    y_noisy = y_true + np.random.normal(0, noise, size=n_samples)
    data = np.column_stack((x, y_noisy))
    file_name = f"{datafile_prefix}_noise{noise}.txt"
    np.savetxt(file_name, data, comments="", fmt="%.6f")
    file_names.append(file_name)
    plt.scatter(x, y_noisy, label=f"Noise level: $\\sigma^*={noise}$")

plt.plot(x, y_true, color="red", label=f"Coeffs: {a}", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Data with Gaussian Noise")
plt.legend()
plt.tight_layout()
plt.savefig(f"{datafile_prefix}.png")
