import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 50
x = np.linspace(0, 10, n_samples)
a = np.array([2.5])

dim = len(a)
X = np.zeros((n_samples, dim))
X[:, 0] = x.copy()
for i in range(1, dim):
    X[:, i] = X[:, i - 1] * x
y_true = X @ a

noise_levels = [1, 3, 5]

file_names = []
for noise in noise_levels:
    y_noisy = y_true + np.random.normal(0, noise, size=n_samples)
    data = np.column_stack((x, y_noisy))
    file_name = f"data_noise{noise}.txt"
    np.savetxt(file_name, data, comments="", fmt="%.6f")
    file_names.append(file_name)
    plt.scatter(x, y_noisy, label=f"Noise level: $\\sigma^*={noise}$")

plt.plot(x, y_true, color="red", label=f"Coeffs: {a}", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Data with Gaussian Noise")
plt.legend()
plt.tight_layout()
plt.savefig("linear_data_gaussian_noise.png")
