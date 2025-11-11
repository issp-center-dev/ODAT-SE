import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 50
x = np.linspace(0, 10, n_samples)

a = 2.5
y_true = a * x

noise_levels = [1, 3, 5]

file_names = []
for noise in noise_levels:
    y_noisy = y_true + np.random.normal(0, noise, size=n_samples)
    data = np.column_stack((x, y_noisy))
    file_name = f"data_noise{noise}.txt"
    np.savetxt(file_name, data, comments='', fmt="%.6f")
    file_names.append(file_name)
    plt.scatter(x, y_noisy, label=f'Noise level: $\\sigma^*={noise}$')

plt.plot(x, y_true, color='red', label=f'True line: $y={a}x$', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Data with Gaussian Noise')
plt.legend()
plt.tight_layout()
plt.savefig("linear_data_gaussian_noise.png")
