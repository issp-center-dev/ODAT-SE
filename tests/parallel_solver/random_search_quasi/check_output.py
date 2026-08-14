# Check the structure of the ColorMap file produced by the quasi-random
# search. The scrambled Sobol sequence is not seeded, so the sampled points
# differ from run to run and cannot be compared against a reference file.
# Verify instead that all points were evaluated and lie within the search
# range given in input.toml.
import sys

import numpy as np

resfile = sys.argv[1]
num_points = int(sys.argv[2])

# search range; must match [algorithm.param] in input.toml
min_list = [-6.0, -6.1]
max_list = [6.0, 6.0]

data = np.loadtxt(resfile, comments="#")

if data.shape != (num_points, len(min_list) + 1):
    sys.exit("ERROR: expected shape {}, got {}".format(
        (num_points, len(min_list) + 1), data.shape))

for k, (lo, hi) in enumerate(zip(min_list, max_list)):
    if not (np.all(data[:, k] >= lo) and np.all(data[:, k] <= hi)):
        sys.exit("ERROR: column {} out of range [{}, {}]".format(k, lo, hi))

if not np.all(np.isfinite(data[:, -1])):
    sys.exit("ERROR: fval contains non-finite values")

print("OK: {} points within bounds".format(num_points))
