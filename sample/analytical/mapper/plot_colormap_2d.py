# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt


def himmel(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


npts = 201
c_x, c_y = np.mgrid[-6 : 6 : npts * 1j, -6 : 6 : npts * 1j]
c_z = himmel(c_x, c_y)
levels = np.logspace(0.35, 3.2, 8)

x = []
y = []
f = []
file_input = open("output/ColorMap.txt", "r")
lines = file_input.readlines()
file_input.close()
for line in lines:
    if line[0] != "/n":
        data = line.split()
        x.append(float(data[0]))
        y.append(float(data[1]))
        f.append(np.log10(float(data[2])))

vmin = np.amin(np.array(f))
vmax = np.amax(np.array(f))

plt.contour(c_x, c_y, c_z, levels, colors="k", zorder=10.0, alpha=1.0, linewidths=0.5)
plt.scatter(
    x,
    y,
    c=f,
    s=50,
    vmin=vmin,
    vmax=vmax,
    cmap="Blues_r",
    linewidth=2,
    alpha=1.0,
    zorder=1.0,
)
plt.xlim(-6.0, 6.0)
plt.ylim(-6.0, 6.0)
plt.colorbar(label="log10(f)")
plt.savefig("output/ColorMapFig.pdf")
# plt.savefig("output/ColorMapFig.png")
