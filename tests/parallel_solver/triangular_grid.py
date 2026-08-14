from typing import Optional, Sequence

import sys
import argparse
import numpy as np

def generate(xrange, yrange, a):
    xgrid = list(np.arange(xrange[0], xrange[1] + 0.1*a, a))
    ygrid = list(np.arange(yrange[0], yrange[1] + 0.1*a, a * np.sqrt(3)/2))

    grid = []
    offset = 0
    for y in ygrid:
        grid += [ (x + offset * a/2, y) for x in xgrid if x + offset * a/2 <= xrange[1] ]
        offset = 1 - offset

    return grid

def output(grid, filename):        
    with open(filename, "w") as fp:
        idx = 0
        for x, y in grid:
            idx += 1
            fp.write(f"{idx} {x} {y}\n")
    
def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", type=float, default=1.0, help="lattice constant")
    parser.add_argument("-x", type=str, default="-6.0,6.0", help="x range, e.g. -6.0,6.0")
    parser.add_argument("-y", type=str, default="-6.0,6.0", help="y range, e.g. -6.0,6.0")
    args = parser.parse_args(argv)

    xrange = list(map(float, args.x.split(",")))
    yrange = list(map(float, args.y.split(",")))

    grid = generate(xrange, yrange, args.a)
    output(grid, "MeshData.txt")

if __name__ == "__main__":
    main()
    
