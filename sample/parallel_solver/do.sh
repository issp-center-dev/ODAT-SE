#!/bin/sh

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

/usr/bin/time mpirun -np 4 python3 parallel_solver.py -m 2 -n 2
