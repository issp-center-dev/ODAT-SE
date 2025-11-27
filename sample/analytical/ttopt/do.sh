#!/bin/sh

np=${1:-1}

export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_ackley.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_alpine.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_exponential.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_griewank.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_himmelblau.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_michaelwicz.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_qing.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_rastrigin.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_rosenbrock.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_schaffer.toml
export OPENBLAS_NUM_THREADS=1; mpiexec -np $np python3 ../../../src/odatse_main.py input_schwefel.toml

#python3 ../../../src/odatse_main.py input.toml
