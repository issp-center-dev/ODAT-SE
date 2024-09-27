#!/bin/sh

mpiexec -np 4 python3 ../../../src/odatse_main.py input.toml

for i in `seq 0 10`; do
  # python3 ./plot_result_2d.py -o output/res_T${i}.png output/0/result_T${i}.txt
  python3 ./plot_result_2d.py -o output/res_T${i}.pdf output/0/result_T${i}.txt
done
