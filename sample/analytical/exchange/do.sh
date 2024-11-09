#!/bin/sh

mpiexec -np 4 python3 ../../../src/odatse_main.py input.toml

for i in `seq 0 79`
do
  python3 ../plot_himmel.py --xcol=3 --ycol=4 --skip=20 --format="o" --output=output/res_T${i}.png output/result_T${i}.txt
done

# python3 ./plot_himmel_multi.py --layout=2x2 --xcol=3 --ycol=4 --skip=20 --format="o" --output=res.png output/result_T79.txt output/result_T60.txt output/result_T40.txt output/result_T0.txt
