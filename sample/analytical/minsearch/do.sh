#!/bin/sh

rm -rf output

time python3 ../../../src/odatse_main.py input.toml
# time mpirun -np 16 python3 ../../../src/odatse_main.py input.toml

# python3 plot.py --xcol=1 --ycol=2 --output=output/res.png output/*/SimplexData.txt
python3 plot.py --xcol=1 --ycol=2 --output=output/res.pdf output/*/SimplexData.txt
