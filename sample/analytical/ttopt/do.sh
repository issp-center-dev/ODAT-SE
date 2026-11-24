#!/bin/sh

rm -rf output

time python3 ../../../src/odatse_main.py input.toml
python3 ./plot.py output/ttopt_history.txt --output output/res.pdf
