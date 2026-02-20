#!/bin/sh

export PYTHONUNBUFFERED=1

python3 ../../../script/summarize_each_T.py -i input.toml

python3 ../../../script/plt_1D_histogram.py --config config.toml
python3 ../../../script/plt_2D_histogram.py --config config.toml
