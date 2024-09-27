#!/bin/sh

python3 ../../../src/odatse_main.py input.toml

# python3 ../plot_himmel.py --xcol=1 --ycol=2 --format="-o" --output=output/res.pdf output/BayesData.txt
# python3 ../plot_himmel.py --xcol=4 --ycol=5 --format="o" --output=output/actions.pdf output/BayesData.txt

python3 ../plot_himmel.py --xcol=1 --ycol=2 --format="-o" --output=output/res.png output/BayesData.txt
python3 ../plot_himmel.py --xcol=4 --ycol=5 --format="o" --output=output/actions.png output/BayesData.txt
