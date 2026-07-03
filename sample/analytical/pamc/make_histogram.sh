#!/bin/sh

export PYTHONUNBUFFERED=1

odatse_summarize_each_T -i input.toml

odatse_plt_1D_histogram --config config.toml
odatse_plt_2D_histogram --config config.toml
