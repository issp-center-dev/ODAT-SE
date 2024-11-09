#!/bin/bash

set -e

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

mpiexec -np 10 python3 ../../../src/odatse_main.py input.toml


echo diff output/best_result.txt ref.txt
res=0
diff output/best_result.txt ref.txt || res=$?
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: best_result.txt and ref.txt differ
  false
fi


python3 hist2d_limitation_sample.py -p 10 -i input.toml -b 0.1
python3 hist2d_limitation_sample.py -p 10 -i input.toml -b 0.1 --layout 2,3 --tlist 9,7,5,3,1,0
python3 hist2d_limitation_sample.py -p 10 -i input.toml -b 0.1 --layout 2,3 --tlist 9,7,5,3,1,0 --format pdf

echo "done."
