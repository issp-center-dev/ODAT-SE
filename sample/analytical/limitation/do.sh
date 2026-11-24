#!/bin/bash

set -e

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

mpiexec -np 10 python3 ../../../src/odatse_main.py input.toml


resfile=output/best_result.txt

# Compare with the reference within a tolerance: an exact diff is too strict
# because the last digits vary with the numpy/scipy/MPI build.
echo ${PYTHON:-python3} ../../../tests/test_utilities/diff_res_mc.py $resfile ref.txt
res=0
${PYTHON:-python3} ../../../tests/test_utilities/diff_res_mc.py $resfile ref.txt || res=$?
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and ref.txt differ
  false
fi


python3 hist2d_limitation_sample.py -p 10 -i input.toml -b 0.1
python3 hist2d_limitation_sample.py -p 10 -i input.toml -b 0.1 --layout 2,3 --tlist 9,7,5,3,1,0
python3 hist2d_limitation_sample.py -p 10 -i input.toml -b 0.1 --layout 2,3 --tlist 9,7,5,3,1,0 --format pdf

echo "done."
