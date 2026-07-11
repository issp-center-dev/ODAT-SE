#!/bin/sh


export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Remove the output directory if it exists
rm -rf output

# Run the Python script with MPI
/usr/bin/time mpirun -np 8 ${PYTHON:-python3} ../parallel_solver.py --nalg 4 --nsolve 2 input.toml

# Define the result file path
resfile=output/ColorMap.txt

# The scrambled Sobol sequence is not seeded, so the sampled points differ
# from run to run. Check the structure of the output instead of comparing
# with a reference file.
echo check $resfile
res=0
${PYTHON:-python3} check_output.py $resfile 128 || res=$?

# Check the result
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile is broken or missing
  false
fi
