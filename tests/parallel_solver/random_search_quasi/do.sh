#!/bin/sh


export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Remove the output directories if they exist
rm -rf output output1 output2 output_serial

# The quasi-random sequence is scrambled with the seed given in input.toml.
# The points are generated on the algorithm-rank-0 process and distributed,
# so the same seed must yield identical results independently of the MPI
# configuration. The exact values depend on the scipy version, so instead of
# comparing with a reference file, run the calculation three times and check
# that the results agree with each other:
#   run 1: 8 processes (nalg=4, nsolve=2)
#   run 2: 8 processes (nalg=4, nsolve=2) -- determinism
#   run 3: 1 process   (nalg=1, nsolve=1) -- independence of parallelism

/usr/bin/time mpirun -np 8 ${PYTHON:-python3} ../parallel_solver.py --nalg 4 --nsolve 2 input.toml
mv output output1

/usr/bin/time mpirun -np 8 ${PYTHON:-python3} ../parallel_solver.py --nalg 4 --nsolve 2 input.toml
mv output output2

/usr/bin/time mpirun -np 1 ${PYTHON:-python3} ../parallel_solver.py --nalg 1 --nsolve 1 input.toml
mv output output_serial

resfile=output1/ColorMap.txt

res=0

# Check the structure of the result (number of points, bounds, finite values)
echo check $resfile
${PYTHON:-python3} check_output.py $resfile 128 || res=$?

# Check that two runs with the same seed give identical results
echo diff output1/ColorMap.txt output2/ColorMap.txt
diff output1/ColorMap.txt output2/ColorMap.txt || res=$?

# Check that the result does not depend on the MPI configuration
echo diff output1/ColorMap.txt output_serial/ColorMap.txt
diff output1/ColorMap.txt output_serial/ColorMap.txt || res=$?

# Check the result
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
