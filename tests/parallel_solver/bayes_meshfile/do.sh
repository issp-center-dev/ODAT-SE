#!/bin/sh


export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1

# physbo is an optional dependency of the bayes algorithm: when it is not
# installed the test is skipped (exit status 77, SKIP_RETURN_CODE in
# CMakeLists.txt). Only its absence skips: an installed physbo that fails to
# import makes the test fail as before.
${PYTHON:-python3} -c 'import importlib.util, sys; sys.exit(77 if importlib.util.find_spec("physbo") is None else 0)'
status=$?
if [ $status -eq 77 ]; then
  echo "physbo is not installed: test skipped"
  exit 77
elif [ $status -ne 0 ]; then
  echo "ERROR: could not check for physbo (status $status)"
  exit 1
fi
export OMPI_MCA_rmaps_base_oversubscribe=1

# Remove the output directory if it exists
rm -rf output

# Run the Python script with MPI
#/usr/bin/time mpirun -np 1 ${PYTHON:-python3} ../parallel_solver.py --nalg 1 --nsolve 1 input.toml
#/usr/bin/time mpirun -np 4 ${PYTHON:-python3} ../parallel_solver.py --nalg 2 --nsolve 2 input.toml
/usr/bin/time mpirun -np 8 ${PYTHON:-python3} ../parallel_solver.py --nalg 4 --nsolve 2 input.toml

# Define the result file path
resfile=output/BayesData.txt
reffile=ref_BayesData.txt

# Compare the result file with the reference file
echo diff $resfile $reffile
res=0
diff $resfile $reffile || ${PYTHON:-python3} ../../almost_diff.py $resfile $reffile || res=$?

# Check the result of the diff command
if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi
