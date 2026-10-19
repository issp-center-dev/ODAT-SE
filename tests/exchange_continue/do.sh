#!/bin/sh

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Command to run the Python script with MPI
CMD="mpiexec -np 4 ${PYTHON:-python3} ../../src/odatse_main.py"

# Remove the output1 directory if it exists
rm -rf output1

# Run the command with the first input file and measure the time taken
time $CMD input1a.toml || { echo "TEST FAILED: input1a.toml failed"; exit 1; }
# Run the command with the continuation input file and measure the time taken
time $CMD --cont input1b.toml || { echo "TEST FAILED: --cont input1b.toml failed"; exit 1; }

# Remove the output2 directory if it exists
rm -rf output2

# Run the command with the second input file and measure the time taken
time $CMD input2.toml || { echo "TEST FAILED: input2.toml failed"; exit 1; }

# Compare every result file of the reference run: the per-rank trial.txt
# and result.txt as well (the continued run appends to them, and must not
# lose or repeat lines around the checkpoint it continues from), not only
# the derived result_T*.txt and best_result.txt.
files=$(cd output2 && find . -name '*.txt' | sort)
[ -n "$files" ] || { echo "TEST FAILED: no result files in output2"; exit 1; }
res=0
for f in $files; do
  echo diff output1/$f output2/$f
  diff output1/$f output2/$f || { res=$?; echo "TEST FAILED: output1/$f and output2/$f differ"; }
done

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  false
fi
