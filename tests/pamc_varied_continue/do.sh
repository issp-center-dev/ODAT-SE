#!/bin/sh

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Command to run the Python script with MPI
CMD="mpiexec -np 2 ${PYTHON:-python3} ../../src/odatse_main.py"

# Remove the output1 directory if it exists
rm -rf output1

# Run the command with input1a.toml
time $CMD input1a.toml

# Run the command with input1b.toml in continuation mode
time $CMD --cont input1b.toml

# Remove the output2 directory if it exists
rm -rf output2

# Run the command with input2.toml
time $CMD input2.toml

# Compare every result file of the reference run: the per-rank trial.txt,
# result.txt and weight.txt as well (the continued run appends to them, and
# must not lose or repeat lines around the checkpoint it continues from),
# not only the derived fx.txt and best_result.txt.
res=0
for f in $(cd output2 && find . -name '*.txt' | sort); do
  echo diff output1/$f output2/$f
  diff output1/$f output2/$f || { res=$?; echo "TEST FAILED: output1/$f and output2/$f differ"; }
done

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  false
fi
