#!/bin/sh
#
# Mapper checkpoint / resume test
#
# Test flow
# ---------
#   Part 1    : run input1.toml and interrupt it as soon as the first
#               checkpoint has been written (every 40 of the 121 grid
#               points).  The solver delay (0.2 s per evaluation) leaves
#               several seconds between that checkpoint and the end of the
#               run, so the interruption lands mid-run.
#   Resume    : --resume input1.toml continues from the last checkpoint.
#   Reference : input2.toml evaluates the same grid in one shot
#               (no delay, no checkpoint) and writes results to output2/.
#
# Pass condition : output1/ColorMap.txt == output2/ColorMap.txt

# Interrupts a command once a file appears (see the script for details)
INTERRUPT="${PYTHON:-python3} ../test_utilities/interrupt_after_file.py"

export PYTHONUNBUFFERED=1

# Command to run the main Python script
CMD="${PYTHON:-python3} ../../src/odatse_main.py"
# Uncomment the following line to run with MPI
# CMD="mpiexec -np 2 ${PYTHON:-python3} ../../src/odatse_main.py"

# --- Part 1: initial run, interrupted after the first checkpoint ---
# The run is stopped when the checkpoint file appears rather than after a
# fixed time, which fired before the first checkpoint on slow machines and
# could not tell whether the run had been interrupted at all. The
# checkpoint is written atomically (temporary file + rename), so it is
# complete once it exists. 120 s only bounds a run that never writes one.
rm -rf output1
if ! ${INTERRUPT} output1/0/status.pickle 120s $CMD input1.toml; then
  echo "ERROR: the run was not interrupted after its first checkpoint. Stop"
  exit 1
fi

# The checkpoint must hold an unfinished run (the mapper also saves one
# when all points are done), otherwise the resume below would not
# evaluate anything.
if ! PYTHONPATH=../../src ${PYTHON:-python3} -c "
import math, pickle, sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('input1.toml', 'rb') as f:
    npoints = math.prod(tomllib.load(f)['algorithm']['param']['num_list'])
with open('output1/0/status.pickle', 'rb') as f:
    n = len(pickle.load(f)['results'])
print(f'checkpoint after {n} of {npoints} points')
sys.exit(0 if 0 < n < npoints else 1)
"; then
  echo "ERROR: the checkpoint is not from the middle of the run. Stop"
  exit 1
fi

# --- Resume: continue from the last checkpoint ---
if ! time $CMD --resume input1.toml; then
  echo "ERROR: the resumed run failed. Stop"
  exit 1
fi

# --- Reference: single uninterrupted run ---
rm -rf output2
if ! time $CMD input2.toml; then
  echo "ERROR: the reference run failed. Stop"
  exit 1
fi

# --- Compare ---
resfile=output1/ColorMap.txt
reffile=output2/ColorMap.txt

echo diff $resfile $reffile
res=0
diff $resfile $reffile || res=$?

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED: $resfile and $reffile differ
  false
fi
