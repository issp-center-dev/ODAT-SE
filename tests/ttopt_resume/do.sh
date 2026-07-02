#!/bin/sh
#
# TTOpt checkpoint / resume test
#
# Strategy
# --------
# TTOpt saves a checkpoint after every complete double sweep (r2l + l2r).
# Because the sweep evaluations are fully deterministic for a fixed seed,
# resuming from a checkpoint and running to the same max_f_eval must yield
# a result identical to a single uninterrupted run.
#
# Test flow
# ---------
#   Part 1    : run input1.toml with a timeout.  The solver delay (0.02 s
#               per evaluation) ensures the timeout fires mid-run while at
#               least one checkpoint has already been written.
#   Resume    : --resume input1.toml continues from the last checkpoint.
#   Reference : input2.toml runs the same problem to completion in one shot
#               (no delay, no checkpoint) and writes results to output2/.
#
# Pass condition : output1/res.txt == output2/res.txt

if [ "$(uname)" = "Darwin" ]; then
  which gtimeout > /dev/null 2>&1 || { echo "gtimeout is not installed"; echo "Please install gtimeout using 'brew install coreutils'"; exit 1; }
  TIMEOUT="gtimeout"
else
  which timeout > /dev/null 2>&1 || { echo "timeout is not installed"; exit 1; }
  TIMEOUT="timeout"
fi

export PYTHONUNBUFFERED=1

CMD="${PYTHON:-python3} ../../src/odatse_main.py"
# CMD="mpiexec -np 2 ${PYTHON:-python3} ../../src/odatse_main.py"

# --- Part 1: initial run, interrupted by timeout ---
rm -rf output1
time ${TIMEOUT} 8s $CMD input1.toml

# Verify that a checkpoint file was created before the timeout fired
if [ ! -f output1/0/status.pickle ]; then
  echo "ERROR: Checkpoint file not found. Stop"
  exit 1
fi

# --- Resume: continue from the last checkpoint ---
time $CMD --resume input1.toml

# --- Reference: single uninterrupted run ---
rm -rf output2
time $CMD input2.toml

# --- Compare ---
resfile=output1/res.txt
reffile=output2/res.txt

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
