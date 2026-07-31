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

# Use a Python replacement for timeout(1), which is not available on macOS
# without GNU coreutils
TIMEOUT="${PYTHON:-python3} ../test_utilities/timeout.py"

export PYTHONUNBUFFERED=1

CMD="${PYTHON:-python3} ../../src/odatse_main.py"
# CMD="mpiexec -np 2 ${PYTHON:-python3} ../../src/odatse_main.py"

# --- Part 1: initial run, interrupted by timeout ---
# The timeout must fire after the first checkpoint (~8 s: end of the first
# double sweep) but before the run completes (~10 s). 9.5 s sits in the
# middle of that window; 8 s was right at its lower edge and failed
# intermittently.
rm -rf output1
time ${TIMEOUT} 9.5s $CMD input1.toml

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
