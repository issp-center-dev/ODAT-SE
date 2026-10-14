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
#   Part 1    : run input1.toml and interrupt it as soon as the first
#               checkpoint has been written.  The solver delay (0.02 s per
#               evaluation) leaves a few seconds between that checkpoint and
#               the end of the run, so the interruption lands mid-run.
#   Resume    : --resume input1.toml continues from the last checkpoint.
#   Reference : input2.toml runs the same problem to completion in one shot
#               (no delay, no checkpoint) and writes results to output2/.
#
# Pass condition : output1/res.txt == output2/res.txt

# Interrupts a command once a file appears (see the script for details)
INTERRUPT="${PYTHON:-python3} ../test_utilities/interrupt_after_file.py"

export PYTHONUNBUFFERED=1

CMD="${PYTHON:-python3} ../../src/odatse_main.py"
# CMD="mpiexec -np 2 ${PYTHON:-python3} ../../src/odatse_main.py"

# --- Part 1: initial run, interrupted after the first checkpoint ---
# The run is stopped when the checkpoint file appears rather than after a
# fixed time: the time of the first checkpoint depends on the machine speed,
# and a fixed timeout fired before it on slower machines. The checkpoint is
# written atomically (temporary file + rename), so it is complete once it
# exists. 120 s only bounds a run that never writes one.
rm -rf output1
if ! ${INTERRUPT} output1/0/status.pickle 120s $CMD input1.toml; then
  echo "ERROR: the run was not interrupted after its first checkpoint. Stop"
  exit 1
fi

# The checkpoint must hold an unfinished run (TTOpt also saves one when the
# evaluation budget is exhausted), otherwise the resume below would not
# evaluate anything.
if ! PYTHONPATH=../../src ${PYTHON:-python3} -c "
import pickle, sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('input1.toml', 'rb') as f:
    max_f_eval = tomllib.load(f)['algorithm']['ttopt']['max_f_eval']
with open('output1/0/status.pickle', 'rb') as f:
    n = pickle.load(f)['f_eval_count']
print(f'checkpoint at f_eval_count = {n} (max_f_eval = {max_f_eval})')
sys.exit(0 if 0 < n < max_f_eval else 1)
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
