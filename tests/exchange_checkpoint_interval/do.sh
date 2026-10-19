#!/bin/sh
#
# Time-based checkpoints of a parallel exchange run land at the same step on
# every rank
#
# With checkpoint_interval alone, each rank used to compare its own clock
# with the interval, so the ranks wrote their checkpoints at different steps
# and could not be resumed together. The decision is now taken on rank 0 and
# broadcast, and the checkpoints are published only once every rank has
# written its own.
#
# Test flow
# ---------
#   Part 1 : run input.toml (4 ranks, checkpoint_interval = 1 s, no
#            checkpoint_steps) to completion: several checkpoints per rank.
#   Check  : for the current checkpoint and every previous generation, the
#            step (istep) is the same on all ranks; at least one checkpoint
#            was written by the timer, i.e. before the final one.
#   Resume : --resume from the completed run must succeed without falling
#            back to an older generation (no "resuming every rank" warning).
#
# Pass condition : the steps agree and the resume prints no fallback warning

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

CMD="mpiexec -np 4 ${PYTHON:-python3} ../../src/odatse_main.py"
TIMEOUT="${PYTHON:-python3} ../test_utilities/timeout.py"

fail() {
  echo "TEST FAILED: $1"
  exit 1
}

rm -rf output
if ! time $CMD input.toml > run.log 2>&1; then
  cat run.log
  fail "the run failed"
fi

PYTHONPATH=../../src ${PYTHON:-python3} -c "
import pickle, sys
from pathlib import Path
nranks = 4
steps = []
for r in range(nranks):
    d = Path('output', str(r))
    mine = []
    for fn in [d / 'status.pickle'] + [d / f'status.pickle.{i}' for i in (1, 2, 3)]:
        if fn.exists():
            with open(fn, 'rb') as f:
                mine.append(pickle.load(f)['istep'])
    steps.append(mine)
    print(f'rank {r}: istep of the generations = {mine}')
if any(s != steps[0] for s in steps[1:]):
    print('ERROR: the ranks wrote their checkpoints at different steps')
    sys.exit(1)
# some generation older than the final checkpoint; not necessarily .1, which
# has the final step too when the timer fired on the last step
if len(steps[0]) < 2 or min(steps[0]) >= steps[0][0]:
    print('ERROR: no timer-driven checkpoint before the final one')
    sys.exit(1)
" || fail "the checkpoints do not agree across the ranks"

$TIMEOUT 300s $CMD --resume input.toml > resume.log 2>&1
status=$?
[ $status -eq 124 ] && fail "the resume hung"
[ $status -eq 0 ] || { cat resume.log; fail "the resume failed (status $status)"; }
if grep -q "resuming every rank from" resume.log; then
  cat resume.log
  fail "the resume had to fall back to an older generation"
fi

echo TEST PASS
true
