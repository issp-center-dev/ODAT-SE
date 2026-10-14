#!/bin/sh
#
# Resuming in parallel when the ranks' checkpoints are from different steps
#
# Every algorithm rank writes its own checkpoint, so a run stopped between two
# ranks' writes leaves them with checkpoints from different steps. The ranks
# of a lockstep algorithm (exchange here) must then agree on the step to
# restart from, instead of running different numbers of steps and blocking in
# the first collective.
#
# Test flow
# ---------
#   Part 1 : run input.toml (4 ranks, a checkpoint every 2000 of 10000 steps)
#            to completion; every rank has status.pickle (step 10000) and
#            status.pickle.1 (step 8001).
#   Case A : rank 1's status.pickle is replaced by its status.pickle.1.
#            --resume must restart every rank from step 8001, warn about it,
#            and reproduce the result of the original run (the restored RNG
#            replays steps 8001-10000).
#   Case B : rank 1's checkpoints are shifted to steps nobody else has.
#            --resume must fail on every rank, naming the steps, not hang.
#   Case C : rank 2 has no checkpoint at all.
#            --resume must fail on every rank, naming the rank, not hang.
#
# Each resume runs under a time limit: a hang is the failure mode this test
# looks for (status 124 from timeout.py).

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

CMD="mpiexec -np 4 ${PYTHON:-python3} ../../src/odatse_main.py"
TIMEOUT="${PYTHON:-python3} ../test_utilities/timeout.py"
LIMIT=300s

fail() {
  echo "TEST FAILED: $1"
  exit 1
}

# Part 1: the original run
rm -rf output output_orig
if ! time $CMD input.toml > run.log 2>&1; then
  cat run.log
  fail "the original run failed"
fi
for r in 0 1 2 3; do
  [ -f output/$r/status.pickle ] && [ -f output/$r/status.pickle.1 ] \
    || fail "rank $r has not written two checkpoint generations"
done
cp -r output output_orig

# Case A: one rank is a generation behind
cp output/1/status.pickle.1 output/1/status.pickle
$TIMEOUT $LIMIT $CMD --resume input.toml > resumeA.log 2>&1
status=$?
[ $status -eq 124 ] && fail "case A: the resume hung"
[ $status -eq 0 ] || { cat resumeA.log; fail "case A: the resume failed (status $status)"; }
grep -q "resuming every rank from istep = 8001" resumeA.log \
  || { cat resumeA.log; fail "case A: the ranks did not restart from the common step 8001"; }
echo "diff output/best_result.txt output_orig/best_result.txt"
diff output/best_result.txt output_orig/best_result.txt \
  || fail "case A: the resumed run does not reproduce the original result"

# Case B: one rank's checkpoints are from steps nobody else has
rm -rf output && cp -r output_orig output
PYTHONPATH=../../src ${PYTHON:-python3} -c "
import glob, pickle
for fn in glob.glob('output/1/status.pickle*'):
    with open(fn, 'rb') as f:
        d = pickle.load(f)
    d['istep'] += 7
    with open(fn, 'wb') as f:
        pickle.dump(d, f)
" || fail "case B: could not rewrite the checkpoints of rank 1"
$TIMEOUT $LIMIT $CMD --resume input.toml > resumeB.log 2>&1
status=$?
[ $status -eq 124 ] && fail "case B: the resume hung"
[ $status -ne 0 ] || fail "case B: the resume succeeded with checkpoints from different steps"
grep -q "no istep in common" resumeB.log \
  || { cat resumeB.log; fail "case B: the mismatch was not reported"; }

# Case C: one rank has no checkpoint
rm -rf output && cp -r output_orig output
rm -f output/2/status.pickle*
$TIMEOUT $LIMIT $CMD --resume input.toml > resumeC.log 2>&1
status=$?
[ $status -eq 124 ] && fail "case C: the resume hung"
[ $status -ne 0 ] || fail "case C: the resume succeeded without a checkpoint on rank 2"
grep -q "no readable checkpoint .* on rank 2" resumeC.log \
  || { cat resumeC.log; fail "case C: the missing checkpoint was not reported"; }

echo TEST PASS
true
