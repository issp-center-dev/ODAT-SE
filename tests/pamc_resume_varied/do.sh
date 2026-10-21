#!/bin/sh
#
# PAMC checkpoint / resume test under MPI, with populations that die out
#
# fix_num_replicas = false, two walkers per rank and resampling at every
# temperature: some ranks lose all their walkers during the run.  Such a rank
# skips the sampling loop, so its istep stops while the others go on; the
# ranks must nevertheless checkpoint at the same temperatures and resume
# together (they agree on (Tindex, index_from_reset), not on istep).
#
# Every algorithm rank writes its own checkpoint. The ranks decide together
# when to write (rank 0 decides) and publish the files together, so that
# after an interruption they all resume from the same step; a resumed run
# must then reproduce an uninterrupted one.
#
# Test flow
# ---------
#   Part 1    : run input1.toml on 4 ranks and interrupt it as soon as all
#               ranks have written their first checkpoint (after the first
#               of the 21 temperatures).  The solver delay (0.03 s per
#               evaluation) leaves several seconds between that checkpoint
#               and the end of the run, so the interruption lands mid-run.
#   Check     : the checkpoints of the four ranks are from the same
#               temperature, strictly inside the run.
#   Resume    : --resume input1.toml continues from that checkpoint.
#   Reference : input2.toml runs the same problem to completion in one shot
#               (no delay, no checkpoint) and writes results to output2/.
#
# Pass condition : every result file of output2/ (best_result.txt, fx.txt,
#                  pr.txt, and result.txt / trial.txt / weight.txt and
#                  their per-temperature splits of each rank) is identical
#                  in output1/.  The output files are flushed at every
#                  checkpoint and cut back to the checkpointed position on
#                  resume, so the resumed run must reproduce the
#                  uninterrupted one line for line.

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

NP=4
CMD="mpiexec -np $NP ${PYTHON:-python3} ../../src/odatse_main.py"
# Interrupts a command once files appear (see the script for details)
INTERRUPT="${PYTHON:-python3} ../test_utilities/interrupt_after_file.py"

fail() {
  echo "TEST FAILED: $1"
  exit 1
}

# --- Part 1: initial run, interrupted after the first checkpoint of all ranks ---
rm -rf output1
${INTERRUPT} -n $NP 'output1/*/status.pickle' 120s $CMD input1.toml \
  || fail "the run was not interrupted after its first checkpoint"

# --- Check: one and the same mid-run temperature on every rank ---
PYTHONPATH=../../src ${PYTHON:-python3} -c "
import pickle, sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('input1.toml', 'rb') as f:
    numT = tomllib.load(f)['algorithm']['pamc']['Tnum']
progress = []
for r in range($NP):
    with open(f'output1/{r}/status.pickle', 'rb') as f:
        d = pickle.load(f)
        progress.append((d['Tindex'], d['index_from_reset'], d['istep'], d['nwalkers']))
print('checkpoints at (Tindex, index_from_reset, istep, nwalkers) =', progress, f'(Tnum = {numT})')
same = len(set(p[:2] for p in progress)) == 1
sys.exit(0 if same and 0 < progress[0][0] < numT - 1 else 1)
" || fail "the checkpoints are not from one and the same temperature in the middle of the run"

# --- Resume: continue from the checkpoint ---
time $CMD --resume input1.toml || fail "the resumed run failed"

# --- Reference: single uninterrupted run ---
rm -rf output2
time $CMD input2.toml || fail "the reference run failed"

# --- Compare: every .txt file of the reference run ---
files=$(cd output2 && find . -name '*.txt' | sort)
files1=$(cd output1 && find . -name '*.txt' | sort)
[ "$files1" = "$files" ] || { echo "output1: $files1"; echo "output2: $files"; echo "TEST FAILED: output1 and output2 do not hold the same result files"; exit 1; }
[ -n "$files" ] || fail "no result files in output2"
for f in $files; do
  echo "diff output1/$f output2/$f"
  diff output1/$f output2/$f || fail "output1/$f and output2/$f differ"
done

echo TEST PASS
true
