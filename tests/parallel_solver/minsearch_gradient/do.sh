#!/bin/sh

# Gradients with a parallel solver (--nsolve > 1). The solver of
# gradient_solver.py uses a collective on solcomm inside evaluate() and
# gradient(), so the workers must compute every gradient together with their
# controller. Checked with minsearch (L-BFGS-B, use_gradient = true):
#  * the run converges to the known minimum and uses the solver's gradient;
#  * a RuntimeError in gradient() on a worker is ignored with ignore_error =
#    true (normal end) and fails the job otherwise (non-zero exit);
#  * a ValueError in gradient() on a worker fails the job even with
#    ignore_error = true.
# No case may hang: each run is killed after a time limit, which fails it.

export OMP_NUM_THREADS=1

export PYTHONUNBUFFERED=1
export OMPI_MCA_rmaps_base_oversubscribe=1

# Seconds allowed for one run. A hang is one failure mode under test, so the
# run is killed (and the test fails) when this expires.
LIMIT=60

# run_with_timeout LIMIT CMD...: run CMD, kill it after LIMIT seconds.
# Sets $status to the exit status of CMD and $timed_out to 1 if it was killed.
run_with_timeout() {
  limit=$1; shift
  timed_out=0
  "$@" &
  pid=$!
  # The watchdog runs sleep in the background and waits for it, so that
  # killing the watchdog also stops the sleep (a foreground sleep would be
  # left behind as an orphan until it expires).
  ( sleep "$limit" & sp=$!
    trap 'kill "$sp" 2>/dev/null; exit 0' TERM
    wait "$sp"
    kill "$pid" 2>/dev/null && touch timed_out.flag ) &
  wd=$!
  wait "$pid"
  status=$?
  kill "$wd" 2>/dev/null
  wait "$wd" 2>/dev/null
  if [ -f timed_out.flag ]; then
    timed_out=1
    rm -f timed_out.flag
  fi
}

# run_case NAME FAILMODE FAILTYPE INPUT: run one job; leaves $status, $timed_out, $log.
run_case() {
  name=$1; mode=$2; type=$3; input=$4
  echo "=== $name (FAILMODE=$mode FAILTYPE=$type $input) ==="
  rm -rf output timed_out.flag
  log=log_$name.txt
  # env(1) rather than "VAR=... function": whether an assignment prefixed to a
  # shell-function call reaches the child processes is unspecified in POSIX.
  run_with_timeout $LIMIT env FAILMODE=$mode FAILTYPE=$type \
    mpirun -np 4 ${PYTHON:-python3} gradient_solver.py --nalg 2 --nsolve 2 "$input" > "$log" 2>&1
}

res=0

# framework messages that must not appear in a run whose failures are
# ignored (an MPI runtime may print unrelated diagnostics containing "ERROR")
no_reported_error() {
  if grep -q -e "ERROR: solver worker" -e "main() raised" -e "ERROR: mismatched collectives" "$log"; then
    echo "FAILED: an error was reported in $log"; res=1; return 1
  fi
  return 0
}

# expect_success NAME FAILMODE FAILTYPE INPUT: the job must exit with status 0
# without hanging and without reported errors.
expect_success() {
  run_case "$1" "$2" "$3" "$4"
  if [ $timed_out -ne 0 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"; res=1; return 1
  fi
  if [ $status -ne 0 ]; then
    echo "FAILED: job exited with status $status (see $log)"; res=1; return 1
  fi
  no_reported_error || return 1
  echo "ok: completed with status 0"
  return 0
}

# expect_failure NAME FAILMODE FAILTYPE INPUT PATTERN...: the job must exit
# non-zero without hanging, and the log must contain every PATTERN.
expect_failure() {
  run_case "$1" "$2" "$3" "$4"; shift 4
  if [ $timed_out -ne 0 ]; then
    echo "FAILED: job hung (killed after ${LIMIT}s)"; res=1; return
  fi
  if [ $status -eq 0 ]; then
    echo "FAILED: job exited with status 0 despite the error"; res=1; return
  fi
  for pat in "$@"; do
    if ! grep -q -F -- "$pat" "$log"; then
      echo "FAILED: '$pat' not found in $log (exit status $status)"; res=1; return
    fi
  done
  echo "ok: failed with status $status and the error was reported"
}

# 1. Converges with the solver's gradient, computed by both ranks of each group.
if expect_success converged none runtime input.toml; then
  echo "compare output/ with ref.txt"
  ODATSE_NOMPI=1 ${PYTHON:-python3} check.py output ref.txt || res=1
fi

# 2. A RuntimeError in gradient() on a worker is ignored with ignore_error =
#    true: the job ends normally (the marker shows the failure did happen) ...
if expect_success worker_runtime_ignored worker runtime input_ignore.toml; then
  if ! grep -q -F "gradient_solver: raising RuntimeError at gradient 2" "$log"; then
    echo "FAILED: the worker's gradient() never failed (see $log)"; res=1
  fi
fi

# ... and fails the job without ignore_error, reported by the controller for
# solver.gradient() with the worker's exception, and by the worker itself
# (its own ERROR line; the exception text alone is also in the controller's
# report).
expect_failure worker_runtime worker runtime input.toml \
  "gradient_solver: raising RuntimeError at gradient 2" \
  "solver.gradient()" \
  "RuntimeError: worker failed at gradient 2" \
  "ERROR: solver worker raised in gradient()"

# 3. ignore_error covers RuntimeError only: a ValueError in gradient() on a
#    worker surfaces as SolverError on the controller and fails the job; the
#    worker reports it as well, ignore_error notwithstanding.
expect_failure worker_value_ignore worker value input_ignore.toml \
  "gradient_solver: raising ValueError at gradient 2" \
  "main() raised SolverError" \
  "solver.gradient()" \
  "ValueError: worker failed at gradient 2" \
  "ERROR: solver worker raised in gradient()"

if [ $res -eq 0 ]; then
  echo TEST PASS
  true
else
  echo TEST FAILED
  false
fi
