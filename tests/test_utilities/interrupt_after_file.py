#!/usr/bin/env python3
# Run a command and interrupt it as soon as a given file appears, so that the
# checkpoint/resume tests stop a run right after its first checkpoint instead
# of after a fixed number of seconds (whether a fixed timeout falls between
# the first checkpoint and the end of the run depends on the machine speed).
#
# Usage: python3 interrupt_after_file.py [-n COUNT] PATTERN LIMIT COMMAND [ARG]...
# PATTERN is a path or a glob pattern (quote it); the command is interrupted
# once at least COUNT files match it (default 1). A parallel run writes one
# checkpoint per rank: "-n 4 'output1/*/status.pickle'" waits for all four.
# LIMIT is a number of seconds, optionally with an s/m/h suffix, after which
# the command is killed even if the files have not appeared.
#
# Exit status:
#   0    the files appeared and the command was then stopped by our signal
#        (SIGTERM first, SIGKILL after a short grace period). The exit
#        status of the command is not inspected: a launcher such as mpiexec
#        stops its ranks on SIGTERM and exits with a status of its own
#        (Open MPI: 1), not with the signal.
#   1    the command exited before the files appeared
#   124  LIMIT expired before the files appeared
#   125  usage error, 127 command not found

import glob
import math
import os
import signal
import subprocess
import sys
import time

KILL_GRACE = 5.0
POLL_INTERVAL = 0.05

def parse_duration(text):
    units = {"s": 1, "m": 60, "h": 3600}
    scale = units.get(text[-1:], None)
    if scale is None:
        return float(text)
    return float(text[:-1]) * scale

def stop(p):
    """SIGTERM, then SIGKILL; return the signals that were sent."""
    sent = [signal.SIGTERM]
    p.terminate()
    try:
        p.wait(timeout=KILL_GRACE)
    except subprocess.TimeoutExpired:
        sent.append(signal.SIGKILL)
        p.kill()
        p.wait()
    return sent

def usage():
    print(f"Usage: {sys.argv[0]} [-n COUNT] PATTERN LIMIT COMMAND [ARG]...", file=sys.stderr)
    return 125

def main():
    args = sys.argv[1:]
    count = 1
    if args[:1] == ["-n"]:
        if len(args) < 2:
            return usage()
        try:
            count = int(args[1])
            if count < 1:
                raise ValueError
        except ValueError:
            print(f"invalid count: {args[1]}", file=sys.stderr)
            return 125
        args = args[2:]
    if len(args) < 3:
        return usage()
    pattern = args[0]
    try:
        limit = parse_duration(args[1])
        if not (math.isfinite(limit) and limit >= 0):
            raise ValueError
    except ValueError:
        print(f"invalid duration: {args[1]}", file=sys.stderr)
        return 125
    try:
        p = subprocess.Popen(args[2:])
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 127

    def appeared():
        return len(glob.glob(pattern)) >= count

    what = pattern if count == 1 else f"{count} files matching {pattern}"
    deadline = time.monotonic() + limit
    while not appeared():
        if p.poll() is not None:
            print(f"command exited (status {p.returncode}) before {what} appeared",
                  file=sys.stderr)
            return 1
        if time.monotonic() > deadline:
            stop(p)
            print(f"{what} did not appear within {args[1]}", file=sys.stderr)
            return 124
        time.sleep(POLL_INTERVAL)

    if p.poll() is not None:
        print(f"command exited (status {p.returncode}) before it could be interrupted",
              file=sys.stderr)
        return 1
    sent = stop(p)
    print(f"interrupted after {what} appeared (sent {', '.join(s.name for s in sent)}; "
          f"command status {p.returncode})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
