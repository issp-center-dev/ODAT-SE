#!/usr/bin/env python3
# Run a command and interrupt it as soon as a given file appears, so that the
# checkpoint/resume tests stop a run right after its first checkpoint instead
# of after a fixed number of seconds (whether a fixed timeout falls between
# the first checkpoint and the end of the run depends on the machine speed).
#
# Usage: python3 interrupt_after_file.py FILE LIMIT COMMAND [ARG]...
# LIMIT is a number of seconds, optionally with an s/m/h suffix, after which
# the command is killed even if FILE has not appeared.
#
# Exit status:
#   0    FILE appeared and the command was terminated by our signal
#        (SIGTERM first, SIGKILL after a short grace period)
#   1    the command exited before FILE appeared, or ended by itself (or by
#        another signal) before it could be interrupted
#   124  LIMIT expired before FILE appeared
#   125  usage error, 127 command not found

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

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} FILE LIMIT COMMAND [ARG]...", file=sys.stderr)
        return 125
    path = sys.argv[1]
    try:
        limit = parse_duration(sys.argv[2])
        if not (math.isfinite(limit) and limit >= 0):
            raise ValueError
    except ValueError:
        print(f"invalid duration: {sys.argv[2]}", file=sys.stderr)
        return 125
    try:
        p = subprocess.Popen(sys.argv[3:])
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 127

    deadline = time.monotonic() + limit
    while not os.path.exists(path):
        if p.poll() is not None:
            print(f"command exited (status {p.returncode}) before {path} appeared",
                  file=sys.stderr)
            return 1
        if time.monotonic() > deadline:
            stop(p)
            print(f"{path} did not appear within {sys.argv[2]}", file=sys.stderr)
            return 124
        time.sleep(POLL_INTERVAL)

    if p.poll() is not None:
        print(f"command exited (status {p.returncode}) before it could be interrupted",
              file=sys.stderr)
        return 1
    sent = stop(p)
    if p.returncode not in [-sig for sig in sent]:
        # it ended by itself (or by another signal) before ours arrived
        print(f"command exited (status {p.returncode}) before it could be interrupted",
              file=sys.stderr)
        return 1
    print(f"interrupted after {path} appeared")
    return 0

if __name__ == "__main__":
    sys.exit(main())
