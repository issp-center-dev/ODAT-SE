#!/usr/bin/env python3
# Minimal cross-platform replacement for coreutils timeout(1), so that the
# checkpoint/resume tests do not require GNU coreutils (gtimeout) on macOS.
#
# Usage: python3 timeout.py DURATION COMMAND [ARG]...
# DURATION is a number of seconds, optionally with an s/m/h suffix as in
# timeout(1). Exits with the command's exit code, or 124 if the timeout
# fired (SIGTERM first, SIGKILL after a short grace period).

import subprocess
import sys

KILL_GRACE = 5.0

def parse_duration(text):
    units = {"s": 1, "m": 60, "h": 3600}
    scale = units.get(text[-1:], None)
    if scale is None:
        return float(text)
    return float(text[:-1]) * scale

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} DURATION COMMAND [ARG]...", file=sys.stderr)
        return 125
    try:
        duration = parse_duration(sys.argv[1])
    except ValueError:
        print(f"invalid duration: {sys.argv[1]}", file=sys.stderr)
        return 125
    try:
        p = subprocess.Popen(sys.argv[2:])
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 127
    try:
        return p.wait(timeout=duration)
    except subprocess.TimeoutExpired:
        p.terminate()
        try:
            p.wait(timeout=KILL_GRACE)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        return 124

if __name__ == "__main__":
    sys.exit(main())
