#!/bin/bash
# Resolve the reference solver relative to this script rather than to an absolute mount point: the corpus
# contains both conventions (the solution directory mounted at /solution, and the solution files copied
# beside the world in /app), and a script that hard-codes either one breaks under the other.  This file
# used to say `python3 /app/ref_solve.py`, a path that only existed while family C's self-check was the
# only thing that ran it.
set -e
python3 "$(dirname "$0")/ref_solve.py" "${APP:-/app}"
