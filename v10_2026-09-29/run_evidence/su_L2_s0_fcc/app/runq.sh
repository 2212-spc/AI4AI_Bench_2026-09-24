#!/bin/bash
# run a python script detached, immune to session hangups
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /app
setsid nohup python3 "$@" > "/app/log_$(basename $1 .py)_$(date +%H%M%S).txt" 2>&1 < /dev/null &
disown
