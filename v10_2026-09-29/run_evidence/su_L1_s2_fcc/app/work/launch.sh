#!/bin/bash
# usage: launch.sh LOG PROXY N OVERRIDE_JSON...
log=$1; shift
setsid nohup python3 /app/work/scale2.py "$@" > "$log" 2>&1 < /dev/null &
