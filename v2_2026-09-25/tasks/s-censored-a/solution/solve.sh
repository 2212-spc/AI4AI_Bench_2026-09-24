#!/bin/bash
set -e
cp /app/../solution/*.py /tmp/ 2>/dev/null || true
python3 "$(dirname "$0")/ref_solve.py" /app
