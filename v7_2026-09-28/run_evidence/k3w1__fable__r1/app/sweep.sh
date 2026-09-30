#!/bin/bash
# usage: sweep.sh n steps eta1 eta2 ...
n=$1; steps=$2; shift 2
for eta in "$@"; do
  r=$(/app/bin/lab train n=$n steps=$steps eta=$eta | tr -d '\n ')
  echo "n=$n steps=$steps eta=$eta $r"
done
