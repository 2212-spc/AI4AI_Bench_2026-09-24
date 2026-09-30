#!/bin/bash
# lines: n steps eta
while read n s e; do
  r=$(/app/bin/lab train n=$n steps=$s eta=$e | tr -d '\n ' | sed 's/"call_id.*//')
  echo "n=$n steps=$s eta=$e $r"
done
