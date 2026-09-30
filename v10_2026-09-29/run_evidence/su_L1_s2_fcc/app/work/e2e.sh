#!/bin/bash
# usage: e2e.sh PROXY SEED LOG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
p=$1; s=$2; log=$3
{ /usr/bin/time -f "cpu-user %U s  wall %e s  maxrss %M KB" python3 /app/work/cand/train.py --data /tmp/corpus$p.npz --steps 20000 --predict /tmp/test${p}_X.npy --out /tmp/pred_${p}_$s.npy --seed $s
  python3 -c "import numpy as np; p=np.load('/tmp/pred_${p}_$s.npy'); y=np.load('/tmp/test${p}_y.npy'); print('proxy $p seed $s TEST ACC', (p==y).mean())"; } > $log 2>&1
