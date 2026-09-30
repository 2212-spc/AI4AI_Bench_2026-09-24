#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /app/repo
echo "== real sample, 1000 steps, seeds 0/1 (dev gold)"
for s in 0 1; do
  python3 train.py --data ../data/sample.npz --steps 1000 --predict /tmp/dev_X.npy --out /tmp/p_$s.npy --seed $s
  python3 -c "import numpy as np; p=np.load('/tmp/p_$s.npy'); y=np.load('../data/dev.npz')['y']; print('seed $s dev acc', (p==y).mean(), 'dtype', p.dtype, 'shape', p.shape)"
done
echo "== synthetic 80k corpus, 20000 steps, production command (timing + gold test)"
for s in 0 1; do
  /usr/bin/time -v python3 train.py --data /tmp/synth_corpus.npz --steps 20000 --predict /tmp/synth_test_X.npy --out /tmp/sp_$s.npy --seed $s 2> /tmp/time_$s.txt
  grep -E "User time|Maximum resident" /tmp/time_$s.txt
  python3 -c "import numpy as np; p=np.load('/tmp/sp_$s.npy'); g=np.load('/tmp/synth_test_g.npy'); print('seed $s synth test acc', (p==g).mean())"
done
