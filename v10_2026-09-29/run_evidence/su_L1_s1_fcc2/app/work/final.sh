export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /app/work
# (1) real sample regression, exact production command shape
( for s in 0 1; do python3 /app/repo/train.py --data /app/data/sample.npz --steps 1000 --predict dev_X.npy --out preal$s.npy --seed $s --log_every 1000; done
  python3 -c "
import numpy as np; yd=np.load('/app/data/dev.npz')['y']
for s in (0,1): print('REAL seed',s,'dev acc',np.mean(np.load(f'preal{s}.npy')==yd))" ) > final_real.log 2>&1 &
# (2) full-scale on proxy A: timing + accuracy, both seeds
( python3 mkproxy.py && for s in 0 1; do /usr/bin/time -f "seed $s CPU user=%U sys=%S wall=%e maxrss=%MkB" python3 /app/repo/train.py --data corpusA.npz --steps 20000 --predict testA_X.npy --out pA$s.npy --seed $s --log_every 20000; done
  python3 -c "
import numpy as np; yt=np.load('testA_y.npy')
for s in (0,1): print('FULL seed',s,'acc',np.mean(np.load(f'pA{s}.npy')==yt))" ) > final_full.log 2>&1 &
# (3) class-dependent noise  (4) ensemble vs single
( python3 runexp.py A 80000 20000 '{"weight_decay":0.67}' 0.35 classdep; python3 runexp.py A 80000 20000 '{"weight_decay":3.0}' 0.35 classdep ) > final_classdep.log 2>&1 &
( python3 runexp.py A 80000 20000 '{"weight_decay":0.67,"ensemble":2,"avg_frac":0.25}'; python3 runexp.py B 80000 20000 '{"weight_decay":0.67,"ensemble":2,"avg_frac":0.25}' ) > final_ens.log 2>&1 &
wait; echo ALLDONE > final_done.flag
