export OMP_NUM_THREADS=1
cd /app/work
(python3 /app/repo/train.py --data corpusA.npz --steps 20000 --predict testA_X.npy --out pA1.npy --seed 1 > seed1.log 2>&1; python3 -c "
import numpy as np; yt=np.load('testA_y.npy'); print('FULL seed 1 acc',np.mean(np.load('pA1.npy')==yt))" >> seed1.log) &
python3 runexp.py A 80000 20000 '{"weight_decay":0.67}' 0.35 classdep > cd1.log 2>&1 &
python3 runexp.py A 80000 20000 '{"weight_decay":3.0}' 0.35 classdep > cd2.log 2>&1 &
python3 runexp.py B 80000 20000 '{"weight_decay":0.67,"ensemble":2,"avg_frac":0.25}' > ensB.log 2>&1 &
wait; echo done > last_done.flag
