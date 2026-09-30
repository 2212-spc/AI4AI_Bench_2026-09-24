cd /app/work
for s in 0 1; do
  /usr/bin/time -f "orig seed $s: cpu %U s, wall %e s" python3 train_orig.py --config /app/work/config_orig.json --data synth_corpus.npz --steps 20000 --predict synth_test_X.npy --out so$s.npy --seed $s
  python3 -c "import numpy as np; print('orig seed $s test acc', (np.load('so$s.npy')==np.load('synth_test_y.npy')).mean())"
  /usr/bin/time -f "new seed $s: cpu %U s, wall %e s" python3 /app/repo/train.py --data synth_corpus.npz --steps 20000 --predict synth_test_X.npy --out sn$s.npy --seed $s
  python3 -c "import numpy as np; print('new seed $s test acc', (np.load('sn$s.npy')==np.load('synth_test_y.npy')).mean())"
done
