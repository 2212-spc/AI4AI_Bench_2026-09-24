export OMP_NUM_THREADS=1
for wd in 0.3 0.67 1.0 3.0; do python3 runexp.py lin 80000 20000 "{\"weight_decay\":$wd}" & done; wait
for wd in 0.3 0.67 1.0 3.0; do python3 runexp.py B 80000 20000 "{\"weight_decay\":$wd}" & done; wait
for nk in classdep; do for wd in 0.67; do python3 runexp.py A 80000 20000 "{\"weight_decay\":$wd}" 0.35 $nk & python3 runexp.py A 80000 20000 "{\"weight_decay\":$wd,\"label_smoothing\":0.1}" 0.35 $nk & done; done; wait
python3 runexp.py A 80000 20000 '{"weight_decay":0.67,"ensemble":3}' & python3 runexp.py A 80000 20000 '{"weight_decay":0.67,"ensemble":2,"avg_frac":0.3}' & python3 runexp.py A 80000 20000 '{"weight_decay":0.67,"batch_size":256}' & python3 runexp.py A 80000 20000 '{"weight_decay":0.67,"lr":0.002}' & wait
echo DONE
