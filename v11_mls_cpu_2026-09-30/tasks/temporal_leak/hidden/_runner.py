"""Child-process runner (lock-step pipe).  Host -> child: length-prefixed pickles
   ("train", dump, train_rows)  -> child fits the fixed model on features.build(dump, train_rows)
   ("day", tables_asof, rows)   -> child returns predictions for rows (the next day is sent only after this reply)
The child never holds any data written after the day it is predicting."""
import os, pickle, struct, sys, traceback
out = os.fdopen(os.dup(1), "wb"); os.dup2(2, 1); sys.stdout = sys.stderr      # agent prints -> stderr
inp = sys.stdin.buffer
sys.path.insert(0, os.getcwd())
import numpy as np


def recv():
    h = inp.read(8)
    if len(h) < 8: return None
    return pickle.loads(inp.read(struct.unpack("<Q", h)[0]))


def send(obj):
    b = pickle.dumps(obj); out.write(struct.pack("<Q", len(b)) + b); out.flush()


def main():
    import features, model
    m = cols = None
    while True:
        msg = recv()
        if msg is None: return
        try:
            if msg[0] == "train":
                _, dump, rows = msg
                X = features.build(dump, rows[["store", "day"]].copy())
                cols = list(X.columns); m = model.fit(X, rows["units"].to_numpy()); send(("ok", None))
            else:
                _, tables, rows = msg
                X = features.build(tables, rows[["store", "day"]].copy())
                send(("ok", np.asarray(model.predict(m, X[cols]), float)))
        except Exception:
            send(("err", traceback.format_exc()[-1500:])); return


main()
