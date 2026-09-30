"""Train on the canonical config over several seeds and report validation MSE.

usage: python run.py [--config configs/canonical.json] [--seeds 0-7]
"""
import argparse, json, sys
import numpy as np

from minilab.data import load
from minilab.model import mse
from minilab.trainer import train


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(t) for t in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/canonical.json")
    ap.add_argument("--seeds", default="0-7")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    xtr, ytr, xva, yva = load("data")
    vals = []
    for s in parse_seeds(args.seeds):
        try:
            p = train(cfg, xtr, ytr, s)
            vals.append(mse(p, xva, yva))
        except FloatingPointError as e:
            print("seed", s, "diverged:", e)
            vals.append(float("inf"))
    vals = np.array(vals)
    print("val_mse per seed:", " ".join("%.5f" % v for v in vals))
    print("val_mse mean %.5f  std %.5f" % (vals.mean(), vals.std()))


if __name__ == "__main__":
    main()
