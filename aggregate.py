"""Reunit les predictions par-fold (.npz) et calcule le score final :
balanced accuracy sujet-niveau + test de permutation sur les 36 sujets.

    python aggregate.py --glob "res_S2/fold_*.npz" --out res_S2.json
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from stats import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="motif des .npz par fold")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-perm", type=int, default=5000)
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"aucun fichier pour {a.glob}")
    probs, y, subj = [], [], []
    for f in files:
        d = np.load(f)
        probs.append(d["probs"])
        y.append(d["y"])
        subj.append(d["subj"])
    probs = np.concatenate(probs)
    y = np.concatenate(y)
    subj = np.concatenate(subj)

    res = evaluate(probs, y, subj, n_perm=a.n_perm)
    res["n_folds"] = len(files)
    print(f"[aggregate] {len(files)} folds | {len(np.unique(subj))} sujets")
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
