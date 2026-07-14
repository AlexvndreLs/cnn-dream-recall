"""Orchestrateur : charge un .h5 poole d'un stade, boucle les folds,
agrege sujet-niveau et calcule la p-value par permutation.

Ex : python run.py --h5 data/all_S2.h5 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import config
from cv import nested_splits
from data import EEGSegments, channel_stats, load_h5
from engine import train_fold
from models import build_model
from stats import evaluate


def run(h5_path, device="cpu", epochs=100, batch_size=64, lr=5e-5,
        model_name="dreamcnn", seed=config.SEED, faithful=False,
        max_folds=None, workers=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_start = time.time()
    print(f"[run] h5={h5_path} device={device} faithful={faithful} "
          f"epochs={epochs} bs={batch_size}", flush=True)
    if device.startswith("cuda"):
        print(f"[run] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    data, subj, label = load_h5(h5_path)
    n_times, n_chans = data.shape[1], data.shape[2]
    print(f"[run] data {data.shape} chargee en {time.time()-t_start:.0f}s",
          flush=True)

    all_probs, all_y, all_subj = [], [], []
    folds = list(nested_splits(subj, label, seed))
    if max_folds:
        folds = folds[:max_folds]
    for i, fold in enumerate(folds):
        tf = time.time()
        print(f"\n[fold {i+1}/{len(folds)}] test paire {fold['test_pair']} "
              f"| train={len(fold['train'])} val={len(fold['val'])} "
              f"test={len(fold['test'])}", flush=True)
        mean, std = channel_stats(data, fold["train"])

        def ds(idx):
            return EEGSegments(data[idx], subj[idx], label[idx]).with_stats(
                mean, std)

        model = build_model(model_name, n_chans=n_chans, n_times=n_times,
                            n_classes=2, faithful=faithful)
        res = train_fold(model, ds(fold["train"]), ds(fold["val"]),
                         ds(fold["test"]), device, epochs, batch_size, lr,
                         faithful=faithful, workers=workers)
        print(f"[fold {i+1}] fini en {time.time()-tf:.0f}s "
              f"| val_bacc {res['val_bacc']:.3f}", flush=True)
        all_probs.append(res["probs"])
        all_y.append(res["y"])
        all_subj.append(res["subj"])

    probs = np.concatenate(all_probs)
    y = np.concatenate(all_y)
    s = np.concatenate(all_subj)
    return evaluate(probs, y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--model", default="dreamcnn")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-folds", type=int, default=None,
                    help="limite le nb de folds (test rapide)")
    ap.add_argument("--faithful", action="store_true",
                    help="reproduit Anirudh a l'identique (bugs inclus)")
    a = ap.parse_args()
    res = run(a.h5, a.device, a.epochs, a.batch_size, a.lr, a.model,
              faithful=a.faithful, max_folds=a.max_folds, workers=a.workers)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
