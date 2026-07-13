"""Test de cablage SANS donnees reelles : genere un .h5 synthetique
(8 sujets, effet de groupe plante) et verifie que toute la chaine tourne
en mode fidele ET moderne. A lancer sur le cluster pour valider l'install.

    python smoke_test.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from cv import build_pairs, nested_splits
from data import EEGSegments, channel_stats
from engine import train_fold
from models import DreamCNN
from stats import evaluate

N_TIMES, N_CHANS = 1250, 19


def make_h5(path, n_subj=8, seg=8, seed=0):
    rng = np.random.default_rng(seed)
    data, subj, label = [], [], []
    for s in range(1, n_subj + 1):
        lab = 0 if s <= n_subj // 2 else 1
        x = rng.standard_normal((seg, N_TIMES, N_CHANS)).astype("float32")
        x[:, :, 3] += 0.6 if lab == 0 else -0.6  # signal de groupe plante
        data.append(x)
        subj += [s] * seg
        label += [lab] * seg
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=np.concatenate(data))
        f.create_dataset("subj", data=np.array(subj))
        f.create_dataset("label", data=np.array(label))
    return np.concatenate(data), np.array(subj), np.array(label)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch {torch.__version__} | device {dev}")

    for faith in (True, False):
        out = DreamCNN(N_CHANS, N_TIMES, faithful=faith)(
            torch.zeros(3, 1, N_TIMES, N_CHANS))
        assert out.shape == (3, 2)
    print("[model] forward OK (faithful + moderne)")

    tmp = Path(tempfile.gettempdir()) / "smoke.h5"
    data, subj, label = make_h5(tmp)
    print(f"[data] h5 synthetique {data.shape}")

    groups, _ = build_pairs(subj, label)
    folds = list(nested_splits(subj, label, 42))
    tested = np.concatenate([subj[f["test"]] for f in folds])
    assert sorted(np.unique(tested)) == sorted(np.unique(subj))
    f0 = folds[0]
    assert not (set(subj[f0["train"]]) & set(subj[f0["test"]]))
    print(f"[cv] {len(folds)} folds, chaque sujet teste 1x, zero fuite OK")

    mean, std = channel_stats(data, f0["train"])
    ds = lambda i: EEGSegments(data[i], subj[i], label[i]).with_stats(mean, std)

    all_probs, all_y, all_subj = [], [], []
    for f in folds:
        m, s = channel_stats(data, f["train"])
        d = lambda i: EEGSegments(data[i], subj[i], label[i]).with_stats(m, s)
        r = train_fold(DreamCNN(N_CHANS, N_TIMES), d(f["train"]),
                       d(f["val"]), d(f["test"]), dev, epochs=3,
                       batch_size=16, patience=5)
        all_probs.append(r["probs"])
        all_y.append(r["y"])
        all_subj.append(r["subj"])
    res = evaluate(np.concatenate(all_probs), np.concatenate(all_y),
                   np.concatenate(all_subj), n_perm=200)
    print(f"[stats] {res}")
    print("\nOK : la chaine complete tourne.")


if __name__ == "__main__":
    main()
