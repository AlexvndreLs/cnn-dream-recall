"""Trace l'historique d'entrainement (loss + metriques vs epoch) a partir
des .npz par fold. Moyenne inter-folds + enveloppe (min/max) par epoch.

    python plot_history.py --glob "res_S2/fold_*.npz" --out fig_S2.png
"""
from __future__ import annotations

import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ["loss", "acc", "bacc", "f1", "precision", "recall",
           "specificity", "auc", "kappa", "mcc"]


def _pad_stack(curves):
    """Empile des courbes de longueurs differentes (early-stop) -> (n, Lmax)
    completees par nan."""
    if not curves:
        return np.empty((0, 0))
    Lmax = max(len(c) for c in curves)
    out = np.full((len(curves), Lmax), np.nan)
    for i, c in enumerate(curves):
        out[i, :len(c)] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default="history.png")
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    if not files:
        raise SystemExit(f"aucun fichier pour {a.glob}")
    data = [np.load(f) for f in files]

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, m in zip(axes.ravel(), METRICS):
        for split, color in [("train", "tab:blue"), ("val", "tab:orange")]:
            key = f"hist_{split}_{m}"
            curves = [d[key] for d in data if key in d]
            M = _pad_stack(curves)
            if M.size == 0:
                continue
            mean = np.nanmean(M, axis=0)
            lo, hi = np.nanmin(M, axis=0), np.nanmax(M, axis=0)
            x = np.arange(len(mean))
            ax.plot(x, mean, color=color, label=split)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Historique d'entrainement ({len(files)} folds)", y=1.02)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print(f"figure -> {a.out}")


if __name__ == "__main__":
    main()
