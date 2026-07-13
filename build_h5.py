"""Construit le .h5 poole d'un stade depuis les derivatives BIDS.

    python build_h5.py --root /scratch/alouis/dream_bids --branch ica \
        --stage S2 --out data/all_S2_ica.h5

Fenetre par defaut 1250 = 5s @ 250Hz. Chaque sujet est charge une fois,
epoche par stade, sous-fenetre, puis empile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bids_loader import analysis_subjects, make_subject_loader
from data import pooled_h5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--branch", default="noica", choices=["noica", "ica"])
    ap.add_argument("--deriv", default="derivatives_250hz_dl")
    ap.add_argument("--stage", required=True,
                    choices=["AWA", "S1", "S2", "SWS", "REM"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=1250)
    ap.add_argument("--hop", type=int, default=None)
    a = ap.parse_args()

    subjects, labels = analysis_subjects()
    loader = make_subject_loader(a.root, a.branch, a.deriv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    path = pooled_h5(a.out, a.stage, subjects, labels, loader,
                     win=a.win, hop=a.hop)
    print(f"ecrit : {path}")


if __name__ == "__main__":
    main()
