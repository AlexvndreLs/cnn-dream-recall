"""Cross-validation sujet-niveau, appariee HDR/LDR, avec val separee du test.

Corrige les deux failles du papier :
- vraie partition (chaque sujet en test exactement 1 fois -> 18 folds),
- val distincte du test -> selection de modele sans fuite.
"""
from __future__ import annotations

import numpy as np


def build_pairs(subjects, labels):
    """Apparie le k-e HDR au k-e LDR (robuste a la numerotation)."""
    subjects = np.asarray(subjects)
    labels = np.asarray(labels)
    hdr = np.unique(subjects[labels == 0])
    ldr = np.unique(subjects[labels == 1])
    n = min(len(hdr), len(ldr))
    pair_of = {}
    for k in range(n):
        pair_of[hdr[k]] = k
        pair_of[ldr[k]] = k
    return np.array([pair_of[s] for s in subjects]), pair_of


def nested_splits(subjects, labels, seed=42):
    """Genere les folds : test = 1 paire, val = 1 paire, train = reste."""
    subjects = np.asarray(subjects)
    groups, _ = build_pairs(subjects, labels)
    pairs = np.unique(groups[groups >= 0]) if (groups < 0).any() \
        else np.unique(groups)
    rng = np.random.default_rng(seed)
    for test_pair in pairs:
        remaining = pairs[pairs != test_pair]
        val_pair = int(rng.choice(remaining))
        test = np.isin(groups, test_pair)
        val = np.isin(groups, val_pair)
        train = ~(test | val)
        yield {
            "train": np.where(train)[0],
            "val": np.where(val)[0],
            "test": np.where(test)[0],
            "test_pair": int(test_pair),
        }
