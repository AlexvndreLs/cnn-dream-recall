"""Chargement des donnees, fenetrage et construction du .h5 poole par stade.

Le builder pooled_h5() comble le script manquant du repo original :
il assemble par sujet -> un .h5 par stade avec data / subj / label.
Brancher `subject_loader` sur ton lecteur BIDS (retourne un array
(n_epochs, T, C) pour un (sujet, stade) donne).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def window_epochs(x, win, hop=None):
    """(n_epochs, T, C) -> (n_windows, win, C). hop=None => non chevauchant.

    Le fenetrage se fait par epoch ; l'appelant garantit qu'un sujet ne
    traverse jamais train/test (fenetrage APRES le split sujet).
    """
    hop = hop or win
    x = np.asarray(x)
    n, T, C = x.shape
    if T < win:
        raise ValueError(f"epoch {T} < fenetre {win}")
    starts = range(0, T - win + 1, hop)
    out = [x[:, s:s + win, :] for s in starts]
    return np.concatenate(out, axis=0)


def pooled_h5(out_path, stage, subjects, labels,
              subject_loader: Callable[[int, str], np.ndarray],
              win=None, hop=None):
    """Ecrit un .h5 poole : datasets 'data' (N,T,C), 'subj' (N,), 'label' (N,).

    subjects : ids sujets ; labels : 0=HDR 1=LDR (memes longueurs).
    subject_loader(subj, stage) -> (n_epochs, T, C) en microvolts.
    """
    out_path = Path(out_path)
    data_parts, subj_parts, lab_parts = [], [], []
    for subj, lab in zip(subjects, labels):
        x = subject_loader(subj, stage)
        if x is None or len(x) == 0:
            continue
        if win:
            x = window_epochs(x, win, hop)
        data_parts.append(x.astype(np.float32))
        subj_parts.append(np.full(len(x), subj, np.int64))
        lab_parts.append(np.full(len(x), lab, np.int64))
    data = np.concatenate(data_parts)
    subj = np.concatenate(subj_parts)
    lab = np.concatenate(lab_parts)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip")
        f.create_dataset("subj", data=subj)
        f.create_dataset("label", data=lab)
        f.attrs["stage"] = stage
        f.attrs["sfreq_ok"] = True
    return out_path


class EEGSegments(Dataset):
    """Dataset a partir d'un .h5 poole. Normalisation z-score par-canal
    calee sur des stats fournies (train) pour eviter toute fuite."""

    def __init__(self, data, subj, label, mean=None, std=None):
        self.data = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        self.subj = torch.as_tensor(np.asarray(subj), dtype=torch.long)
        self.label = torch.as_tensor(np.asarray(label), dtype=torch.long)
        self.mean = mean
        self.std = std

    def with_stats(self, mean, std):
        self.mean, self.std = mean, std
        return self

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        x = self.data[i]
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return x.unsqueeze(0), self.subj[i], self.label[i]


def load_h5(path):
    with h5py.File(path, "r") as f:
        return (np.asarray(f["data"]), np.asarray(f["subj"]),
                np.asarray(f["label"]))


def channel_stats(data, idx):
    """z-score par canal calcule sur les indices train uniquement."""
    x = np.asarray(data)[idx]
    mean = x.mean(axis=(0, 1), keepdims=False)
    std = x.std(axis=(0, 1), keepdims=False) + 1e-7
    return (torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32))
