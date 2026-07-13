"""Entrainement/evaluation d'un fold. Early stopping sur la balanced
accuracy de VALIDATION (jamais le test)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler


def _balanced_sampler(labels):
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 len(w), replacement=True)


@torch.no_grad()
def _evaluate(model, loader, device, logprob=False):
    model.eval()
    probs, ys, subs = [], [], []
    for x, subj, y in loader:
        out = model(x.to(device))
        out = out.exp() if logprob else out.softmax(1)
        probs.append(out.cpu().numpy())
        ys.append(y.numpy())
        subs.append(subj.numpy())
    return (np.concatenate(probs), np.concatenate(ys), np.concatenate(subs))


def train_fold(model, train_ds, val_ds, test_ds, device,
               epochs=100, batch_size=64, lr=5e-5, patience=15,
               faithful=False):
    """faithful=True : NLLLoss (log-probs) + selection sur le fold de TEST
    (reproduit la fuite d'Anirudh). Sinon CrossEntropy + selection sur val."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.NLLLoss() if faithful else nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=_balanced_sampler(train_ds.label.numpy()))
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    monitor_loader = test_loader if faithful else val_loader

    best_val, best_state, since = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for x, _subj, y in train_loader:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        p, y, _ = _evaluate(model, monitor_loader, device, faithful)
        val_bacc = balanced_accuracy_score(y, p.argmax(1))
        if val_bacc > best_val:
            best_val, best_state, since = val_bacc, copy.deepcopy(
                model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    probs, ys, subs = _evaluate(model, test_loader, device, faithful)
    return {"probs": probs, "y": ys, "subj": subs, "val_bacc": best_val}
