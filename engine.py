"""Entrainement/evaluation d'un fold.
- Early stopping sur la balanced accuracy de VALIDATION (jamais le test).
- Optim CUDA : DataLoader multi-workers + pin_memory, AMP bf16 (H100),
  cudnn.benchmark. Logging temps/epoch.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

torch.backends.cudnn.benchmark = True


def _balanced_sampler(labels):
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 len(w), replacement=True)


def _loader(ds, batch_size, workers, sampler=None, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      drop_last=False)


@torch.no_grad()
def _evaluate(model, loader, device, logprob=False, amp=False):
    model.eval()
    probs, ys, subs = [], [], []
    for x, subj, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        out = out.float()
        out = out.exp() if logprob else out.softmax(1)
        probs.append(out.cpu().numpy())
        ys.append(y.numpy())
        subs.append(subj.numpy())
    return (np.concatenate(probs), np.concatenate(ys), np.concatenate(subs))


def train_fold(model, train_ds, val_ds, test_ds, device,
               epochs=100, batch_size=64, lr=5e-5, patience=15,
               faithful=False, workers=4, verbose=True):
    """faithful=True : NLLLoss (log-probs) + selection sur le fold de TEST
    (reproduit la fuite d'Anirudh). Sinon CrossEntropy + selection sur val."""
    model.to(device)
    amp = device.startswith("cuda") if isinstance(device, str) else False
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.NLLLoss() if faithful else nn.CrossEntropyLoss()

    train_loader = _loader(train_ds, batch_size, workers,
                           sampler=_balanced_sampler(train_ds.label.numpy()))
    val_loader = _loader(val_ds, batch_size, 0)      # petits, pas de workers
    test_loader = _loader(test_ds, batch_size, 0)
    monitor_loader = test_loader if faithful else val_loader

    best_val, best_state, since = -1.0, None, 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        for x, _subj, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
        p, y, _ = _evaluate(model, monitor_loader, device, faithful, amp)
        val_bacc = balanced_accuracy_score(y, p.argmax(1))
        improved = val_bacc > best_val
        if improved:
            best_val, best_state, since = val_bacc, copy.deepcopy(
                model.state_dict()), 0
        else:
            since += 1
        if verbose:
            print(f"    epoch {ep:3d} | val_bacc {val_bacc:.3f} "
                  f"| best {best_val:.3f} | {time.time() - t0:.1f}s"
                  f"{'  *' if improved else ''}", flush=True)
        if since >= patience:
            if verbose:
                print(f"    early stop @ epoch {ep}", flush=True)
            break

    model.load_state_dict(best_state)
    probs, ys, subs = _evaluate(model, test_loader, device, faithful, amp)
    return {"probs": probs, "y": ys, "subj": subs, "val_bacc": best_val}
