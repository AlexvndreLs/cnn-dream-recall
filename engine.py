"""Entrainement/evaluation d'un fold, avec historique complet par epoch.

- Early stopping sur la balanced accuracy de VALIDATION (jamais le test).
- Optim CUDA : DataLoader multi-workers + pin_memory, AMP bf16 (H100),
  cudnn.benchmark.
- Metriques par epoch (train + val) : loss, acc, balanced_acc, f1,
  precision, recall/sensibilite, specificite, roc_auc, kappa, mcc.
  Historique renvoye pour plot ulterieur.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, WeightedRandomSampler

torch.backends.cudnn.benchmark = True

METRIC_KEYS = ["loss", "acc", "bacc", "f1", "precision", "recall",
               "specificity", "auc", "kappa", "mcc"]


def _metrics(y_true, y_pred, prob1, loss):
    """Panoplie standard classification binaire. prob1 = proba classe 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = (y_true == y_pred).mean()
    try:
        auc = roc_auc_score(y_true, prob1)
    except ValueError:
        auc = float("nan")
    return {
        "loss": float(loss),
        "acc": float(acc),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1,
                                            zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1,
                                      zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0,
                                           zero_division=0)),
        "auc": float(auc),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def _balanced_sampler(labels):
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    w = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 len(w), replacement=True)


def _loader(ds, batch_size, workers, sampler=None):
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0)


@torch.no_grad()
def _evaluate(model, loader, device, crit, logprob=False, amp=False):
    """Retourne prob(classe1), y_true, y_pred, subj, loss_moyenne."""
    model.eval()
    probs, ys, subs, losses, ns = [], [], [], 0.0, 0
    for x, subj, y in loader:
        x = x.to(device, non_blocking=True)
        yt = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(x)
            loss = crit(out, yt)
        out = out.float()
        p = out.exp() if logprob else out.softmax(1)
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
        subs.append(subj.numpy())
        losses += loss.item() * len(y)
        ns += len(y)
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return probs, ys, np.concatenate(subs), losses / max(ns, 1)


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
    val_loader = _loader(val_ds, batch_size, 0)
    test_loader = _loader(test_ds, batch_size, 0)
    monitor_loader = test_loader if faithful else val_loader

    history = {f"train_{k}": [] for k in METRIC_KEYS}
    history.update({f"val_{k}": [] for k in METRIC_KEYS})
    history["epoch_time"] = []

    best_val, best_state, since = -1.0, None, 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tr_y, tr_pred, tr_p1, tr_loss, ntr = [], [], [], 0.0, 0
        for x, _subj, y in train_loader:
            x = x.to(device, non_blocking=True)
            yt = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(x)
                loss = crit(out, yt)
            loss.backward()
            opt.step()
            with torch.no_grad():
                p = out.float().exp() if faithful else out.float().softmax(1)
            tr_y.append(y.numpy())
            tr_pred.append(p.argmax(1).cpu().numpy())
            tr_p1.append(p[:, 1].detach().cpu().numpy())
            tr_loss += loss.item() * len(y)
            ntr += len(y)
        tr_m = _metrics(np.concatenate(tr_y), np.concatenate(tr_pred),
                        np.concatenate(tr_p1), tr_loss / max(ntr, 1))

        vp, vy, _, vloss = _evaluate(model, monitor_loader, device, crit,
                                     faithful, amp)
        va_m = _metrics(vy, vp.argmax(1), vp[:, 1], vloss)

        for k in METRIC_KEYS:
            history[f"train_{k}"].append(tr_m[k])
            history[f"val_{k}"].append(va_m[k])
        history["epoch_time"].append(time.time() - t0)

        improved = va_m["bacc"] > best_val
        if improved:
            best_val, best_state, since = va_m["bacc"], copy.deepcopy(
                model.state_dict()), 0
        else:
            since += 1
        if verbose:
            print(f"    ep {ep:3d} | tr_loss {tr_m['loss']:.3f} "
                  f"va_loss {va_m['loss']:.3f} | va_bacc {va_m['bacc']:.3f} "
                  f"va_f1 {va_m['f1']:.3f} va_auc {va_m['auc']:.3f} "
                  f"| {history['epoch_time'][-1]:.1f}s{'  *' if improved else ''}",
                  flush=True)
        if since >= patience:
            if verbose:
                print(f"    early stop @ epoch {ep}", flush=True)
            break

    model.load_state_dict(best_state)
    probs, ys, subs, _ = _evaluate(model, test_loader, device, crit,
                                   faithful, amp)
    hist = {f"hist_{k}": np.asarray(v) for k, v in history.items()}
    return {"probs": probs, "y": ys, "subj": subs,
            "val_bacc": best_val, **hist}
