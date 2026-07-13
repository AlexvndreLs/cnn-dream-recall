"""Agregation sujet-niveau + test de permutation.

Reduit chaque sujet a UNE decision (proba moyenne sur ses segments),
puis teste la balanced accuracy sur ces N sujets independants ->
resout d'un coup la pseudo-replication et la non-independance des folds.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def aggregate_subjects(probs, y, subj):
    """Moyenne les probas par sujet -> (subjects, y_true, y_pred)."""
    probs = np.asarray(probs)
    y = np.asarray(y)
    subj = np.asarray(subj)
    subs = np.unique(subj)
    y_true, y_pred = [], []
    for s in subs:
        m = subj == s
        y_true.append(int(np.round(y[m].mean())))
        y_pred.append(int(probs[m].mean(0).argmax()))
    return subs, np.array(y_true), np.array(y_pred)


def permutation_pvalue(y_true, y_pred, n_perm=5000, seed=42):
    """Permute les labels sujets pour batir la distribution nulle de la
    balanced accuracy. p = P(nul >= observe)."""
    rng = np.random.default_rng(seed)
    obs = balanced_accuracy_score(y_true, y_pred)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = balanced_accuracy_score(rng.permutation(y_true), y_pred)
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return obs, float(p), null


def evaluate(probs, y, subj, n_perm=5000, seed=42):
    _, y_true, y_pred = aggregate_subjects(probs, y, subj)
    bacc, p, _ = permutation_pvalue(y_true, y_pred, n_perm, seed)
    return {"subject_bacc": bacc, "p_perm": p, "n_subjects": len(y_true)}
