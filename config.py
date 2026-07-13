"""Constantes stables du pipeline CNN dream-recall (format Alex : 250 Hz)."""
from __future__ import annotations

SFREQ = 250
SEGMENT_SEC = 5.0
N_TIMES = int(SFREQ * SEGMENT_SEC)  # 1250

# Ordre canonique branche alex / guided_backprop (les 19 premiers = EEG).
CH_NAMES = [
    "Fz", "Cz", "Pz", "C3", "C4", "T3", "T4", "Fp1", "Fp2", "O1",
    "O2", "F3", "F4", "P3", "P4", "FC1", "FC2", "CP1", "CP2",
]
N_CHANS = len(CH_NAMES)

STAGES = ["AWA", "S1", "S2", "SWS", "REM"]

# Polarite canon labo (Arthur) : LR -> 0, HR -> 1.
CLASS_NAMES = {0: "LR", 1: "HR"}

# Sujets (config.py branche alex). 21/22 exclus, raison inconnue.
HR_SUBJECTS = set(range(1, 19))
LR_SUBJECTS = {19, 20} | set(range(23, 39))
EXCLUDED_SUBJECTS = {21, 22}

BANDS = {
    "delta": (0.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 32.0),
    "gamma": (32.0, 42.0),
}

SEED = 42
