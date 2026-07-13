"""Lecteur BIDS derivatives -> epochs 30s par stade, pour alimenter pooled_h5.

Sortie de preprocess_subject_v3.py (branche noica, 250Hz, 19 EEG) :
  {root}/derivatives/preprocessed-{branch}/sub-XX/eeg/
      sub-XX_task-sleep_proc-clean_eeg.vhdr      (signal continu)
      sub-XX_task-sleep_proc-clean_events.tsv    (stades, prefixes scorer)

Les stades sont prefixes par l'annotateur : 'per/Sleep stage S2', 'jbe/...'.
Scorer = 'per' par defaut, 'jbe' pour s19 (per_s19 corrompu). On lit le
_events.tsv directement (robuste, independant des marqueurs BrainVision).

    loader = make_subject_loader("/scratch/alouis/dream_bids", "noica")
    x = loader(1, "S2")          # (n_epochs, 7500, 19)
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

import config

# stade CNN -> descriptions (sans prefixe scorer). SWS = S3+S4.
STAGE_TO_ANNOT = {
    "AWA": {"Sleep stage W"},
    "S1": {"Sleep stage S1"},
    "S2": {"Sleep stage S2"},
    "SWS": {"Sleep stage S3", "Sleep stage S4"},
    "REM": {"Sleep stage R"},
}
EPOCH_SEC = 30.0


def subject_label(subj: int) -> int:
    if subj in config.HR_SUBJECTS:
        return 1
    if subj in config.LR_SUBJECTS:
        return 0
    raise ValueError(f"sujet {subj} hors HR/LR (exclu ?)")


def scorer_for(subj: int) -> str:
    return "jbe" if subj in config.PER_BLACKLIST else "per"


def _base(root, branch, subj) -> Path:
    sub = f"sub-{subj:02d}"
    return (Path(root) / "derivatives" / f"preprocessed-{branch}" / sub /
            "eeg" / f"{sub}_task-sleep_proc-clean")


def vhdr_path(root, branch, subj) -> Path:
    return _base(root, branch, subj).with_name(
        _base(root, branch, subj).name + "_eeg.vhdr")


def events_path(root, branch, subj) -> Path:
    return _base(root, branch, subj).with_name(
        _base(root, branch, subj).name + "_events.tsv")


def read_stage_events(path, scorer, stage):
    """Liste (onset_s, duration_s) des annotations du stade pour ce scorer."""
    want = {f"{scorer}/{d}" for d in STAGE_TO_ANNOT[stage]}
    out = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_on, i_dur, i_tt = (header.index("onset"), header.index("duration"),
                             header.index("trial_type"))
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[i_tt] in want:
                out.append((float(c[i_on]), float(c[i_dur])))
    return out


def slice_epochs(data, sf, events, win):
    """Decoupe les fenetres 30s (sous-decoupe si annotation plus longue)."""
    segs = []
    for onset, dur in events:
        start = int(round(onset * sf))
        d = int(round(dur * sf)) or win
        for s in range(start, start + d - win + 1, win):
            seg = data[:, s:s + win]
            if seg.shape[1] == win:
                segs.append(seg.T)
    return segs


def _load_raw(path):
    raw = mne.io.read_raw_brainvision(str(path), preload=True, verbose="ERROR")
    raw.pick(config.CH_NAMES[:config.N_CHANS])
    return raw


def load_subject_stage(root, branch, subj, stage):
    """(n_epochs, T, 19) pour un (sujet, stade). T = 30s * sfreq."""
    raw = _load_raw(vhdr_path(root, branch, subj))
    sf = raw.info["sfreq"]
    win = int(round(EPOCH_SEC * sf))
    data = raw.get_data()
    ev = read_stage_events(events_path(root, branch, subj),
                           scorer_for(subj), stage)
    segs = slice_epochs(data, sf, ev, win)
    if not segs:
        return np.empty((0, win, config.N_CHANS), dtype=np.float32)
    return np.asarray(segs, dtype=np.float32)


def make_subject_loader(root, branch="noica"):
    def loader(subj, stage):
        return load_subject_stage(root, branch, subj, stage)
    return loader


def analysis_subjects():
    subs = sorted(config.HR_SUBJECTS | config.LR_SUBJECTS)
    return subs, [subject_label(s) for s in subs]
