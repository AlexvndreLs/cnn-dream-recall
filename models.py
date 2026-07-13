"""Architecture CNN modernisee, derivee de Kemtur (2020).

Corrections vs original :
- BatchNorm + activation appliquees des la 1re conv (bn1 etait inutilise).
- Sortie en logits bruts + CrossEntropyLoss (plus de ReLU avant log-softmax).
- Taille du flatten inferee dynamiquement -> supporte 250 Hz / 1250 points.
- Entree (B, 1, T, C), conv temporelle puis conv spatiale sur les electrodes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool=None, dropout=0.0,
                 dilation=1, bn_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_ch) if bn_act else None
        self.pool = nn.MaxPool2d(pool) if pool else None
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = F.elu(self.bn(x))
        if self.pool is not None:
            x = self.pool(x)
        if self.drop is not None:
            x = self.drop(x)
        return x


class DreamCNN(nn.Module):
    def __init__(self, n_chans=19, n_times=1250, n_classes=2,
                 temporal_kernel=50, temporal_filters=20, spatial_filters=80,
                 faithful=False):
        super().__init__()
        self.faithful = faithful
        # faithful : conv1 SANS bn/activation (reproduit le bn1 inutilise).
        self.temporal = ConvBlock(1, temporal_filters, (temporal_kernel, 1),
                                  bn_act=not faithful)
        self.spatial = ConvBlock(temporal_filters, spatial_filters,
                                 (1, n_chans), pool=(5, 1))
        self.block3 = ConvBlock(spatial_filters, 100, (5, 1), pool=(5, 1),
                                dropout=0.5, dilation=2)
        self.block4 = ConvBlock(100, 160, (10, 1), pool=(5, 1))
        n_feat = self._infer_flatten(n_chans, n_times)
        if faithful:
            # fc1->bn->relu->fc2->bn->relu->logsoftmax (+ NLLLoss cote engine).
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(n_feat, 100), nn.BatchNorm1d(100), nn.ReLU(),
                nn.Linear(100, n_classes), nn.BatchNorm1d(n_classes), nn.ReLU(),
                nn.LogSoftmax(dim=1),
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(n_feat, 100), nn.BatchNorm1d(100), nn.ReLU(),
                nn.Linear(100, n_classes),
            )

    def _features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.flatten(x, 1)

    def _infer_flatten(self, n_chans, n_times):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            n = self._features(torch.zeros(2, 1, n_times, n_chans)).shape[1]
        if was_training:
            self.train()
        return n

    def forward(self, x):
        return self.head(self._features(x))


def build_model(name="dreamcnn", **kwargs):
    if name == "dreamcnn":
        return DreamCNN(**kwargs)
    if name == "braindecode":
        from braindecode.models import Deep4Net
        return Deep4Net(kwargs["n_chans"], kwargs["n_classes"],
                        n_times=kwargs["n_times"], final_conv_length="auto")
    raise ValueError(name)
