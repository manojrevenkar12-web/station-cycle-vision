"""Frame encoder and temporal models.

FrameEncoder: three consecutive frames (motion is in their differences) -> a 128-d feature. The
spatial map is kept (no global pooling) because *where* the hands are is most of the signal.

Temporal models over encoder features, (B, T, D) -> (B, T, K) logits:
  FrameOnly      a per-frame classifier; no temporal context
  CausalTCN      MS-TCN-style dilated convolutions that only look back; runs live, one frame at a time
  OfflineTransformer  bidirectional attention over the whole recording; re-segments after the fact
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

FEAT = 128
STACK = 3  # frames per encoder input


def stack_frames(frames: torch.Tensor) -> torch.Tensor:
    """(T, 64, 64, 3) uint8 -> (T, 9, 64, 64) float: frame t with t-1 and t-2 (repeating the first)."""
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    prev1 = torch.cat([x[:1], x[:-1]])
    prev2 = torch.cat([x[:1], x[:1], x[:-2]])[: len(x)]
    return torch.cat([prev2, prev1, x], dim=1)


class FrameEncoder(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        ch = [9, 32, 48, 64, 64]
        layers: list[nn.Module] = []
        for a, b in zip(ch[:-1], ch[1:], strict=True):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)  # 64 -> 4
        self.fc = nn.Linear(64 * 4 * 4, FEAT)
        self.head = nn.Linear(FEAT, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(self.conv(x).flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class FrameOnly(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEAT, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, f: torch.Tensor) -> list[torch.Tensor]:
        return [self.net(f)]


class CausalStage(nn.Module):
    """Dilated residual layers, left-padded so frame t sees only frames <= t."""

    def __init__(self, d_in: int, hidden: int, layers: int, n_classes: int) -> None:
        super().__init__()
        self.inp = nn.Conv1d(d_in, hidden, 1)
        self.dil = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, dilation=2**i) for i in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(hidden, hidden, 1) for _ in range(layers)])
        self.out = nn.Conv1d(hidden, n_classes, 1)
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D, T)
        h = self.inp(x)
        for i, (d, p) in enumerate(zip(self.dil, self.pw, strict=True)):
            pad = 2 * 2**i
            h = h + p(F.relu(d(F.pad(h, (pad, 0)))))
        return self.out(h)


class CausalTCN(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 64, layers: int = 8, stages: int = 2) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [CausalStage(FEAT, hidden, layers, n_classes)]
            + [CausalStage(n_classes, hidden, layers, n_classes) for _ in range(stages - 1)]
        )

    @property
    def receptive_field(self) -> int:
        first = self.stages[0]
        return 1 + sum(2 * 2**i for i in range(first.layers))

    def forward(self, f: torch.Tensor) -> list[torch.Tensor]:  # f: (B, T, D)
        x = f.transpose(1, 2)
        outs = []
        for i, s in enumerate(self.stages):
            x = s(x if i == 0 else F.softmax(x, dim=1))
            outs.append(x.transpose(1, 2))
        return outs


class OfflineTransformer(nn.Module):
    """Bidirectional; sinusoidal positions relative to the recording start."""

    def __init__(self, n_classes: int, d: int = 64, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.inp = nn.Linear(FEAT, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, n_classes)
        self.d = d

    def forward(self, f: torch.Tensor) -> list[torch.Tensor]:
        t = torch.arange(f.shape[1], device=f.device).float()[:, None]
        i = torch.arange(0, self.d, 2, device=f.device).float()[None, :]
        pe = torch.zeros(f.shape[1], self.d, device=f.device)
        pe[:, 0::2] = torch.sin(t / 10000 ** (i / self.d))
        pe[:, 1::2] = torch.cos(t / 10000 ** (i / self.d))
        return [self.out(self.enc(self.inp(f) + pe[None] / math.sqrt(self.d)))]


def segmentation_loss(outs: list[torch.Tensor], y: torch.Tensor, smooth: float) -> torch.Tensor:
    """Cross-entropy on every stage, plus MS-TCN's truncated MSE on log-probability changes."""
    loss = torch.zeros(())
    for o in outs:
        loss = loss + F.cross_entropy(o.reshape(-1, o.shape[-1]), y.reshape(-1))
        if smooth > 0 and o.shape[1] > 1:
            lp = F.log_softmax(o, dim=-1)
            d = (lp[:, 1:] - lp[:, :-1].detach()) ** 2
            loss = loss + smooth * torch.clamp(d, max=16).mean()
    return loss
