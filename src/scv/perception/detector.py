"""One small network, three heads: hand centres, tool centre (heatmaps at stride 4, CenterNet style)
and a person mask at full resolution for anonymisation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SIZE = 64


class Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def block(a: int, b: int, s: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(a, b, 3, stride=s, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

        self.e1 = block(3, 24, 1)  # 64
        self.e2 = block(24, 32, 2)  # 32
        self.e3 = block(32, 48, 2)  # 16
        self.e4 = block(48, 48, 1)  # 16
        self.heat = nn.Conv2d(48, 2, 1)  # hand, tool
        self.off = nn.Conv2d(48, 2, 1)  # sub-cell offset (x, y) in cells
        self.up2 = block(48 + 32, 32, 1)
        self.up1 = block(32 + 24, 16, 1)
        self.mask = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.e1(x)
        b = self.e2(a)
        c = self.e4(self.e3(b))
        heat = self.heat(c)
        off = self.off(c)
        u = self.up2(torch.cat([F.interpolate(c, scale_factor=2, mode="nearest"), b], 1))
        u = self.up1(torch.cat([F.interpolate(u, scale_factor=2, mode="nearest"), a], 1))
        return heat, off, self.mask(u)[:, 0]


def gaussian_targets(points: np.ndarray, visible: np.ndarray, stride: int = 4, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """points (N, P, 2) image coords -> heatmap (N, H, W), offset (N, 2, H, W), offset mask (N, H, W)."""
    n = len(points)
    g = SIZE // stride
    heat = np.zeros((n, g, g), dtype=np.float32)
    off = np.zeros((n, 2, g, g), dtype=np.float32)
    om = np.zeros((n, g, g), dtype=np.float32)
    yy, xx = np.mgrid[0:g, 0:g]
    for i in range(n):
        for p, v in zip(points[i], visible[i], strict=True):
            if not v:
                continue
            c = p / stride
            cx, cy = int(np.clip(c[0], 0, g - 1)), int(np.clip(c[1], 0, g - 1))
            heat[i] = np.maximum(heat[i], np.exp(-((xx - c[0] + 0.5) ** 2 + (yy - c[1] + 0.5) ** 2) / (2 * sigma**2)))
            off[i, :, cy, cx] = c[0] - cx, c[1] - cy
            om[i, cy, cx] = 1
    return heat, off, om


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    neg = 1 - pos
    pl = -torch.log(p) * (1 - p) ** 2 * pos
    nl = -torch.log(1 - p) * p**2 * (1 - target) ** 4 * neg
    return (pl.sum() + nl.sum()) / pos.sum().clamp_min(1)


def decode(heat: np.ndarray, off: np.ndarray, k: int, threshold: float, stride: int = 4) -> list[tuple[float, float, float]]:
    """Peaks of one heatmap (H, W) (probabilities): up to k (x, y, score), 3x3 non-maximum suppression."""
    h = torch.from_numpy(heat)[None, None]
    peak = (F.max_pool2d(h, 3, 1, 1) == h)[0, 0].numpy() & (heat >= threshold)
    ys, xs = np.nonzero(peak)
    order = np.argsort(-heat[ys, xs])[:k]
    out = []
    for i in order:
        y, x = ys[i], xs[i]
        out.append(((x + off[0, y, x]) * stride, (y + off[1, y, x]) * stride, float(heat[y, x])))
    return out
