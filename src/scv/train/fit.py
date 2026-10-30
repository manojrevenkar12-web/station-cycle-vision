"""Training loops: frame encoder, temporal models, detector. Fixed schedules; no test data is seen."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..perception.detector import Detector, focal_loss, gaussian_targets
from ..sim.render import Episode
from ..sim.station import STEPS
from ..temporal.models import FrameEncoder, segmentation_loss, stack_frames

K = len(STEPS)


def _cosine(step: int, total: int) -> float:
    return 0.5 * (1 + math.cos(math.pi * min(step, total) / max(total, 1)))


def augment_batch(x: torch.Tensor, spec: dict[str, Any], g: torch.Generator) -> torch.Tensor:
    """Same random camera and light for all channels of one sample (the stacked frames move together)."""
    n = x.shape[0]
    rot = (torch.rand(n, generator=g) * 2 - 1) * math.radians(spec["rotation_deg"])
    lo, hi = spec["scale"]
    sc = lo + torch.rand(n, generator=g) * (hi - lo)
    sh = (torch.rand(n, 2, generator=g) * 2 - 1) * spec["shift"] / 32.0
    cos, sin = torch.cos(rot) / sc, torch.sin(rot) / sc
    theta = torch.stack([torch.stack([cos, -sin, sh[:, 0]], 1), torch.stack([sin, cos, sh[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    glo, ghi = spec["gain"]
    gain = glo + torch.rand(n, 1, 1, 1, generator=g) * (ghi - glo)
    tint = 1 + (torch.rand(n, 3, 1, 1, generator=g) * 2 - 1) * spec["tint"]
    x = x * gain * tint.repeat(1, x.shape[1] // 3, 1, 1)
    x = x + torch.randn(x.shape, generator=g) * spec["noise"] * torch.rand(n, 1, 1, 1, generator=g)
    return x.clamp(0, 1)


def _stack_index(eps: list[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All frames as uint8 plus, for each frame, the index of t-1 and t-2 within its own recording."""
    frames = np.concatenate([e.frames for e in eps])
    starts = np.concatenate(
        [np.full(len(e.frames), s) for e, s in zip(eps, np.cumsum([0] + [len(e.frames) for e in eps[:-1]]), strict=True)]
    )
    i = np.arange(len(frames))
    return frames, i, np.maximum(i - 1, starts), np.maximum(i - 2, starts)


def _batch(frames: np.ndarray, i0: np.ndarray, i1: np.ndarray, i2: np.ndarray) -> torch.Tensor:
    def t(ix: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frames[ix]).permute(0, 3, 1, 2).float() / 255.0

    return torch.cat([t(i2), t(i1), t(i0)], dim=1)


def train_encoder(eps: list[Episode], cfg: dict[str, Any], epochs: int, augment: bool, seed: int) -> FrameEncoder:
    """Frames stay uint8 in memory; each batch is stacked and converted on the fly."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    frames, i0, i1, i2 = _stack_index(eps)
    y = torch.from_numpy(np.concatenate([e.labels for e in eps]))
    model = FrameEncoder(K)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs = cfg["batch"]
    steps = max(1, epochs * len(frames) // bs)
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(frames), bs)
        xb = _batch(frames, i0[idx], i1[idx], i2[idx])
        if augment:
            xb = augment_batch(xb, cfg["augment"], g)
        loss = F.cross_entropy(model(xb), y[idx])
        opt.param_groups[0]["lr"] = cfg["lr"] * _cosine(step, steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def encode(model: FrameEncoder, ep: Episode, batch: int = 512) -> np.ndarray:
    model.eval()
    x = stack_frames(torch.from_numpy(ep.frames))
    return torch.cat([model.features(x[i : i + batch]) for i in range(0, len(x), batch)]).numpy()


def train_temporal(
    model: torch.nn.Module,
    seqs: list[tuple[np.ndarray, np.ndarray]],
    cfg: dict[str, Any],
    epochs: int,
    seed: int,
    crop: int = 400,
) -> torch.nn.Module:
    """Whole recordings as sequences; a random crop per step keeps batches small and varied."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = max(1, epochs * len(seqs))
    model.train()
    for step in range(steps):
        f, y = seqs[rng.integers(len(seqs))]
        n = min(crop, len(f))
        s = int(rng.integers(0, len(f) - n + 1))
        fb = torch.from_numpy(f[s : s + n])[None]
        yb = torch.from_numpy(y[s : s + n])[None]
        loss = segmentation_loss(model(fb), yb, cfg["smooth_weight"])
        opt.param_groups[0]["lr"] = cfg["lr"] * _cosine(step, steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_temporal(model: torch.nn.Module, f: np.ndarray) -> np.ndarray:
    """(T, D) -> (T, K) logits of the last stage."""
    model.eval()
    return model(torch.from_numpy(f)[None])[-1][0].numpy()


def train_detector(eps: list[Episode], cfg: dict[str, Any], enc_cfg: dict[str, Any], epochs: int, seed: int) -> Detector:
    """Targets are built per batch from the truth; frames stay uint8 in memory."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    stride = int(cfg.get("frame_stride", 1))
    frames = np.concatenate([e.frames[::stride] for e in eps])
    hands = np.concatenate([e.hands[::stride] for e in eps])
    vis = np.concatenate([e.hand_visible[::stride] for e in eps])
    tool = np.concatenate([e.tool[::stride] for e in eps])[:, None, :]
    tvis = np.all((tool >= 0) & (tool < 64), axis=2)
    person = np.concatenate([e.person[::stride] for e in eps])
    model = Detector()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    bs = cfg["batch"]
    steps = max(1, epochs * len(frames) // bs)
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(frames), bs)
        hh, ho, hm = gaussian_targets(hands[idx], vis[idx])
        th, to, tm = gaussian_targets(tool[idx], tvis[idx])
        heat = torch.from_numpy(np.stack([hh, th], 1))
        off = torch.from_numpy(np.stack([ho, to], 1))
        om = torch.from_numpy(np.stack([hm, tm], 1))
        xb = torch.from_numpy(frames[idx]).permute(0, 3, 1, 2).float() / 255.0
        # photometric augmentation only: the targets are in image coordinates
        gain = 0.7 + torch.rand(bs, 1, 1, 1, generator=g) * 0.6
        xb = (xb * gain + torch.randn(xb.shape, generator=g) * 0.03).clamp(0, 1)
        ph, po, pm = model(xb)
        loss = focal_loss(ph, heat)
        pred_off = po[:, None].expand(-1, 2, -1, -1, -1)
        mask = om[:, :, None]
        loss = loss + F.l1_loss(pred_off * mask, off * mask, reduction="sum") / mask.sum().clamp_min(1)
        # the person mask is weighted towards recall: missing a person is worse than masking bench
        loss = loss + F.binary_cross_entropy_with_logits(pm, torch.from_numpy(person[idx]).float(), pos_weight=torch.tensor(4.0))
        opt.param_groups[0]["lr"] = cfg["lr"] * _cosine(step, steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def detect_episode(model: Detector, ep: Episode, batch: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> heat probs (T, 2, 16, 16), offsets (T, 2, 16, 16), person probs (T, 64, 64)."""
    x = torch.from_numpy(ep.frames).permute(0, 3, 1, 2).float() / 255.0
    hs, os_, ms = [], [], []
    for i in range(0, len(x), batch):
        h, o, m = model(x[i : i + batch])
        hs.append(torch.sigmoid(h).numpy())
        os_.append(o.numpy())
        ms.append(torch.sigmoid(m).numpy())
    return np.concatenate(hs), np.concatenate(os_), np.concatenate(ms)
