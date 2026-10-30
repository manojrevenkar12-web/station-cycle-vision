"""Anonymisation before any clip leaves the station: pixelate everything the mask head calls a person.

The threshold is low and the mask is dilated on purpose: a missed pixel of a person is worse than a
pixelated pixel of bench. Leak rate (person pixels left visible) is gated in config/release.yaml.
"""

from __future__ import annotations

import cv2
import numpy as np


def redact_mask(prob: np.ndarray, threshold: float, dilate_px: int) -> np.ndarray:
    m = (prob >= threshold).astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        m = cv2.dilate(m, k)
    return m.astype(bool)


def pixelate(frame: np.ndarray, mask: np.ndarray, block: int = 8) -> np.ndarray:
    h, w = mask.shape
    small = cv2.resize(frame, (w // block, h // block), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    out[mask] = coarse[mask]
    return out


def leak_metrics(truth: np.ndarray, redacted: np.ndarray) -> dict[str, float]:
    """truth, redacted: (T, H, W) bool. Leak = person pixels not redacted; cost = bench pixels redacted."""
    person = truth.sum()
    leaked = (truth & ~redacted).sum()
    bench = (~truth).sum()
    frames_with_person = truth.reshape(len(truth), -1).any(1)
    frame_leak = (truth & ~redacted).reshape(len(truth), -1).sum(1)
    return {
        "leak_rate": float(leaked / max(person, 1)),
        "over_redaction": float((redacted & ~truth).sum() / max(bench, 1)),
        "frames_leaking_over_5px": float(np.mean(frame_leak[frames_with_person] > 5)) if frames_with_person.any() else 0.0,
    }
