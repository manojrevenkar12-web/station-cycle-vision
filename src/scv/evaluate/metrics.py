"""Temporal action segmentation metrics (frame accuracy, edit score, F1@k as in MS-TCN) and the
process metrics a plant cares about: cycles found, cycle-time error, deviations caught."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..process.cycles import FoundCycle, segments
from ..sim.station import Cycle


def frame_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))


def edit_score(pred: np.ndarray, y: np.ndarray) -> float:
    """100 * (1 - normalised Levenshtein distance between segment label sequences)."""
    a = [s.step for s in segments(pred)]
    b = [s.step for s in segments(y)]
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return float(100 * (1 - d[n, m] / max(n, m, 1)))


def f1_at(pred: np.ndarray, y: np.ndarray, overlap: float) -> float:
    """Segmental F1 at an IoU threshold (Lea et al., 2017)."""
    ps, ys = segments(pred), segments(y)
    used = np.zeros(len(ys), dtype=bool)
    tp = 0
    for p in ps:
        best, bi = 0.0, -1
        for i, g in enumerate(ys):
            if g.step != p.step or used[i]:
                continue
            inter = max(0, min(p.end, g.end) - max(p.start, g.start))
            union = max(p.end, g.end) - min(p.start, g.start)
            iou = inter / union
            if iou > best:
                best, bi = iou, i
        if bi >= 0 and best >= overlap:
            tp += 1
            used[bi] = True
    fp = len(ps) - tp
    fn = len(ys) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return float(100 * 2 * prec * rec / max(prec + rec, 1e-9))


def segmentation(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": frame_accuracy(pred, y),
        "edit": edit_score(pred, y),
        "f1_10": f1_at(pred, y, 0.10),
        "f1_25": f1_at(pred, y, 0.25),
        "f1_50": f1_at(pred, y, 0.50),
    }


def match_cycles(found: list[FoundCycle], truth: list[Cycle], fps: int, min_iou: float = 0.5) -> list[tuple[Cycle, FoundCycle]]:
    pairs = []
    used: set[int] = set()
    for t in truth:
        ts, te = t.start * fps, t.end * fps
        best, bi = 0.0, -1
        for i, f in enumerate(found):
            if i in used:
                continue
            inter = max(0.0, min(te, f.end) - max(ts, f.start))
            union = max(te, f.end) - min(ts, f.start)
            if inter / union > best:
                best, bi = inter / union, i
        if bi >= 0 and best >= min_iou:
            used.add(bi)
            pairs.append((t, found[bi]))
    return pairs


def process_metrics(found: list[FoundCycle], truth: list[Cycle], fps: int) -> dict[str, Any]:
    pairs = match_cycles(found, truth, fps)
    err = [abs(f.seconds(fps) - (t.end - t.start)) for t, f in pairs]
    td = [bool(t.deviations) for t, _ in pairs]
    fd = [bool(f.deviations) for _, f in pairs]
    tp = sum(1 for a, b in zip(td, fd, strict=True) if a and b)
    typed = sum(1 for t, f in pairs if t.deviations and f.deviations and set(t.deviations) == set(f.deviations))
    return {
        "cycles_true": len(truth),
        "cycles_found": len(found),
        "cycles_matched": len(pairs),
        "cycle_time_mae_s": float(np.mean(err)) if err else float("nan"),
        "cycle_time_p90_err_s": float(np.percentile(err, 90)) if err else float("nan"),
        "deviation_true": int(sum(td)),
        "deviation_flagged": int(sum(fd)),
        "deviation_tp": int(tp),
        "deviation_typed_right": int(typed),
    }


def summarise_process(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Pool episode-level process metrics into rates."""
    s = {k: sum(r[k] for r in rows) for k in ("cycles_true", "cycles_found", "cycles_matched", "deviation_true", "deviation_flagged", "deviation_tp", "deviation_typed_right")}
    maes = [r["cycle_time_mae_s"] * r["cycles_matched"] for r in rows if r["cycles_matched"]]
    return {
        "cycle_recall": s["cycles_matched"] / max(s["cycles_true"], 1),
        "cycle_precision": s["cycles_matched"] / max(s["cycles_found"], 1),
        "cycle_time_mae_s": sum(maes) / max(s["cycles_matched"], 1),
        "deviation_recall": s["deviation_tp"] / max(s["deviation_true"], 1),
        "deviation_precision": s["deviation_tp"] / max(s["deviation_flagged"], 1),
        "deviation_type_accuracy": s["deviation_typed_right"] / max(s["deviation_tp"], 1),
        **{k: int(v) for k, v in s.items()},
    }
