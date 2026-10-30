"""Multi-object tracking of hands: constant-velocity Kalman filters and Hungarian assignment.

Identities matter: the left hand loads plates, the right hand fastens. A track keeps its id through
short occlusions (a colleague walking past) for up to `max_missed` frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    x: np.ndarray  # state [px, py, vx, vy]
    P: np.ndarray
    missed: int = 0
    hits: int = 1
    history: list[tuple[int, float, float]] = field(default_factory=list)


class Tracker:
    def __init__(self, gate_px: float, max_missed: int, q: float = 1.0, r: float = 1.0) -> None:
        self.gate = gate_px
        self.max_missed = max_missed
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = 1.0
        self.H = np.eye(2, 4)
        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r
        self.tracks: list[Track] = []
        self.next_id = 0

    def step(self, frame: int, detections: np.ndarray) -> list[tuple[int, float, float]]:
        """detections (N, 2) -> [(track id, x, y)] for tracks updated this frame."""
        for t in self.tracks:
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
        dets = np.asarray(detections, dtype=float).reshape(-1, 2)
        assigned: dict[int, int] = {}
        if self.tracks and len(dets):
            pred = np.array([t.x[:2] for t in self.tracks])
            cost = np.linalg.norm(pred[:, None, :] - dets[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r_, c_ in zip(rows, cols, strict=True):
                if cost[r_, c_] <= self.gate:
                    assigned[r_] = c_
        out = []
        used = set(assigned.values())
        for i, t in enumerate(self.tracks):
            if i in assigned:
                z = dets[assigned[i]]
                S = self.H @ t.P @ self.H.T + self.R
                K = t.P @ self.H.T @ np.linalg.inv(S)
                t.x = t.x + K @ (z - self.H @ t.x)
                t.P = (np.eye(4) - K @ self.H) @ t.P
                t.missed = 0
                t.hits += 1
                t.history.append((frame, float(t.x[0]), float(t.x[1])))
                out.append((t.id, float(t.x[0]), float(t.x[1])))
            else:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        for j, z in enumerate(dets):
            if j not in used:
                t = Track(self.next_id, np.array([z[0], z[1], 0.0, 0.0]), np.eye(4) * 10)
                t.history.append((frame, float(z[0]), float(z[1])))
                self.tracks.append(t)
                out.append((t.id, float(z[0]), float(z[1])))
                self.next_id += 1
        return out


def mot_metrics(
    truth: np.ndarray, truth_vis: np.ndarray, hyp: list[list[tuple[int, float, float]]], match_px: float
) -> dict[str, float]:
    """CLEAR MOT (MOTA, ID switches) and IDF1 for two ground-truth objects per frame.

    truth (T, G, 2), truth_vis (T, G), hyp[t] = [(id, x, y)].
    """
    T, G = truth_vis.shape
    fn = fp = idsw = tp = 0
    last: dict[int, int] = {}
    pairs: dict[tuple[int, int], int] = {}
    n_gt = int(truth_vis.sum())
    hyp_count: dict[int, int] = {}
    for t in range(T):
        gts = [g for g in range(G) if truth_vis[t, g]]
        hs = hyp[t]
        for hid, _, _ in hs:
            hyp_count[hid] = hyp_count.get(hid, 0) + 1
        if gts and hs:
            cost = np.array([[np.hypot(truth[t, g, 0] - x, truth[t, g, 1] - y) for (_, x, y) in hs] for g in gts])
            rows, cols = linear_sum_assignment(cost)
            matched = [(gts[r], hs[c][0]) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= match_px]
        else:
            matched = []
        tp += len(matched)
        fn += len(gts) - len(matched)
        fp += len(hs) - len(matched)
        for g, hid in matched:
            if g in last and last[g] != hid:
                idsw += 1
            last[g] = hid
            pairs[(g, hid)] = pairs.get((g, hid), 0) + 1
    # IDF1: best one-to-one mapping between truth objects and hypothesis ids
    gids = sorted({g for g, _ in pairs})
    hids = sorted({h for _, h in pairs})
    idtp = 0
    if gids and hids:
        m = np.array([[pairs.get((g, h), 0) for h in hids] for g in gids])
        r, c = linear_sum_assignment(-m)
        idtp = int(m[r, c].sum())
    n_hyp = sum(hyp_count.values())
    return {
        "mota": 1 - (fn + fp + idsw) / max(n_gt, 1),
        "idf1": 2 * idtp / max(n_gt + n_hyp, 1),
        "id_switches": idsw,
        "recall": tp / max(n_gt, 1),
        "precision": tp / max(tp + fp, 1),
    }
