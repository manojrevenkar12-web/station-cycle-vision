"""From per-frame step labels to segments, cycles, cycle times and conformance to standard work.

This is the part a production engineer reads: a cycle starts when the worker picks a base plate
and ends when the plate leaves the fixture; its duration is the cycle time; its step sequence is
compared with the station's standard work to name what was missing, extra or out of order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.station import STEP_ID, STEPS

IDLE = STEP_ID["idle"]


@dataclass
class Seg:
    step: int
    start: int  # frame, inclusive
    end: int  # frame, exclusive


@dataclass
class FoundCycle:
    start: int
    end: int
    steps: list[str]
    deviations: list[str] = field(default_factory=list)

    def seconds(self, fps: int) -> float:
        return (self.end - self.start) / fps


def segments(labels: np.ndarray) -> list[Seg]:
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return [Seg(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def clean(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Merge segments shorter than `min_len` frames into the longer neighbour (flicker removal)."""
    out = labels.copy()
    changed = True
    while changed:
        changed = False
        segs = segments(out)
        for i, s in enumerate(segs):
            if s.end - s.start >= min_len or len(segs) == 1:
                continue
            left = segs[i - 1] if i > 0 else None
            right = segs[i + 1] if i + 1 < len(segs) else None
            nb = max((x for x in (left, right) if x is not None), key=lambda x: x.end - x.start)
            out[s.start : s.end] = nb.step
            changed = True
            break
    return out


def find_cycles(labels: np.ndarray) -> list[FoundCycle]:
    """A cycle runs from the start of a pick_base to the end of the next unload."""
    segs = [s for s in segments(labels) if s.step != IDLE]
    cycles: list[FoundCycle] = []
    cur: list[Seg] = []
    for s in segs:
        name = STEPS[s.step]
        if name == "pick_base":
            cur = [s]
        elif cur:
            cur.append(s)
            if name == "unload":
                cycles.append(FoundCycle(cur[0].start, s.end, [STEPS[x.step] for x in cur]))
                cur = []
    return cycles


def align(seq: list[str], standard: list[str]) -> tuple[int, list[tuple[str, str | None, str | None]]]:
    """Damerau-Levenshtein alignment of a cycle's steps with the standard work.

    Returns the distance and the operations: ('ok'|'sub'|'ins'|'del'|'swap', observed, expected).
    """
    n, m = len(seq), len(standard)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq[i - 1] == standard[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
            if i > 1 and j > 1 and seq[i - 1] == standard[j - 2] and seq[i - 2] == standard[j - 1] and seq[i - 1] != seq[i - 2]:
                d[i, j] = min(d[i, j], d[i - 2, j - 2] + 1)
    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 1 and j > 1 and seq[i - 1] == standard[j - 2] and seq[i - 2] == standard[j - 1] and seq[i - 1] != seq[i - 2] and d[i, j] == d[i - 2, j - 2] + 1:
            ops.append(("swap", seq[i - 1], standard[j - 1]))
            i, j = i - 2, j - 2
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (seq[i - 1] != standard[j - 1]):
            ops.append(("ok" if seq[i - 1] == standard[j - 1] else "sub", seq[i - 1], standard[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("ins", seq[i - 1], None))
            i -= 1
        else:
            ops.append(("del", None, standard[j - 1]))
            j -= 1
    return int(d[n, m]), ops[::-1]


def deviations(seq: list[str], standard: list[str]) -> list[str]:
    """Name what differs from standard work: missing, extra, order."""
    _, ops = align(seq, standard)
    out = set()
    for op, _obs, _exp in ops:
        if op == "del":
            out.add("missing")
        elif op == "ins":
            out.add("extra")
        elif op in ("swap", "sub"):
            out.add("order")
    return sorted(out)


def process(labels: np.ndarray, standard: list[str], fps: int, min_segment_s: float) -> list[FoundCycle]:
    cl = clean(labels, max(1, int(round(min_segment_s * fps))))
    cycles = find_cycles(cl)
    for c in cycles:
        c.deviations = deviations(c.steps, standard)
    return cycles
