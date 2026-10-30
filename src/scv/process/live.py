"""The process layer on a live stream: one frame label in, events out.

A label change is committed only after `hold` consecutive frames agree on the new label (a causal
debounce: it delays boundaries by `hold` frames and never revises the past). Committed segments
drive the same cycle rules as the offline layer: a cycle starts at pick_base, ends with unload,
and its step sequence is aligned with the standard work. `run_live` on a whole label sequence is
the reference the edge service must reproduce exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..sim.station import STEP_ID, STEPS
from .cycles import deviations

IDLE = STEP_ID["idle"]


@dataclass
class LiveProcess:
    standard: list[str]
    fps: int
    hold: int
    frame: int = 0
    current: int | None = None
    current_start: int = 0
    candidate: int | None = None
    candidate_start: int = 0
    candidate_count: int = 0
    cycle_steps: list[str] = field(default_factory=list)
    cycle_start: int | None = None

    def push(self, label: int) -> list[dict[str, Any]]:
        """Feed one frame's label; returns events committed at this frame."""
        events: list[dict[str, Any]] = []
        t = self.frame
        self.frame += 1
        if self.current is None:
            self.current, self.current_start = label, t
            return events
        if label == self.current:
            self.candidate, self.candidate_count = None, 0
            return events
        if label != self.candidate:
            self.candidate, self.candidate_start, self.candidate_count = label, t, 0
        self.candidate_count += 1
        if self.candidate_count < self.hold:
            return events
        # commit: the previous segment ended where the candidate started
        events += self._close(self.current, self.current_start, self.candidate_start)
        self.current, self.current_start = self.candidate, self.candidate_start
        self.candidate, self.candidate_count = None, 0
        events.append({"type": "step_started", "frame": self.current_start, "step": STEPS[self.current]})
        return events

    def _close(self, step: int, start: int, end: int) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = [{"type": "step_ended", "frame": end, "step": STEPS[step], "seconds": (end - start) / self.fps}]
        name = STEPS[step]
        if step == IDLE:
            return events
        if name == "pick_base":
            self.cycle_steps, self.cycle_start = [name], start
        elif self.cycle_start is not None:
            self.cycle_steps.append(name)
            if name == "unload":
                devs = deviations(self.cycle_steps, self.standard)
                events.append({
                    "type": "cycle_completed",
                    "start_frame": self.cycle_start,
                    "end_frame": end,
                    "cycle_time_s": (end - self.cycle_start) / self.fps,
                    "steps": list(self.cycle_steps),
                    "deviations": devs,
                })
                self.cycle_steps, self.cycle_start = [], None
        return events

    def flush(self) -> list[dict[str, Any]]:
        """End of stream: close the open segment."""
        if self.current is None:
            return []
        ev = self._close(self.current, self.current_start, self.frame)
        self.current = None
        return ev


def run_live(labels: np.ndarray, standard: list[str], fps: int, hold: int) -> list[dict[str, Any]]:
    lp = LiveProcess(standard, fps, hold)
    out: list[dict[str, Any]] = []
    for lab in labels:
        out += lp.push(int(lab))
    return out + lp.flush()
