"""One recording: a station, a recording condition, a worker, some minutes of video, and the truth."""

from __future__ import annotations

from typing import Any

import numpy as np

from .render import Episode, render_episode
from .station import Station, schedule


def make_episode(station: Station, dom_name: str, dom: dict[str, Any], cfg: dict[str, Any], seconds: float, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    segs, cycles, speed = schedule(station, cfg, seconds, rng)
    return render_episode(station, segs, cycles, seconds, cfg["fps"], dom_name, dom, rng, speed)
