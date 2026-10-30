"""Stations, standard work, and what a worker actually does: a timed sequence of steps with deviations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

STEPS = ["idle", "pick_base", "place_base", "pick_screw", "fasten", "inspect", "unload"]
STEP_ID = {s: i for i, s in enumerate(STEPS)}
DEVIATIONS = ["missing", "order", "extra"]
LAYOUT_KEYS = ["base_bin", "screw_bin", "fixture", "output", "tool_rest", "rest_left", "rest_right"]


@dataclass(frozen=True)
class Station:
    name: str
    layout: dict[str, tuple[float, float]]
    standard_work: tuple[str, ...]
    bench: tuple[float, float, float]

    @staticmethod
    def from_dict(name: str, d: dict[str, Any]) -> Station:
        unknown = set(d) - {"layout", "standard_work", "bench"}
        if unknown:
            raise ValueError(f"station {name}: unknown keys {sorted(unknown)}")
        layout = d["layout"]
        missing = set(LAYOUT_KEYS) - set(layout)
        if missing:
            raise ValueError(f"station {name}: layout lacks {sorted(missing)}")
        for k, v in layout.items():
            if k not in LAYOUT_KEYS or len(v) != 2 or not all(4 <= c <= 92 for c in v):
                raise ValueError(f"station {name}: {k} must be [x, y] inside the 96 x 96 bench")
        sw = tuple(d["standard_work"])
        bad = [s for s in sw if s not in STEPS or s == "idle"]
        if bad or not sw:
            raise ValueError(f"station {name}: unknown steps {bad}")
        if sw[0] != "pick_base" or sw[-1] != "unload":
            raise ValueError(f"station {name}: a cycle starts with pick_base and ends with unload")
        for i, s in enumerate(sw):
            if s == "fasten" and (i == 0 or sw[i - 1] != "pick_screw"):
                raise ValueError(f"station {name}: every fasten follows a pick_screw")
        bench = tuple(float(c) for c in d.get("bench", [0.5, 0.45, 0.4]))
        return Station(name, {k: (float(v[0]), float(v[1])) for k, v in layout.items()}, sw, bench)  # type: ignore[arg-type]

    @property
    def screws(self) -> int:
        return sum(1 for s in self.standard_work if s == "fasten")


@dataclass
class Segment:
    step: str
    start: float  # seconds
    end: float
    cycle: int  # -1 between cycles
    screw: int = -1  # which screw slot a fastening is for


@dataclass
class Cycle:
    index: int
    start: float
    end: float
    steps: list[str]
    deviations: list[str] = field(default_factory=list)


def plan_cycle(sw: tuple[str, ...], dev_cfg: dict[str, float], rng: np.random.Generator) -> tuple[list[str], list[str], bool]:
    """The steps a worker performs in one cycle, the deviations in it, and whether they pause."""
    steps = list(sw)
    devs: list[str] = []
    fasten_idx = [i for i, s in enumerate(steps) if s == "fasten"]
    if fasten_idx and rng.uniform() < dev_cfg["missing"] and len(fasten_idx) > 1:
        i = int(rng.choice(fasten_idx))
        del steps[i - 1 : i + 1]
        devs.append("missing")
    if rng.uniform() < dev_cfg["order"]:
        # swap two adjacent steps that differ, never moving pick_base or unload
        cand = [i for i in range(1, len(steps) - 2) if steps[i] != steps[i + 1] and not (steps[i] == "pick_screw" and steps[i + 1] == "fasten")]
        if cand:
            i = int(rng.choice(cand))
            steps[i], steps[i + 1] = steps[i + 1], steps[i]
            devs.append("order")
    if rng.uniform() < dev_cfg["extra"]:
        fi = [i for i, s in enumerate(steps) if s == "fasten"]
        if fi:
            i = int(rng.choice(fi))
            steps[i + 1 : i + 1] = ["pick_screw", "fasten"]
            devs.append("extra")
    pause = bool(rng.uniform() < dev_cfg["pause"])
    return steps, devs, pause


def schedule(station: Station, cfg: dict[str, Any], seconds: float, rng: np.random.Generator) -> tuple[list[Segment], list[Cycle], float]:
    """Segments covering [0, seconds) and the cycles completed in them. Returns the worker speed too."""
    durs = cfg["durations"]
    speed = float(rng.uniform(*cfg["worker_speed"]))
    jitter = cfg["duration_jitter"]
    segs: list[Segment] = []
    cycles: list[Cycle] = []
    t = float(rng.uniform(0, 2.0))
    if t > 0:
        segs.append(Segment("idle", 0.0, t, -1))
    k = 0
    while t < seconds:
        steps, devs, pause = plan_cycle(station.standard_work, cfg["deviations"], rng)
        start = t
        pause_at = int(rng.integers(1, len(steps))) if pause else -1
        screw = 0
        for i, s in enumerate(steps):
            if i == pause_at:
                d = float(rng.uniform(3, 8))
                segs.append(Segment("idle", t, t + d, k))
                t += d
            d = durs[s] * speed * float(np.exp(rng.normal(0, jitter)))
            slot = -1
            if s == "fasten":
                slot = screw % max(1, station.screws)
                screw += 1
            segs.append(Segment(s, t, t + d, k, slot))
            t += d
        cycles.append(Cycle(k, start, t, steps, devs))
        k += 1
        gap = float(rng.uniform(*durs["idle_between_cycles"]))
        segs.append(Segment("idle", t, t + gap, -1))
        t += gap
    # only cycles that finished inside the recording count
    cycles = [c for c in cycles if c.end <= seconds]
    return segs, cycles, speed
