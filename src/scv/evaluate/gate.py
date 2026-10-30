"""`scv gate`: floors (enforced) and targets (reported) from config/release.yaml."""

from __future__ import annotations

import json
from typing import Any

from .. import config


def measured(r: dict[str, Any]) -> dict[str, float]:
    live = r["edge"]["live"]["train"]
    return {
        "cycle_recall": live["cycle_recall"],
        "cycle_time_mae_s": live["cycle_time_mae_s"],
        "deviation_recall": live["deviation_recall"],
        "deviation_precision": live["deviation_precision"],
        "leak_rate": r["perception"]["train"]["leak_rate"],
        "int8_agreement": live["int8_float_agreement"],
        "edge_ms_per_frame": r["edge"]["latency"]["int8"]["median_ms"],
    }


LOWER_IS_BETTER = {"cycle_time_mae_s", "leak_rate", "edge_ms_per_frame"}


def gate(profile: str = "full") -> tuple[bool, list[str]]:
    path = config.ROOT / "reports" / "results.json" if profile == "full" else config.ROOT / "runs" / profile / "results.json"
    r = json.loads(path.read_text())
    if "edge" not in r:
        return False, ["FAIL  edge results missing: run `scv edge` then `scv report`"]
    rel = config.load("release.yaml")
    m = measured(r)
    ok = True
    lines = []

    def good(k: str, v: float | None, thr: float) -> bool:
        if v is None:
            return False
        return v <= thr if k in LOWER_IS_BETTER else v >= thr

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.4g}"

    for k, thr in (rel.get("floors") or {}).items():
        g = good(k, m[k], thr)
        ok &= g
        lines.append(f"{'pass' if g else 'FAIL'}  floor  {k:22s} {fmt(m[k])} ({'<=' if k in LOWER_IS_BETTER else '>='} {thr})")
    for k, thr in rel["targets"].items():
        g = good(k, m[k], thr)
        lines.append(
            f"{'met ' if g else 'miss'}  target {k:22s} {fmt(m[k])} ({'<=' if k in LOWER_IS_BETTER else '>='} {thr}, reported, not gated)"
        )
    return ok, lines
