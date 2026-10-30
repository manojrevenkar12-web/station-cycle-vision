"""Settings live in config/*.yaml; code reads them here."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .sim.station import Station

ROOT = Path(os.environ.get("SCV_ROOT", Path(__file__).resolve().parents[2]))


@cache
def load(name: str) -> dict[str, Any]:
    with open(ROOT / "config" / name) as f:
        return yaml.safe_load(f)


def stations() -> dict[str, Station]:
    return {k: Station.from_dict(k, v) for k, v in load("stations.yaml")["stations"].items()}


def domains() -> dict[str, dict[str, Any]]:
    return load("domains.yaml")


def heldout() -> tuple[Station, dict[str, Any]] | None:
    """The station and recording conditions written blind (heldout/station.yaml), if present."""
    p = ROOT / "heldout" / "station.yaml"
    if not p.exists():
        return None
    with open(p) as f:
        d = yaml.safe_load(f)
    base = dict(load("domains.yaml")["train"])
    unknown = set(d.get("conditions", {})) - set(base)
    if unknown:
        raise ValueError(f"heldout conditions: unknown keys {sorted(unknown)}")
    base.update(d.get("conditions", {}))
    return Station.from_dict("HELDOUT", d["station"]), base
