"""The recordings an experiment uses, by split. Deterministic in (profile, seed offsets)."""

from __future__ import annotations

from typing import Any

from .. import config
from .render import Episode
from .video import make_episode


def build(profile: str) -> dict[str, Any]:
    exp = config.load("experiment.yaml")
    p = exp["profiles"][profile]
    seconds = p.get("seconds", exp["seconds"])
    cfg = config.load("stations.yaml")
    st = config.stations()
    dm = config.domains()
    out: dict[str, Any] = {"train": [], "val": [], "test": {}, "onboard_train": [], "onboard_test": [], "heldout": []}
    for i, s in enumerate(exp["train_stations"]):
        for e in range(p["train_episodes"]):
            out["train"].append(make_episode(st[s], "train", dm["train"], cfg, seconds, 1_000 * i + e))
        for e in range(p["val_episodes"]):
            out["val"].append(make_episode(st[s], "train", dm["train"], cfg, seconds, 50_000 + 1_000 * i + e))
    for j, d in enumerate(exp["test_domains"]):
        eps: list[Episode] = []
        for i, s in enumerate(exp["train_stations"]):
            for e in range(p["test_episodes"]):
                eps.append(make_episode(st[s], d, dm[d], cfg, seconds, 100_000 + 10_000 * j + 1_000 * i + e))
        out["test"][d] = eps
    new = st[exp["onboarding_station"]]
    out["onboard_train"].append(make_episode(new, "train", dm["train"], cfg, seconds, 200_000))
    for e in range(p["onboarding_episodes"]):
        out["onboard_test"].append(make_episode(new, "train", dm["train"], cfg, seconds, 210_000 + e))
    held = config.heldout()
    if held is not None:
        hs, hd = held
        for e in range(p["heldout_episodes"]):
            out["heldout"].append(make_episode(hs, "heldout", hd, cfg, seconds, 300_000 + e))
    return out
