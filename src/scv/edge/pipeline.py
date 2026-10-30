"""`scv edge`: export the deployed models, check the edge path against PyTorch, and measure what the
camera box reports (live, debounced, int8) on every test set, plus latency and anonymisation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..evaluate import metrics as M
from ..evaluate.experiment import standard_of
from ..perception.anonymise import leak_metrics, redact_mask
from ..process.cycles import FoundCycle
from ..process.live import run_live
from ..sim.dataset import build
from ..sim.station import STEPS
from .export import EdgeModels, export_all


def live_cycles(labels: np.ndarray, standard: list[str], fps: int, hold: int) -> list[FoundCycle]:
    ev = run_live(labels, standard, fps, hold)
    return [FoundCycle(e["start_frame"], e["end_frame"], e["steps"], e["deviations"]) for e in ev if e["type"] == "cycle_completed"]


def run_edge(profile: str = "full", log=print) -> dict[str, Any]:
    run_dir = config.ROOT / "runs" / profile
    out = run_dir / "edge"
    log("[edge] export ONNX (float, int8) and the streaming TCN")
    manifest = export_all(run_dir, out, len(STEPS))
    exp = config.load("experiment.yaml")
    svc = config.load("service.yaml")
    fps = int(config.load("stations.yaml")["fps"])
    hold = max(1, int(round(exp["process"]["min_segment_s"] * fps)))
    data = build(profile)
    tests = dict(data["test"])
    tests["new_station"] = data["onboard_test"]
    if data["heldout"]:
        tests["heldout"] = data["heldout"]

    log("[edge] float vs int8, live process on every test set")
    em = {"float": EdgeModels(out, int8=False), "int8": EdgeModels(out, int8=True)}
    per: dict[str, Any] = {}
    for d, eps in tests.items():
        row: dict[str, Any] = {}
        labs: dict[str, list[np.ndarray]] = {}
        for name, m in em.items():
            labs[name] = []
            for e in eps:
                s = m.new_state()
                labs[name].append(np.array([int(np.argmax(m.step(f, s))) for f in e.frames]))
        agree = float(np.mean(np.concatenate([a == b for a, b in zip(labs["float"], labs["int8"], strict=True)])))
        seg = [M.segmentation(p, e.labels) for p, e in zip(labs["int8"], eps, strict=True)]
        proc = [M.process_metrics(live_cycles(p, standard_of(e), fps, hold), e.cycles, fps) for p, e in zip(labs["int8"], eps, strict=True)]
        row = {"int8_float_agreement": agree, **{k: float(np.mean([x[k] for x in seg])) for k in seg[0]}, **M.summarise_process(proc)}
        per[d] = row

    log("[edge] latency and anonymised clips")
    sample = np.load(run_dir / "sample.npz")
    lat = {name: m.latency(sample["frames"], n=600) for name, m in em.items()}
    a = svc["anonymise"]
    probs = em["int8"].person_mask(sample["frames"])
    red = np.stack([redact_mask(p, a["threshold"], a["dilate_px"]) for p in probs])
    anon = leak_metrics(sample["person"], red)

    res = {
        "manifest": manifest,
        "latency": lat,
        "cameras_per_core_at_fps": int(lat["int8"]["frames_per_s_per_core"] // fps),
        "fps": fps,
        "debounce_frames": hold,
        "live": per,
        "anonymisation_sample": anon,
        "cpu": _cpu(),
    }
    with open(run_dir / "edge.json", "w") as f:
        json.dump(res, f, indent=1)
    if profile == "full":
        dst = config.ROOT / "models"
        dst.mkdir(exist_ok=True)
        for p in out.glob("*"):
            if p.suffix in (".onnx", ".json"):
                shutil.copy(p, dst / p.name)
    log(f"[edge] int8 latency {lat['int8']['median_ms']:.2f} ms/frame; train-domain live accuracy {per['train']['accuracy']:.3f}")
    return res


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"
