"""`scv run`: data, encoders, temporal models, detector and tracker, evaluated on every test set.

Questions (docs/REPORT.md is generated from the answers):
  Q1 How well are work steps, cycles and deviations recognised on the stations it was trained on?
  Q2 What breaks when the camera, the light or the people around change, and does augmentation help?
  Q3 Live (causal, on the edge) against offline (bidirectional, in the cloud): what does waiting buy?
  Q4 A new station: how good is it out of the box, and after one labelled recording?
  Q5 Can hands be detected and tracked, and can people be anonymised before a clip leaves the station?
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import torch

from .. import config
from ..perception.anonymise import leak_metrics, redact_mask
from ..perception.detector import decode
from ..perception.tracker import Tracker, mot_metrics
from ..process.cycles import process
from ..sim.dataset import build
from ..sim.render import Episode
from ..sim.station import STEPS
from ..temporal.models import CausalTCN, FrameOnly, OfflineTransformer
from ..tracking import Tracking
from ..train.fit import detect_episode, encode, predict_temporal, train_detector, train_encoder, train_temporal
from . import metrics as M

K = len(STEPS)


def standard_of(ep: Episode) -> list[str]:
    if ep.station == "HELDOUT":
        held = config.heldout()
        assert held is not None
        return list(held[0].standard_work)
    return list(config.stations()[ep.station].standard_work)


def evaluate_labels(eps: list[Episode], preds: list[np.ndarray], min_seg: float) -> dict[str, Any]:
    seg = [M.segmentation(p, e.labels) for p, e in zip(preds, eps, strict=True)]
    proc = [M.process_metrics(process(p, standard_of(e), e.fps, min_seg), e.cycles, e.fps) for p, e in zip(preds, eps, strict=True)]
    # segmentation metrics after the same flicker removal the process layer uses
    return {
        **{k: float(np.mean([s[k] for s in seg])) for k in seg[0]},
        **M.summarise_process(proc),
    }


def perception_eval(det, eps: list[Episode], cfg: dict[str, Any]) -> dict[str, Any]:
    dcfg, tcfg = cfg["detector"], cfg["tracker"]
    hits = misses = false = 0
    mot = []
    leaks = []
    for ep in eps:
        heat, off, person = detect_episode(det, ep)
        tr = Tracker(tcfg["gate_px"], tcfg["max_missed"])
        hyp = []
        for t in range(len(ep.frames)):
            found = decode(heat[t, 0], off[t], 2, dcfg["peak_threshold"])
            pts = np.array([[x, y] for x, y, _ in found]).reshape(-1, 2)
            gt = ep.hands[t][ep.hand_visible[t]]
            if len(gt) and len(pts):
                d = np.linalg.norm(gt[:, None] - pts[None], axis=2)
                ok = (d.min(1) <= dcfg["match_px"]).sum()
            else:
                ok = 0
            hits += ok
            misses += len(gt) - ok
            false += max(0, len(pts) - ok)
            hyp.append(tr.step(t, pts))
        mot.append(mot_metrics(ep.hands, ep.hand_visible, hyp, dcfg["match_px"]))
        red = np.stack([redact_mask(person[t], cfg["anonymise"]["threshold"], cfg["anonymise"]["dilate_px"]) for t in range(len(person))])
        leaks.append(leak_metrics(ep.person, red))
    return {
        "hand_recall": hits / max(hits + misses, 1),
        "hand_precision": hits / max(hits + false, 1),
        "mota": float(np.mean([m["mota"] for m in mot])),
        "idf1": float(np.mean([m["idf1"] for m in mot])),
        "id_switches_per_min": float(sum(m["id_switches"] for m in mot) / max(1, sum(len(e.frames) for e in eps) / (eps[0].fps * 60))),
        "leak_rate": float(np.mean([x["leak_rate"] for x in leaks])),
        "over_redaction": float(np.mean([x["over_redaction"] for x in leaks])),
        "frames_leaking_over_5px": float(np.mean([x["frames_leaking_over_5px"] for x in leaks])),
    }


def run(profile: str = "full", log=print) -> dict[str, Any]:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    exp = config.load("experiment.yaml")
    svc = config.load("service.yaml")
    p = exp["profiles"][profile]
    out_dir = config.ROOT / "runs" / profile
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    trk = Tracking(profile)
    t0 = time.time()
    log(f"[data] rendering recordings ({profile})")
    data = build(profile)
    tests: dict[str, list[Episode]] = dict(data["test"])
    tests["new_station"] = data["onboard_test"]
    if data["heldout"]:
        tests["heldout"] = data["heldout"]
    min_seg = exp["process"]["min_segment_s"]

    log("[encoder] plain and augmented")
    enc = {}
    for name, aug in (("plain", False), ("augmented", True)):
        with trk.run(f"encoder-{name}", {"augment": aug, "epochs": p["encoder_epochs"], **{k: v for k, v in exp["encoder"].items() if k != "augment"}}):
            enc[name] = train_encoder(data["train"], exp["encoder"], p["encoder_epochs"], aug, seed=0)
    feats = {n: {id(e): encode(m, e) for e in data["train"] + data["val"] + data["onboard_train"] + [x for v in tests.values() for x in v]} for n, m in enc.items()}

    tc = exp["temporal"]
    specs = {
        "frame-only": ("augmented", lambda: FrameOnly(K)),
        "causal-tcn (no augmentation)": ("plain", lambda: CausalTCN(K, tc["hidden"], tc["layers"], tc["stages"])),
        "causal-tcn": ("augmented", lambda: CausalTCN(K, tc["hidden"], tc["layers"], tc["stages"])),
        "offline-transformer": ("augmented", lambda: OfflineTransformer(K, tc["hidden"], tc["transformer_layers"], tc["transformer_heads"])),
    }
    temporal: dict[str, torch.nn.Module] = {}
    for name, (encn, make) in specs.items():
        log(f"[temporal] {name}")
        seqs = [(feats[encn][id(e)], e.labels) for e in data["train"]]
        with trk.run(f"temporal-{name}", {"encoder": encn, "epochs": p["temporal_epochs"], **tc}):
            temporal[name] = train_temporal(make(), seqs, tc, p["temporal_epochs"], seed=0)

    log("[eval] segmentation, cycles, deviations")
    results: dict[str, Any] = {}
    preds_store: dict[str, dict[str, list[np.ndarray]]] = {}
    for name, (encn, _) in specs.items():
        results[name] = {}
        preds_store[name] = {}
        for d, eps in tests.items():
            preds = [predict_temporal(temporal[name], feats[encn][id(e)]).argmax(1) for e in eps]
            preds_store[name][d] = preds
            results[name][d] = evaluate_labels(eps, preds, min_seg)
        trk.log_metrics(f"temporal-{name}", {f"{d}_{k}": v for d, r in results[name].items() for k, v in r.items() if isinstance(v, float)})

    log("[onboard] one labelled recording of the new station")
    import copy

    tuned = copy.deepcopy(temporal["causal-tcn"])
    onboard_seqs = [(feats["augmented"][id(e)], e.labels) for e in data["onboard_train"]]
    tuned = train_temporal(tuned, onboard_seqs, {**tc, "lr": tc["lr"] / 4}, max(1, p["temporal_epochs"] // 3), seed=1)
    preds = [predict_temporal(tuned, feats["augmented"][id(e)]).argmax(1) for e in data["onboard_test"]]
    onboarding = {"zero_shot": results["causal-tcn"]["new_station"], "one_recording": evaluate_labels(data["onboard_test"], preds, min_seg)}

    log("[perception] detector, tracker, anonymiser")
    with trk.run("detector", {"epochs": p["detector_epochs"], **exp["detector"]}):
        det = train_detector(data["train"], exp["detector"], exp["encoder"], p["detector_epochs"], seed=0)
    pcfg = {"detector": exp["detector"], "tracker": exp["tracker"], "anonymise": svc["anonymise"]}
    perception = {d: perception_eval(det, eps, pcfg) for d, eps in tests.items()}

    # an example timeline for the dashboard: truth and live predictions on one recording per domain
    timelines = {}
    for d, eps in tests.items():
        e = eps[0]
        timelines[d] = {
            "station": e.station,
            "fps": e.fps,
            "truth": e.labels.tolist(),
            "live": preds_store["causal-tcn"][d][0].tolist(),
            "offline": preds_store["offline-transformer"][d][0].tolist(),
            "cycles": [{"start": c.start, "end": c.end, "deviations": c.deviations, "steps": c.steps} for c in e.cycles],
            "standard_work": standard_of(e),
            "found": [{"start": c.start / e.fps, "end": c.end / e.fps, "deviations": c.deviations, "steps": c.steps}
                      for c in process(preds_store["causal-tcn"][d][0], standard_of(e), e.fps, min_seg)],
        }

    torch.save(enc["augmented"].state_dict(), out_dir / "models" / "encoder.pt")
    torch.save(temporal["causal-tcn"].state_dict(), out_dir / "models" / "tcn.pt")
    torch.save(det.state_dict(), out_dir / "models" / "detector.pt")
    sample = data["test"]["train"][0]
    np.savez_compressed(out_dir / "sample.npz", frames=sample.frames, labels=sample.labels, person=sample.person)
    calib = np.concatenate([e.frames[::7] for e in data["val"]])
    np.savez_compressed(out_dir / "calibration.npz", frames=calib)

    res = {
        "profile": profile,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        "steps": STEPS,
        "domains": list(tests),
        "sizes": {
            "train_frames": int(sum(len(e.frames) for e in data["train"])),
            "train_recordings": len(data["train"]),
            "test_recordings": {d: len(v) for d, v in tests.items()},
            "fps": data["train"][0].fps,
        },
        "model": {
            "encoder_parameters": sum(t.numel() for t in enc["augmented"].parameters()),
            "tcn_parameters": sum(t.numel() for t in temporal["causal-tcn"].parameters()),
            "tcn_receptive_field_frames": temporal["causal-tcn"].receptive_field,
            "detector_parameters": sum(t.numel() for t in det.parameters()),
        },
        "temporal": results,
        "onboarding": onboarding,
        "perception": perception,
        "timelines": timelines,
        "mlflow": trk.summary(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(res, f, indent=1)
    log(f"[done] {res['seconds']} s")
    return res

