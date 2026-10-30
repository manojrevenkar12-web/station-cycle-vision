"""reports/results.json and docs/REPORT.md, generated; the held-out first run, recorded once."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .. import config

HELD = config.ROOT / "reports" / "heldout"
LABEL = {
    "train": "trained stations",
    "camera": "camera moved",
    "lighting": "lighting",
    "busy": "people walking past, dropped frames",
    "new_station": "new station (S5)",
    "heldout": "held-out station (blind)",
}


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def _s(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f} s"


def merge(profile: str) -> dict[str, Any]:
    run = config.ROOT / "runs" / profile
    r = json.loads((run / "results.json").read_text())
    e = run / "edge.json"
    if e.exists():
        r["edge"] = json.loads(e.read_text())
    return r


def record_first_run(r: dict[str, Any]) -> dict[str, Any] | None:
    if "heldout" not in r["domains"] or r["profile"] != "full":
        return None
    path = HELD / "first_run.json"
    if path.exists():
        return json.loads(path.read_text())
    first = {
        "recorded_at": r["generated_at"],
        "station_sha256": hashlib.sha256((config.ROOT / "heldout" / "station.yaml").read_bytes()).hexdigest(),
        "temporal": {m: v["heldout"] for m, v in r["temporal"].items()},
        "perception": r["perception"]["heldout"],
        "edge_live": r.get("edge", {}).get("live", {}).get("heldout"),
    }
    HELD.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(first, indent=1))
    lines = [
        f"| {m} | {_pct(v['accuracy'])} | {v['edit']:.1f} | {v['f1_50']:.1f} | {_pct(v['cycle_recall'])} | {_s(v['cycle_time_mae_s'])} |"
        for m, v in first["temporal"].items()
    ]
    (HELD / "FIRST_RUN.md").write_text(
        f"# Held-out station: first run\n\nRecorded {first['recorded_at']}; station file sha256 `{first['station_sha256']}`. "
        "Never rewritten. Written from `heldout/WRITING_STATIONS.md` by someone who had not seen the code, the other stations or "
        "any result.\n\n| Model | Frame accuracy | Edit | F1@50 | Cycles found | Cycle-time error |\n|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(lines)
        + "\n"
    )
    return first


def write_reports(profile: str = "full") -> dict[str, Any]:
    r = merge(profile)
    first = record_first_run(r)
    if first is not None:
        r["heldout_first_run"] = first
    out = config.ROOT / "reports" if profile == "full" else config.ROOT / "runs" / profile
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(r, indent=1))
    target = config.ROOT / "docs" / "REPORT.md" if profile == "full" else out / "REPORT.md"
    target.write_text(markdown(r))
    return r


def _table(rows: dict[str, dict[str, Any]], domains: list[str]) -> str:
    head = "| Model | Domain | Frame acc. | Edit | F1@50 | Cycles found | Cycle-time error | Deviations caught | Deviation precision |\n|---|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [head]
    for m, per in rows.items():
        for d in domains:
            v = per[d]
            lines.append(
                f"| {m} | {LABEL.get(d, d)} | {_pct(v['accuracy'])} | {v['edit']:.1f} | {v['f1_50']:.1f} | {_pct(v['cycle_recall'])} | "
                f"{_s(v['cycle_time_mae_s'])} | {_pct(v['deviation_recall'])} ({v['deviation_tp']}/{v['deviation_true']}) | {_pct(v['deviation_precision'])} |"
            )
    return "\n".join(lines)


def markdown(r: dict[str, Any]) -> str:
    t = r["temporal"]
    doms = r["domains"]
    tcn, plain, off, fo = t["causal-tcn"], t["causal-tcn (no augmentation)"], t["offline-transformer"], t["frame-only"]
    shifts = [d for d in ("camera", "lighting", "busy") if d in doms]

    def gain(a: dict, b: dict, k: str) -> str:
        return f"{100 * (a[k] - b[k]):+.1f} pts"

    q2 = "; ".join(
        f"{LABEL[d]}: frame accuracy {_pct(tcn[d]['accuracy'])} with augmentation, {_pct(plain[d]['accuracy'])} without ({gain(tcn[d], plain[d], 'accuracy')})"
        for d in shifts
    )
    q3 = (
        f"On the trained stations the live causal TCN reaches edit {tcn['train']['edit']:.1f} and F1@50 {tcn['train']['f1_50']:.1f}; "
        f"the offline transformer {off['train']['edit']:.1f} and {off['train']['f1_50']:.1f}; per-frame only {fo['train']['edit']:.1f} and {fo['train']['f1_50']:.1f}. "
        f"Cycle-time error: live {_s(tcn['train']['cycle_time_mae_s'])}, offline {_s(off['train']['cycle_time_mae_s'])}."
    )
    ob = r["onboarding"]
    q4 = (
        f"On the new station S5 the trained model reaches frame accuracy {_pct(ob['zero_shot']['accuracy'])} and finds {_pct(ob['zero_shot']['cycle_recall'])} of cycles "
        f"out of the box; after fine-tuning the temporal model on one labelled {r.get('seconds_per_recording', 120)}-second recording: "
        f"{_pct(ob['one_recording']['accuracy'])} and {_pct(ob['one_recording']['cycle_recall'])}."
    )
    p = r["perception"]
    prow = "\n".join(
        f"| {LABEL.get(d, d)} | {_pct(v['hand_recall'])} | {_pct(v['hand_precision'])} | {v['mota']:.3f} | {v['idf1']:.3f} | {v['id_switches_per_min']:.1f} | "
        f"{100 * v['leak_rate']:.2f}% | {_pct(v['over_redaction'])} |"
        for d, v in p.items()
    )
    e = r.get("edge")
    edge_md = ""
    if e:
        live = e["live"]
        lrow = "\n".join(
            f"| {LABEL.get(d, d)} | {_pct(v['int8_float_agreement'])} | {_pct(v['accuracy'])} | {_pct(v['cycle_recall'])} | {_s(v['cycle_time_mae_s'])} | "
            f"{_pct(v['deviation_recall'])} | {_pct(v['deviation_precision'])} |"
            for d, v in live.items()
        )
        edge_md = f"""
## On the camera box

ONNX Runtime only: an int8 encoder (static QDQ quantisation, per-channel) and the causal TCN as a
one-frame step with explicit state, matching the offline PyTorch model (tests/test_edge.py). Labels
are debounced over {e["debounce_frames"]} frames and turned into events by the same code the service
runs. Latency per frame, one core ({e["cpu"]}): int8 {e["latency"]["int8"]["median_ms"]:.2f} ms
median ({e["latency"]["int8"]["p99_ms"]:.2f} ms p99), float {e["latency"]["float"]["median_ms"]:.2f} ms;
at {e["fps"]} fps one core keeps up with {e["cameras_per_core_at_fps"]} cameras at this 64 x 64 resolution.

| Test set | int8 = float | Frame acc. | Cycles found | Cycle-time error | Deviations caught | Deviation precision |
|---|---:|---:|---:|---:|---:|---:|
{lrow}

Anonymisation on a sample recording through the ONNX detector: {100 * e["anonymisation_sample"]["leak_rate"]:.2f}% of person pixels left
visible, {_pct(e["anonymisation_sample"]["over_redaction"])} of bench pixels pixelated.
"""
    held = ""
    if "heldout_first_run" in r:
        f = r["heldout_first_run"]
        h = f["temporal"]["causal-tcn"]
        held = f"""
## Held-out station

Written blind from `heldout/WRITING_STATIONS.md`, run once ({f["recorded_at"]}), never rewritten
(`reports/heldout/first_run.json`). Causal TCN: frame accuracy {_pct(h["accuracy"])}, edit {h["edit"]:.1f},
cycles found {_pct(h["cycle_recall"])}, cycle-time error {_s(h["cycle_time_mae_s"])}, deviations caught
{_pct(h["deviation_recall"])}.
"""
    return f"""# Report

Generated by `scv report` ({r["generated_at"]}, profile `{r["profile"]}`). Do not edit.
{r["sizes"]["train_recordings"]} labelled recordings ({r["sizes"]["train_frames"]:,} frames at {r["sizes"]["fps"]} fps) from 4 stations;
test recordings per set: {r["sizes"]["test_recordings"]}. Encoder {r["model"]["encoder_parameters"]:,} parameters; causal TCN
{r["model"]["tcn_parameters"]:,} parameters, receptive field {r["model"]["tcn_receptive_field_frames"]} frames; detector
{r["model"]["detector_parameters"]:,} parameters.

## Q1 and Q2. Steps, cycles and deviations, on trained stations and under shift

{_table({"causal-tcn": tcn, "causal-tcn (no augmentation)": plain}, doms)}

Augmentation: {q2}.

## Q3. Live against offline

{q3}

{_table({"frame-only": fo, "offline-transformer": off}, ["train", "camera", "busy"])}

## Q4. A new station

{q4}

## Q5. Hands, tracks and anonymisation

| Test set | Hand recall | Hand precision | MOTA | IDF1 | ID switches / min | Person pixels leaked | Bench pixelated |
|---|---:|---:|---:|---:|---:|---:|---:|
{prow}
{edge_md}{held}"""
