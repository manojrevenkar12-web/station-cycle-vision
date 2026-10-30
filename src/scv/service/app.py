"""The edge service: one process on the camera box, ONNX Runtime only.

  POST /v1/stations/{station}/frames   raw uint8 frames (N x 64 x 64 x 3) -> step and cycle events
  GET  /v1/stations/{station}/cycles   completed cycles with cycle time and deviations
  GET  /v1/stations/{station}/events   everything committed, oldest first, after `after_id`
  GET  /v1/clips/{clip_id}             an anonymised clip (npz) kept for a cycle that deviated
  POST /v1/stations/{station}/reset    start a fresh stream (camera restarted)
  GET  /healthz /readyz /metrics

Video never leaves the box except as anonymised clips around deviations; the default store is a
SQLite file on the box. Labels are debounced exactly as `scv.process.live.run_live` does, which the
tests use as the reference.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .. import config
from ..edge.export import EdgeModels
from ..perception.anonymise import pixelate, redact_mask
from ..process.live import LiveProcess

FRAME_BYTES = 64 * 64 * 3


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, station TEXT NOT NULL, stream TEXT NOT NULL,"
                " type TEXT NOT NULL, frame INTEGER NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            self.db.execute("CREATE INDEX IF NOT EXISTS events_station ON events(station, id)")
            self.db.commit()

    def add(self, station: str, stream: str, events: list[dict[str, Any]]) -> list[int]:
        ids = []
        with self.lock:
            for e in events:
                cur = self.db.execute(
                    "INSERT INTO events (station, stream, type, frame, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (station, stream, e["type"], int(e.get("frame", e.get("end_frame", 0))), json.dumps(e), time.time()),
                )
                ids.append(int(cur.lastrowid))
            self.db.commit()
        return ids

    def query(self, station: str, type_: str | None, after_id: int, limit: int) -> list[dict[str, Any]]:
        sql = "SELECT id, stream, payload FROM events WHERE station = ? AND id > ?"
        args: list[Any] = [station, after_id]
        if type_:
            sql += " AND type = ?"
            args.append(type_)
        sql += " ORDER BY id LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
        return [{"id": r[0], "stream": r[1], **json.loads(r[2])} for r in rows]


class Session:
    def __init__(self, models: EdgeModels, standard: list[str], fps: int, hold: int, keep: int) -> None:
        self.state = models.new_state()
        self.process = LiveProcess(standard, fps, hold)
        self.frames: deque[np.ndarray] = deque(maxlen=keep)
        self.stream = uuid.uuid4().hex[:12]
        self.lock = threading.Lock()


def create_app(model_dir: Path | None = None, db_path: Path | None = None, clips_dir: Path | None = None) -> FastAPI:
    svc = config.load("service.yaml")
    st_cfg = config.load("stations.yaml")
    exp = config.load("experiment.yaml")
    fps = int(st_cfg["fps"])
    hold = max(1, int(round(exp["process"]["min_segment_s"] * fps)))
    model_dir = model_dir or config.ROOT / "models"
    db_path = db_path or config.ROOT / svc["events"]["database"]
    clips_dir = clips_dir or db_path.parent / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    stations = {k: list(v.standard_work) for k, v in config.stations().items()}
    held = config.heldout()
    if held is not None:
        stations["HELDOUT"] = list(held[0].standard_work)

    app = FastAPI(title="station-cycle-vision edge", version="1.0")
    reg = CollectorRegistry()
    m_frames = Counter("scv_frames_total", "Frames processed", ["station"], registry=reg)
    m_latency = Histogram("scv_frame_seconds", "Model time per frame", buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05), registry=reg)
    m_cycles = Counter("scv_cycles_total", "Cycles completed", ["station"], registry=reg)
    m_devs = Counter("scv_deviations_total", "Cycles that deviated from standard work, by kind", ["station", "kind"], registry=reg)
    m_ct = Gauge("scv_last_cycle_seconds", "Last cycle time", ["station"], registry=reg)
    store = Store(db_path)
    models: dict[str, EdgeModels] = {}
    sessions: dict[str, Session] = {}
    lock = threading.Lock()

    def get_models() -> EdgeModels:
        if "m" not in models:
            models["m"] = EdgeModels(model_dir, int8=True)
        return models["m"]

    def session(station: str) -> Session:
        if station not in stations:
            raise HTTPException(404, f"unknown station {station}; this box serves {sorted(stations)}")
        with lock:
            if station not in sessions:
                if len(sessions) >= svc["stream"]["sessions_max"]:
                    raise HTTPException(429, "this box already serves its maximum number of stations")
                sessions[station] = Session(get_models(), stations[station], fps, hold, int(svc["clips"]["keep_seconds"] * fps))
            return sessions[station]

    def keep_clip(s: Session, station: str) -> str:
        frames = np.stack(list(s.frames))
        mask = get_models().person_mask(frames)
        a = svc["anonymise"]
        red = np.stack([pixelate(f, redact_mask(m, a["threshold"], a["dilate_px"]), a["block_px"]) for f, m in zip(frames, mask, strict=True)])
        clip_id = uuid.uuid4().hex
        np.savez_compressed(clips_dir / f"{clip_id}.npz", frames=red, station=station)
        return clip_id

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> Response:
        try:
            m = get_models()
        except Exception as e:  # noqa: BLE001
            return Response(json.dumps({"status": "models not loaded", "error": str(e)}), status_code=503, media_type="application/json")
        return Response(json.dumps({"status": "ready", "encoder": m.encoder_file}), media_type="application/json")

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(reg), media_type="text/plain; version=0.0.4")

    @app.post("/v1/stations/{station}/frames")
    async def frames(station: str, request: Request) -> dict[str, Any]:
        body = await request.body()
        if len(body) == 0 or len(body) % FRAME_BYTES:
            raise HTTPException(400, f"body must be N x 64 x 64 x 3 uint8 frames ({FRAME_BYTES} bytes each), got {len(body)} bytes")
        n = len(body) // FRAME_BYTES
        if n > svc["stream"]["max_frames_per_request"]:
            raise HTTPException(413, f"at most {svc['stream']['max_frames_per_request']} frames per request")
        arr = np.frombuffer(body, dtype=np.uint8).reshape(n, 64, 64, 3)
        s = session(station)
        m = get_models()
        out: list[dict[str, Any]] = []
        with s.lock:
            for f in arr:
                t = time.perf_counter()
                label = int(np.argmax(m.step(f, s.state)))
                m_latency.observe(time.perf_counter() - t)
                s.frames.append(f)
                for e in s.process.push(label):
                    if e["type"] == "cycle_completed":
                        m_cycles.labels(station).inc()
                        m_ct.labels(station).set(e["cycle_time_s"])
                        for kind in e["deviations"]:
                            m_devs.labels(station, kind).inc()
                        if e["deviations"]:
                            e["clip"] = keep_clip(s, station)
                    out.append(e)
            m_frames.labels(station).inc(n)
            store.add(station, s.stream, out)
            frame = s.process.frame
        return {"station": station, "stream": s.stream, "frames": n, "next_frame": frame, "events": out}

    @app.post("/v1/stations/{station}/reset")
    def reset(station: str) -> dict[str, Any]:
        s = session(station)
        with s.lock:
            ev = s.process.flush()
            store.add(station, s.stream, ev)
        with lock:
            sessions.pop(station, None)
        return {"station": station, "flushed": ev}

    @app.get("/v1/stations/{station}/cycles")
    def cycles(station: str, limit: int = 50) -> dict[str, Any]:
        if station not in stations:
            raise HTTPException(404, f"unknown station {station}")
        rows = store.query(station, "cycle_completed", 0, 10_000)[-max(1, min(limit, 1000)) :]
        times = [r["cycle_time_s"] for r in rows]
        return {
            "station": station,
            "standard_work": stations[station],
            "cycles": rows,
            "summary": {"count": len(rows), "median_s": float(np.median(times)) if times else None,
                        "deviating": sum(1 for r in rows if r["deviations"])},
        }

    @app.get("/v1/stations/{station}/events")
    def events(station: str, after_id: int = 0, limit: int = 500) -> dict[str, Any]:
        if station not in stations:
            raise HTTPException(404, f"unknown station {station}")
        return {"station": station, "events": store.query(station, None, after_id, max(1, min(limit, 5000)))}

    @app.get("/v1/clips/{clip_id}")
    def clip(clip_id: str) -> Response:
        if not clip_id.isalnum() or len(clip_id) != 32:
            raise HTTPException(400, "bad clip id")
        p = clips_dir / f"{clip_id}.npz"
        if not p.exists():
            raise HTTPException(404, "no such clip")
        buf = io.BytesIO(p.read_bytes())
        return Response(buf.getvalue(), media_type="application/octet-stream")

    return app
