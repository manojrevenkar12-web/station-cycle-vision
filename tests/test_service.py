import numpy as np
import pytest
from fastapi.testclient import TestClient

from scv import config
from scv.edge.export import EdgeModels
from scv.process.live import run_live
from scv.service.app import create_app


@pytest.fixture()
def client(edge_dir, tmp_path):
    return TestClient(create_app(model_dir=edge_dir, db_path=tmp_path / "events.sqlite", clips_dir=tmp_path / "clips"))


def test_health_readiness_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["encoder"] == "encoder.int8.onnx"
    assert "scv_frames_total" in client.get("/metrics").text


def test_bad_input_is_refused_plainly(client):
    assert client.post("/v1/stations/S1/frames", content=b"").status_code == 400
    assert client.post("/v1/stations/S1/frames", content=b"x" * 100).status_code == 400
    assert client.post("/v1/stations/NOPE/frames", content=bytes(64 * 64 * 3)).status_code == 404
    too_many = bytes(64 * 64 * 3 * (config.load("service.yaml")["stream"]["max_frames_per_request"] + 1))
    assert client.post("/v1/stations/S1/frames", content=too_many).status_code == 413
    assert client.get("/v1/clips/../../etc").status_code in (400, 404)


def test_streaming_in_chunks_equals_the_live_reference(client, edge_dir, episode):
    """Whatever the chunking, the service emits exactly what run_live emits on the same labels."""
    frames = episode.frames
    events = []
    for i in range(0, len(frames), 37):
        r = client.post("/v1/stations/S1/frames", content=frames[i : i + 37].tobytes())
        assert r.status_code == 200
        events += r.json()["events"]
    events += client.post("/v1/stations/S1/reset").json()["flushed"]
    em = EdgeModels(edge_dir, int8=True)
    s = em.new_state()
    labels = np.array([int(np.argmax(em.step(f, s))) for f in frames])
    ref = run_live(labels, list(config.stations()["S1"].standard_work), episode.fps, 2)
    strip = [{k: v for k, v in e.items() if k != "clip"} for e in events]
    assert strip == ref
    stored = client.get("/v1/stations/S1/events?limit=5000").json()["events"]
    assert len(stored) == len(ref)
    assert client.get("/v1/stations/S1/cycles").json()["summary"]["count"] == sum(1 for e in ref if e["type"] == "cycle_completed")
