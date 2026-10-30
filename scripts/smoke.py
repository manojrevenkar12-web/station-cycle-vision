"""Stream a rendered recording to a running edge service and check it reports cycles.

python scripts/smoke.py http://localhost:8080
"""

from __future__ import annotations

import json
import sys
import urllib.request

from scv import config
from scv.sim.video import make_episode


def post(url: str, body: bytes) -> dict:
    req = urllib.request.Request(url, data=body, method="POST", headers={"content-type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main(base: str) -> int:
    cfg = config.load("stations.yaml")
    ep = make_episode(config.stations()["S1"], "train", config.domains()["train"], cfg, 120, 424242)
    events = []
    for i in range(0, len(ep.frames), 100):
        events += post(f"{base}/v1/stations/S1/frames", ep.frames[i : i + 100].tobytes())["events"]
    with urllib.request.urlopen(f"{base}/v1/stations/S1/cycles", timeout=30) as r:
        cycles = json.loads(r.read())
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as r:
        metrics = r.read().decode()
    found = cycles["summary"]["count"]
    print(json.dumps({"true_cycles": len(ep.cycles), "reported_cycles": found, "median_s": cycles["summary"]["median_s"]}))
    assert "scv_frames_total" in metrics
    assert found >= max(1, len(ep.cycles) - 2), "the service should report (almost) every cycle of a bench recording"
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"))
