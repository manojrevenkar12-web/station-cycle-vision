import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scv import config
from scv.sim.station import STEPS
from scv.sim.video import make_episode


@pytest.fixture(scope="session")
def episode():
    cfg = config.load("stations.yaml")
    return make_episode(config.stations()["S1"], "train", config.domains()["train"], cfg, 60, 7)


@pytest.fixture(scope="session")
def edge_dir(tmp_path_factory) -> Path:
    """Untrained models exported exactly like trained ones: enough to test plumbing and parity."""
    from scv.edge.export import export_all
    from scv.perception.detector import Detector
    from scv.temporal.models import CausalTCN, FrameEncoder

    tc = config.load("experiment.yaml")["temporal"]
    run = tmp_path_factory.mktemp("run")
    (run / "models").mkdir()
    torch.manual_seed(0)
    torch.save(FrameEncoder(len(STEPS)).state_dict(), run / "models" / "encoder.pt")
    torch.save(CausalTCN(len(STEPS), tc["hidden"], tc["layers"], tc["stages"]).state_dict(), run / "models" / "tcn.pt")
    torch.save(Detector().state_dict(), run / "models" / "detector.pt")
    rng = np.random.default_rng(0)
    np.savez_compressed(run / "calibration.npz", frames=rng.integers(0, 255, (64, 64, 64, 3), dtype=np.uint8))
    out = run / "edge"
    export_all(run, out, len(STEPS))
    assert json.loads((out / "manifest.json").read_text())["classes"] == len(STEPS)
    return out
