"""End to end on the quick profile: render, train, evaluate, export, report. Shapes and parity, not numbers."""

import json

import pytest

from scv import config
from scv.cli import main

pytestmark = pytest.mark.slow


def test_quick_profile_end_to_end():
    assert main(["run", "--profile", "quick"]) == 0
    assert main(["edge", "--profile", "quick"]) == 0
    assert main(["report", "--profile", "quick"]) == 0
    r = json.loads((config.ROOT / "runs" / "quick" / "results.json").read_text())
    assert set(r["temporal"]) == {"frame-only", "causal-tcn (no augmentation)", "causal-tcn", "offline-transformer"}
    assert {"train", "camera", "lighting", "busy", "new_station", "heldout"} <= set(r["domains"])
    assert r["edge"]["latency"]["int8"]["median_ms"] > 0
    assert r["mlflow"]["backend"] == "mlflow"
    assert (config.ROOT / "runs" / "quick" / "REPORT.md").read_text().startswith("# Report")
    first = config.ROOT / "reports" / "heldout" / "first_run.json"
    if first.exists():  # a quick run never writes the held-out first run
        assert json.loads(first.read_text())["recorded_at"] != r["generated_at"]
