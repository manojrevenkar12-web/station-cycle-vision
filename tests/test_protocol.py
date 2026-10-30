"""The experiment's rules, checked mechanically."""

import hashlib
import json

from scv import config


def test_heldout_station_is_valid_blind_and_unchanged():
    assert (config.ROOT / "heldout" / "STATUS").read_text().strip() in {"blind", "regression"}
    assert config.heldout() is not None
    text = (config.ROOT / "heldout" / "station.yaml").read_bytes()
    recorded = (config.ROOT / "heldout" / "station.sha256").read_text().split()[0]
    assert hashlib.sha256(text).hexdigest() == recorded
    first = config.ROOT / "reports" / "heldout" / "first_run.json"
    if first.exists():
        assert json.loads(first.read_text())["station_sha256"] == recorded


def test_no_test_set_trains_or_tunes_anything():
    src = (config.ROOT / "src" / "scv" / "evaluate" / "experiment.py").read_text()
    # training uses data["train"] (and the new station's single onboarding recording) only
    assert 'train_encoder(data["train"]' in src and 'train_detector(data["train"]' in src
    assert '[(feats[encn][id(e)], e.labels) for e in data["train"]]' in src
    # the detector threshold is chosen on validation recordings
    assert 'hand_f1(det, data["val"], thr' in src


def test_onboarding_and_test_recordings_of_the_new_station_are_different():
    from scv.sim import dataset

    src = (config.ROOT / "src" / "scv" / "sim" / "dataset.py").read_text()
    assert "200_000" in src and "210_000 + e" in src
    assert dataset is not None


def test_release_targets_were_not_lowered():
    rel = config.load("release.yaml")["targets"]
    assert rel == {
        "cycle_recall": 0.95,
        "cycle_time_mae_s": 1.0,
        "deviation_recall": 0.80,
        "deviation_precision": 0.80,
        "leak_rate": 0.01,
        "int8_agreement": 0.98,
        "edge_ms_per_frame": 20.0,
    }
