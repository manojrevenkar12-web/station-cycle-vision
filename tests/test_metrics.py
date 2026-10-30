import numpy as np

from scv.evaluate import metrics as M
from scv.perception.anonymise import leak_metrics, redact_mask
from scv.perception.tracker import Tracker, mot_metrics


def test_segmentation_metrics_on_known_cases():
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    assert M.segmentation(y, y) == {"accuracy": 1.0, "edit": 100.0, "f1_10": 100.0, "f1_25": 100.0, "f1_50": 100.0}
    flicker = y.copy()
    flicker[5] = 2
    s = M.segmentation(flicker, y)
    assert s["accuracy"] > 0.96 and s["edit"] < 80  # one wrong frame, two extra segments
    shifted = np.r_[[0] * 2, y[:-2]]  # every boundary 2 frames late
    assert M.f1_at(shifted, y, 0.5) == 100.0 and M.f1_at(shifted, y, 0.75) < 100.0


def test_tracker_keeps_identities_through_a_short_occlusion():
    T = 40
    truth = np.zeros((T, 2, 2))
    truth[:, 0] = np.stack([np.linspace(10, 30, T), np.full(T, 20)], 1)
    truth[:, 1] = np.stack([np.linspace(50, 30, T), np.full(T, 40)], 1)
    vis = np.ones((T, 2), dtype=bool)
    tr = Tracker(gate_px=6, max_missed=5)
    hyp = []
    for t in range(T):
        dets = truth[t] if not (15 <= t < 19) else truth[t, 1:]  # hand 0 hidden for 4 frames
        hyp.append(tr.step(t, dets))
    vis[15:19, 0] = False
    m = mot_metrics(truth, vis, hyp, match_px=3)
    assert m["id_switches"] == 0 and m["idf1"] > 0.95 and m["mota"] > 0.95


def test_anonymiser_dilation_closes_small_leaks():
    truth = np.zeros((1, 16, 16), dtype=bool)
    truth[0, 4:10, 4:10] = True
    prob = np.zeros((16, 16))
    prob[5:9, 5:9] = 0.9  # the model found the core but not the edge
    loose = leak_metrics(truth, redact_mask(prob, 0.5, 0)[None])
    dilated = leak_metrics(truth, redact_mask(prob, 0.5, 2)[None])
    assert loose["leak_rate"] > 0.5 and dilated["leak_rate"] == 0.0
