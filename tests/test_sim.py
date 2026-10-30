import numpy as np
import pytest

from scv import config
from scv.sim.render import camera_homography, to_image
from scv.sim.station import STEP_ID, Station, plan_cycle, schedule
from scv.sim.video import make_episode


def test_every_configured_station_is_valid():
    st = config.stations()
    assert {"S1", "S2", "S3", "S4", "S5"} <= set(st)
    for s in st.values():
        assert s.standard_work[0] == "pick_base" and s.standard_work[-1] == "unload"


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({"standard_work": ["place_base", "unload"]}, "starts with pick_base"),
        ({"standard_work": ["pick_base", "fasten", "unload"]}, "follows a pick_screw"),
        ({"standard_work": ["pick_base", "dance", "unload"]}, "unknown steps"),
        ({"layout": {"base_bin": [0, 0]}}, "layout lacks"),
    ],
)
def test_station_validation_names_the_problem(bad, msg):
    good = config.load("stations.yaml")["stations"]["S1"]
    with pytest.raises(ValueError, match=msg):
        Station.from_dict("X", {**good, **bad})


def test_schedule_covers_time_without_gaps_and_cycles_follow_their_plan():
    s = config.stations()["S2"]
    cfg = config.load("stations.yaml")
    segs, cycles, speed = schedule(s, cfg, 300, np.random.default_rng(1))
    for a, b in zip(segs[:-1], segs[1:], strict=True):
        assert abs(a.end - b.start) < 1e-9
    assert 0.8 <= speed <= 1.25
    assert len(cycles) >= 10
    for c in cycles:
        in_cycle = [g.step for g in segs if g.cycle == c.index and g.step != "idle"]
        assert in_cycle == c.steps


def test_deviations_change_the_plan_the_way_they_say():
    sw = tuple(config.stations()["S2"].standard_work)
    rng = np.random.default_rng(2)
    for kind in ("missing", "order", "extra"):
        cfg = {"missing": 0.0, "order": 0.0, "extra": 0.0, "pause": 0.0, kind: 1.0}
        steps, devs, _ = plan_cycle(sw, cfg, rng)
        assert devs == [kind]
        if kind == "missing":
            assert len(steps) == len(sw) - 2 and steps.count("fasten") == sw.count("fasten") - 1
        if kind == "extra":
            assert steps.count("fasten") == sw.count("fasten") + 1
        if kind == "order":
            assert sorted(steps) == sorted(sw) and steps != list(sw)


def test_episode_truth_is_consistent(episode):
    e = episode
    assert e.frames.shape == (300, 64, 64, 3) and e.frames.dtype == np.uint8
    assert set(np.unique(e.labels)) <= set(range(7))
    for c in e.cycles:
        first = int(np.ceil(c.start * e.fps))
        assert e.labels[first] == STEP_ID["pick_base"] or e.labels[first + 1] == STEP_ID["pick_base"]
    assert e.person.any(axis=(1, 2)).all()  # the worker is always in view
    assert e.hand_visible.mean() > 0.9


def test_rendering_is_deterministic():
    cfg = config.load("stations.yaml")
    a = make_episode(config.stations()["S3"], "busy", config.domains()["busy"], cfg, 20, 5)
    b = make_episode(config.stations()["S3"], "busy", config.domains()["busy"], cfg, 20, 5)
    assert np.array_equal(a.frames, b.frames) and np.array_equal(a.labels, b.labels)


def test_hand_truth_lands_on_hand_pixels(episode):
    """The projected hand position must sit on the worker mask (the camera math is right)."""
    e = episode
    on = 0
    total = 0
    for t in range(0, len(e.frames), 5):
        for h, v in zip(e.hands[t], e.hand_visible[t], strict=True):
            if v:
                x, y = int(round(h[0])), int(round(h[1]))
                if 0 <= x < 64 and 0 <= y < 64:
                    total += 1
                    on += bool(e.person[t, max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].any())
    assert on / total > 0.95


def test_homography_maps_bench_corners_inside_the_frame_for_a_straight_camera():
    rng = np.random.default_rng(0)
    dom = {"rotation_deg": [0, 0], "scale": [1, 1], "shift": [0, 0], "tilt": [0, 0]}
    h = camera_homography(rng, dom)
    pts = to_image(h, np.array([[0.0, 0.0], [96.0, 96.0], [48.0, 48.0]]))
    assert np.allclose(pts, [[0, 0], [64, 64], [32, 32]], atol=1e-6)
