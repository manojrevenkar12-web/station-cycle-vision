import numpy as np

from scv.process.cycles import align, clean, deviations, find_cycles, process, segments
from scv.process.live import run_live
from scv.sim.station import STEP_ID

SW = ["pick_base", "place_base", "pick_screw", "fasten", "pick_screw", "fasten", "inspect", "unload"]


def labels(seq, n=5, gap=3):
    out = [STEP_ID["idle"]] * gap
    for s in seq:
        out += [STEP_ID[s]] * n
    return np.array(out + [STEP_ID["idle"]] * gap)


def test_segments_and_flicker_removal():
    y = np.array([0, 0, 1, 1, 1, 2, 1, 1, 0])
    assert [(s.step, s.start, s.end) for s in segments(y)] == [(0, 0, 2), (1, 2, 5), (2, 5, 6), (1, 6, 8), (0, 8, 9)]
    assert clean(np.r_[y, 0], 2).tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]


def test_a_conforming_cycle_has_no_deviation():
    cycles = find_cycles(labels(SW))
    assert len(cycles) == 1
    assert cycles[0].steps == SW
    assert deviations(cycles[0].steps, SW) == []


def test_deviations_are_named():
    missing = SW[:4] + SW[6:]
    extra = SW[:6] + ["pick_screw", "fasten"] + SW[6:]
    order = SW[:5] + ["inspect", "fasten", "unload"]
    assert deviations(missing, SW) == ["missing"]
    assert deviations(extra, SW) == ["extra"]
    assert deviations(order, SW) == ["order"]
    d, ops = align(order, SW)
    assert d == 1 and ("swap", "fasten", "inspect") in ops


def test_cycle_time_is_from_first_pick_to_end_of_unload():
    c = process(labels(SW, n=5, gap=3), SW, fps=5, min_segment_s=0.2)[0]
    assert c.seconds(5) == len(SW) * 5 / 5


def test_live_matches_offline_on_clean_labels():
    y = np.concatenate([labels(SW), labels(SW[:4] + SW[6:])])
    ev = run_live(y, SW, fps=5, hold=2)
    done = [e for e in ev if e["type"] == "cycle_completed"]
    off = process(y, SW, 5, 0.4)
    assert [d["steps"] for d in done] == [c.steps for c in off]
    assert [d["deviations"] for d in done] == [c.deviations for c in off]
    # the debounce delays both ends equally, so cycle times are exact
    assert [d["cycle_time_s"] for d in done] == [c.seconds(5) for c in off]


def test_live_debounce_ignores_single_frame_flicker():
    y = labels(SW, n=6)
    y[10] = STEP_ID["inspect"]
    ev = run_live(y, SW, fps=5, hold=2)
    assert [e["deviations"] for e in ev if e["type"] == "cycle_completed"] == [[]]
