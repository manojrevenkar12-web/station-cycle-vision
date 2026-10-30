"""Overhead video of a station: the bench, bins, fixture, product, tool, the worker's arms and hands,
people walking through, then the camera (homography), lighting and sensor noise.

Everything the models must learn is visible in the pixels: hands moving between places, the plate
being carried, screws appearing, the tool moving, the product lifted for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .station import STEP_ID, Cycle, Segment, Station

WORLD = 96
SIZE = 64
PLATE = (18, 12)  # w, h in world units


@dataclass
class Episode:
    station: str
    domain: str
    frames: np.ndarray  # (T, 64, 64, 3) uint8
    labels: np.ndarray  # (T,) step id
    cycle: np.ndarray  # (T,) cycle index, -1 between cycles
    hands: np.ndarray  # (T, 2, 2) image coords, left then right
    hand_visible: np.ndarray  # (T, 2) bool
    tool: np.ndarray  # (T, 2) image coords
    person: np.ndarray  # (T, 64, 64) bool: pixels showing a person (worker or passer-by)
    cycles: list[Cycle]
    fps: int
    meta: dict[str, Any]


def _draw(rng: np.random.Generator, *a: float) -> float:
    lo, hi = a
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def camera_homography(rng: np.random.Generator, dom: dict[str, Any]) -> np.ndarray:
    th = np.deg2rad(_draw(rng, *dom["rotation_deg"]))
    s = _draw(rng, *dom["scale"])
    dx, dy = _draw(rng, *dom["shift"]), _draw(rng, *dom["shift"])
    tilt = _draw(rng, *dom["tilt"])
    c = WORLD / 2
    src = np.float32([[0, 0], [WORLD, 0], [WORLD, WORLD], [0, WORLD]])
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = (src - c) @ rot.T * s + c + [dx, dy]
    # tilt: the far (top) edge looks shorter
    top = pts[:2].mean(axis=0)
    pts[:2] = top + (pts[:2] - top) * (1 - tilt)
    dst = (pts * (SIZE / WORLD)).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_image(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts, np.ones((*pts.shape[:-1], 1))], axis=-1) @ h.T
    return p[..., :2] / p[..., 2:3]


def _ease(p: float) -> float:
    return 0.5 - 0.5 * np.cos(np.pi * min(max(p, 0.0), 1.0))


class Renderer:
    def __init__(self, station: Station, dom: dict[str, Any], rng: np.random.Generator) -> None:
        self.st = station
        self.rng = rng
        L = station.layout
        self.L = {k: np.array(v, dtype=float) for k, v in L.items()}
        self.H = camera_homography(rng, dom)
        self.gain = _draw(rng, *dom["gain"])
        self.tint = np.array([_draw(rng, *dom["tint"]) for _ in range(3)])
        self.vignette = _draw(rng, *dom["vignette"])
        self.vig_centre = rng.uniform(16, 48, size=2)
        self.noise = _draw(rng, *dom["noise"])
        self.sleeve = np.array([[0.15, 0.25, 0.55], [0.2, 0.2, 0.22], [0.55, 0.15, 0.12], [0.1, 0.4, 0.3]][rng.integers(0, 4)])
        self.glove = np.array([[0.85, 0.65, 0.5], [0.2, 0.35, 0.8], [0.1, 0.1, 0.12], [0.6, 0.45, 0.35]][rng.integers(0, 4)])
        self.bench = self._bench_texture(np.random.default_rng(abs(hash(station.name)) % 2**32))
        n = station.screws
        w, h = PLATE
        self.slots = np.array([[-w / 2 + w * (i + 1) / (n + 1), (-1) ** i * h / 4] for i in range(max(1, n))])
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        r2 = ((xx - self.vig_centre[0]) ** 2 + (yy - self.vig_centre[1]) ** 2) / (SIZE**2)
        self.vig = (1 - self.vignette * np.clip(r2 * 2, 0, 1))[..., None]

    def _bench_texture(self, rng: np.random.Generator) -> np.ndarray:
        base = np.ones((WORLD, WORLD, 3)) * np.array(self.st.bench)
        grain = cv2.GaussianBlur(rng.normal(0, 0.04, (WORLD, WORLD)), (0, 0), sigmaX=6, sigmaY=0.8)
        img = base + grain[..., None]
        # bins, fixture and output tray are fixed furniture
        for key, colour, size in [("base_bin", (0.25, 0.3, 0.35), (16, 14)), ("screw_bin", (0.3, 0.3, 0.3), (12, 10)),
                                  ("output", (0.2, 0.32, 0.22), (20, 16)), ("fixture", (0.22, 0.22, 0.24), (24, 18))]:
            x, y = self.L[key]
            cv2.rectangle(img, (int(x - size[0] / 2), int(y - size[1] / 2)), (int(x + size[0] / 2), int(y + size[1] / 2)), colour, -1)
        # stock in the bins
        bx, by = self.L["base_bin"]
        for i in range(3):
            cv2.rectangle(img, (int(bx - 6), int(by - 5 + i * 3)), (int(bx + 6), int(by - 3 + i * 3)), (0.72, 0.72, 0.7), -1)
        sx, sy = self.L["screw_bin"]
        for i in range(8):
            cv2.circle(img, (int(sx - 4 + (i % 4) * 2.5), int(sy - 2 + (i // 4) * 3)), 1, (0.85, 0.8, 0.3), -1)
        return np.clip(img, 0, 1)

    def plate(self, img: np.ndarray, centre: np.ndarray, screws: int, scale: float = 1.0) -> None:
        w, h = PLATE[0] * scale, PLATE[1] * scale
        x, y = centre
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0.78, 0.78, 0.76), -1)
        cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0.45, 0.45, 0.45), 1)
        for k in range(min(screws, len(self.slots))):
            p = centre + self.slots[k] * scale
            cv2.circle(img, (int(p[0]), int(p[1])), 1, (0.9, 0.82, 0.25), -1)

    def frame(self, hands: np.ndarray, tool: np.ndarray, tool_held: bool, plate_at: np.ndarray | None, plate_scale: float,
              screws: int, stack: int, passers: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        img = self.bench.copy()
        mask = np.zeros((WORLD, WORLD), dtype=np.uint8)
        ox, oy = self.L["output"]
        for i in range(min(stack, 4)):
            cv2.rectangle(img, (int(ox - 7), int(oy - 5 + i * 2)), (int(ox + 7), int(oy + 3 + i * 2)), (0.7, 0.7, 0.68), -1)
        if plate_at is not None:
            self.plate(img, plate_at, screws, plate_scale)
        # tool: a short bar, at its rest or in the right hand pointing at the work
        t0 = tool
        t1 = tool + (np.array([-4.0, -6.0]) if tool_held else np.array([5.0, -5.0]))
        cv2.line(img, (int(t0[0]), int(t0[1])), (int(t1[0]), int(t1[1])), (0.85, 0.5, 0.1), 2)
        # arms from below the bottom edge, then hands
        for i, hnd in enumerate(hands):
            shoulder = np.array([self.L["rest_left" if i == 0 else "rest_right"][0] + (-8 if i == 0 else 8), WORLD + 10])
            cv2.line(img, tuple(int(v) for v in shoulder), tuple(int(v) for v in hnd), tuple(self.sleeve.tolist()), 6)
            cv2.line(mask, tuple(int(v) for v in shoulder), tuple(int(v) for v in hnd), 1, 6)
        for hnd in hands:
            cv2.ellipse(img, (int(hnd[0]), int(hnd[1])), (4, 3), 0, 0, 360, tuple(self.glove.tolist()), -1)
            cv2.ellipse(mask, (int(hnd[0]), int(hnd[1])), (4, 3), 0, 0, 360, 1, -1)
        for p in passers:  # someone walking past: head and shoulders seen from above
            cv2.ellipse(img, (int(p[0]), int(p[1])), (14, 9), 0, 0, 360, (0.18, 0.18, 0.2), -1)
            cv2.circle(img, (int(p[0]), int(p[1])), 5, (0.12, 0.09, 0.07), -1)
            cv2.ellipse(mask, (int(p[0]), int(p[1])), (14, 9), 0, 0, 360, 1, -1)
        cam = cv2.warpPerspective(img.astype(np.float32), self.H, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpPerspective(mask, self.H, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0)
        cam = cam * self.gain * self.tint * self.vig
        cam = cam + self.rng.normal(0, self.noise, cam.shape)
        return (np.clip(cam, 0, 1) * 255).astype(np.uint8), m.astype(bool)


def render_episode(station: Station, segs: list[Segment], cycles: list[Cycle], seconds: float, fps: int,
                   dom_name: str, dom: dict[str, Any], rng: np.random.Generator, speed: float) -> Episode:
    r = Renderer(station, dom, rng)
    L = r.L
    n = int(seconds * fps)
    frames = np.zeros((n, SIZE, SIZE, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    cyc = np.full(n, -1, dtype=np.int64)
    hands_img = np.zeros((n, 2, 2))
    vis = np.zeros((n, 2), dtype=bool)
    tool_img = np.zeros((n, 2))
    person = np.zeros((n, SIZE, SIZE), dtype=bool)

    # people walking through
    per_min = _draw(rng, *dom["occlusions_per_minute"])
    n_pass = rng.poisson(per_min * seconds / 60)
    passes = []
    for _ in range(n_pass):
        t0 = rng.uniform(0, seconds)
        dur = rng.uniform(2, 4)
        y = rng.uniform(20, 76)
        direction = rng.choice([-1, 1])
        passes.append((t0, dur, y, direction))
    drop = _draw(rng, *dom["frame_drop"])

    seg_i = 0
    left_from = L["rest_left"].copy()
    right_from = L["rest_right"].copy()
    left_to, right_to = left_from.copy(), right_from.copy()
    stack = 0
    screws_done = 0
    cur_cycle = None
    for f in range(n):
        t = f / fps
        while seg_i < len(segs) - 1 and segs[seg_i].end <= t:
            # hands start the next segment where the last one left them
            left_from, right_from = left_to.copy(), right_to.copy()
            if segs[seg_i].step == "unload":
                stack += 1
            if segs[seg_i].step == "fasten":
                screws_done += 1
            seg_i += 1
        s = segs[seg_i]
        if s.cycle != cur_cycle and s.step == "pick_base":
            cur_cycle = s.cycle
            screws_done = 0
        p = (t - s.start) / max(s.end - s.start, 1e-6)
        fix = L["fixture"]
        targets = {
            "pick_base": (L["base_bin"], L["rest_right"]),
            "place_base": (fix + [-2, 0], L["rest_right"]),
            "pick_screw": (fix + [-12, 4], L["screw_bin"]),
            "fasten": (fix + [-12, 4], fix + (r.slots[s.screw % len(r.slots)] if s.screw >= 0 else 0) + [3, 3]),
            "inspect": (fix + [-11, -8], fix + [11, -8]),
            "unload": (L["output"], L["rest_right"]),
            "idle": (L["rest_left"], L["rest_right"]),
        }
        left_to, right_to = (np.array(v, dtype=float) for v in targets[s.step])
        e = _ease(p / 0.45)
        jitter = rng.normal(0, 0.6, (2, 2)) * (0.4 if p > 0.45 else 1.0) * speed
        hands = np.stack([left_from + (left_to - left_from) * e, right_from + (right_to - right_from) * e]) + jitter

        # where the plate is
        plate_at, scale = None, 1.0
        in_cycle = s.cycle >= 0
        if s.step == "pick_base" and p >= 0.6:
            plate_at = hands[0] + [0, -3]
        elif s.step == "place_base":
            plate_at = hands[0] + [0, -3] if p < 0.8 else fix.copy()
        elif s.step == "unload":
            plate_at = fix.copy() if p < 0.3 else (hands[0] + [0, -3] if p < 0.85 else None)
        elif s.step == "inspect":
            plate_at, scale = fix + [0, -6], 1.25
        elif in_cycle and s.step != "pick_base":
            plate_at = fix.copy()
        shown_screws = screws_done + (1 if s.step == "fasten" and p >= 0.85 else 0)
        if s.step == "unload" and p >= 0.85:
            shown_stack = stack + 1
        else:
            shown_stack = stack
        tool_held = s.step == "fasten" and 0.2 <= p <= 0.9
        tool = hands[1] + [1, -1] if tool_held else L["tool_rest"].copy()

        passers = []
        for t0, dur, y, d in passes:
            if t0 <= t <= t0 + dur:
                q = (t - t0) / dur
                x = -20 + q * (WORLD + 40)
                passers.append(np.array([x if d > 0 else WORLD - x, y]))
        if f > 0 and rng.uniform() < drop:
            frames[f] = frames[f - 1]  # the camera repeated a frame
            person[f] = person[f - 1]
        else:
            frames[f], person[f] = r.frame(hands, tool, tool_held, plate_at, scale, shown_screws, shown_stack, passers)
        labels[f] = STEP_ID[s.step]
        cyc[f] = s.cycle
        hi = to_image(r.H, hands)
        hands_img[f] = hi
        vis[f] = np.all((hi >= 0) & (hi < SIZE), axis=1)
        for pp in passers:  # a hand under a passer-by is not visible
            pi = to_image(r.H, pp[None])[0]
            vis[f] &= ~(np.abs(hi[:, 0] - pi[0]) < 9 * SIZE / WORLD + 1) | ~(np.abs(hi[:, 1] - pi[1]) < 6 * SIZE / WORLD + 1)
        tool_img[f] = to_image(r.H, tool[None])[0]
    meta = {"speed": speed, "gain": r.gain, "passers": len(passes), "frame_drop": drop}
    return Episode(station.name, dom_name, frames, labels, cyc, hands_img, vis, tool_img, person, cycles, fps, meta)
