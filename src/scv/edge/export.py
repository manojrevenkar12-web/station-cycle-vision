"""Export for the edge box: ONNX encoder (float and static int8), ONNX streaming TCN, ONNX detector.

The edge image runs ONNX Runtime only (no PyTorch). `EdgeModels` loads the files and runs one frame
at a time; the service and the evaluation of the edge path both use it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

STATE_PREFIX = "state_"


def export_all(run_dir: Path, out: Path, k: int) -> dict[str, Any]:
    import torch
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    from ..perception.detector import Detector
    from ..temporal.models import CausalTCN, FrameEncoder, stack_frames
    from .. import config
    from .stream import StreamingTCN

    out.mkdir(parents=True, exist_ok=True)
    tc = config.load("experiment.yaml")["temporal"]
    enc = FrameEncoder(k)
    enc.load_state_dict(torch.load(run_dir / "models" / "encoder.pt"))
    enc.eval()
    tcn = CausalTCN(k, tc["hidden"], tc["layers"], tc["stages"])
    tcn.load_state_dict(torch.load(run_dir / "models" / "tcn.pt"))
    tcn.eval()
    det = Detector()
    det.load_state_dict(torch.load(run_dir / "models" / "detector.pt"))
    det.eval()

    class Feat(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.e = enc

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.e.features(x)

    class DetOut(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d = det

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            h, o, m = self.d(x)
            return torch.sigmoid(h), o, torch.sigmoid(m)

    x9 = torch.zeros(1, 9, 64, 64)
    torch.onnx.export(Feat(), (x9,), str(out / "encoder.onnx"), input_names=["frames"], output_names=["features"],
                      dynamic_axes={"frames": {0: "n"}, "features": {0: "n"}}, opset_version=17, dynamo=False)
    st = StreamingTCN(tcn).eval()
    state = st.init_state()
    names = [f"{STATE_PREFIX}{i}" for i in range(len(state))]
    torch.onnx.export(st, (torch.zeros(1, 128), *state), str(out / "tcn_step.onnx"), input_names=["features", *names],
                      output_names=["logits", *[f"new_{n}" for n in names]], opset_version=17, dynamo=False)
    torch.onnx.export(DetOut(), (torch.zeros(1, 3, 64, 64),), str(out / "detector.onnx"), input_names=["frame"],
                      output_names=["heat", "offset", "person"], dynamic_axes={"frame": {0: "n"}, "heat": {0: "n"}, "offset": {0: "n"}, "person": {0: "n"}},
                      opset_version=17, dynamo=False)

    calib = np.load(run_dir / "calibration.npz")["frames"]
    xs = stack_frames(torch.from_numpy(calib)).numpy()

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter([{"frames": xs[i : i + 32]} for i in range(0, len(xs), 32)])

        def get_next(self):  # noqa: ANN201
            return next(self.it, None)

    quantize_static(str(out / "encoder.onnx"), str(out / "encoder.int8.onnx"), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    manifest = {"classes": k, "state": [list(s) for s in st.shapes], "stack": 3, "size": 64,
                "files": {p.name: p.stat().st_size for p in sorted(out.glob("*.onnx"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


class EdgeModels:
    """ONNX Runtime sessions for one camera box; `step` processes one frame."""

    def __init__(self, model_dir: Path, int8: bool = True, threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        enc = "encoder.int8.onnx" if int8 and (model_dir / "encoder.int8.onnx").exists() else "encoder.onnx"
        self.encoder_file = enc
        self.enc = ort.InferenceSession(str(model_dir / enc), so, providers=prov)
        self.tcn = ort.InferenceSession(str(model_dir / "tcn_step.onnx"), so, providers=prov)
        self.det = ort.InferenceSession(str(model_dir / "detector.onnx"), so, providers=prov)
        self.manifest = json.loads((model_dir / "manifest.json").read_text())
        self.state_names = [i.name for i in self.tcn.get_inputs() if i.name.startswith(STATE_PREFIX)]

    def new_state(self) -> dict[str, Any]:
        return {"state": [np.zeros((1, c, n), np.float32) for c, n in self.manifest["state"]], "prev": []}

    def step(self, frame: np.ndarray, s: dict[str, Any]) -> np.ndarray:
        """frame (64, 64, 3) uint8 -> logits (K,). Keeps the last two frames and the TCN state in `s`."""
        f = frame.astype(np.float32).transpose(2, 0, 1) / 255.0
        prev = s["prev"]
        p1 = prev[-1] if prev else f
        p2 = prev[-2] if len(prev) > 1 else p1
        x = np.concatenate([p2, p1, f])[None]
        feat = self.enc.run(None, {"frames": x})[0]
        outs = self.tcn.run(None, {"features": feat, **dict(zip(self.state_names, s["state"], strict=True))})
        s["state"] = outs[1:]
        s["prev"] = [p1, f]
        return outs[0][0]

    def person_mask(self, frames: np.ndarray) -> np.ndarray:
        x = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return self.det.run(None, {"frame": x})[2]

    def latency(self, frames: np.ndarray, n: int = 300) -> dict[str, float]:
        s = self.new_state()
        for f in frames[:20]:
            self.step(f, s)
        ts = []
        for i in range(n):
            t = time.perf_counter()
            self.step(frames[i % len(frames)], s)
            ts.append(time.perf_counter() - t)
        med = float(np.median(ts) * 1000)
        return {"median_ms": med, "p99_ms": float(np.percentile(ts, 99) * 1000), "frames_per_s_per_core": 1000 / med}
