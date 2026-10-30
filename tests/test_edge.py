import numpy as np
import torch

from scv import config
from scv.edge.export import EdgeModels
from scv.edge.stream import StreamingTCN
from scv.process.live import run_live
from scv.sim.station import STEPS
from scv.temporal.models import CausalTCN, FrameEncoder, stack_frames


def test_streaming_tcn_equals_the_offline_model():
    torch.manual_seed(3)
    tc = config.load("experiment.yaml")["temporal"]
    m = CausalTCN(len(STEPS), tc["hidden"], tc["layers"], tc["stages"]).eval()
    f = torch.randn(1, 400, 128)
    with torch.no_grad():
        off = m(f)[-1][0]
        s = StreamingTCN(m).eval()
        st = s.init_state()
        on = []
        for t in range(400):
            o, *st = s(f[:, t], *st)
            on.append(o[0])
    assert torch.max(torch.abs(off - torch.stack(on))) < 1e-5


def test_onnx_edge_path_equals_pytorch(edge_dir, episode):
    """Float ONNX encoder + streaming ONNX TCN, frame by frame == PyTorch on the whole recording."""
    run = edge_dir.parent
    tc = config.load("experiment.yaml")["temporal"]
    enc = FrameEncoder(len(STEPS))
    enc.load_state_dict(torch.load(run / "models" / "encoder.pt"))
    tcn = CausalTCN(len(STEPS), tc["hidden"], tc["layers"], tc["stages"])
    tcn.load_state_dict(torch.load(run / "models" / "tcn.pt"))
    enc.eval()
    tcn.eval()
    frames = episode.frames[:120]
    with torch.no_grad():
        ref = tcn(enc.features(stack_frames(torch.from_numpy(frames)))[None])[-1][0].numpy()
    em = EdgeModels(edge_dir, int8=False)
    s = em.new_state()
    got = np.stack([em.step(f, s) for f in frames])
    assert np.max(np.abs(got - ref)) < 1e-3


def test_int8_encoder_is_smaller_and_runs(edge_dir, episode):
    assert (edge_dir / "encoder.int8.onnx").stat().st_size < (edge_dir / "encoder.onnx").stat().st_size / 2
    em = EdgeModels(edge_dir, int8=True)
    s = em.new_state()
    out = np.stack([em.step(f, s) for f in episode.frames[:30]])
    assert out.shape == (30, len(STEPS)) and np.isfinite(out).all()
    assert em.latency(episode.frames[:10], n=20)["median_ms"] > 0


def test_live_reference_is_deterministic_on_model_output(edge_dir, episode):
    em = EdgeModels(edge_dir, int8=True)
    s = em.new_state()
    labels = np.array([int(np.argmax(em.step(f, s))) for f in episode.frames[:100]])
    sw = list(config.stations()["S1"].standard_work)
    assert run_live(labels, sw, 5, 2) == run_live(labels.copy(), sw, 5, 2)
