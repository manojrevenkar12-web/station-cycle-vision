import numpy as np
import torch

from scv import config
from scv.sim.station import STEPS
from scv.temporal.models import CausalTCN, segmentation_loss
from scv.train.ddp import ddp_gradients


def test_two_ranks_take_the_same_step_as_one_process():
    tc = config.load("experiment.yaml")["temporal"]
    cfg = {"classes": len(STEPS), "hidden": 16, "layers": 3, "stages": 2, "smooth_weight": tc["smooth_weight"]}
    torch.manual_seed(0)
    model = CausalTCN(cfg["classes"], cfg["hidden"], cfg["layers"], cfg["stages"])
    rng = np.random.default_rng(0)
    f = rng.normal(size=(2, 50, 128)).astype(np.float32)
    y = rng.integers(0, len(STEPS), size=(2, 50))
    grads = ddp_gradients(model.state_dict(), [(f[:1], y[:1]), (f[1:], y[1:])], cfg)
    # one process, both sequences: the mean of the per-rank losses
    model.zero_grad()
    loss = 0.5 * (
        segmentation_loss(model(torch.from_numpy(f[:1])), torch.from_numpy(y[:1]), cfg["smooth_weight"])
        + segmentation_loss(model(torch.from_numpy(f[1:])), torch.from_numpy(y[1:]), cfg["smooth_weight"])
    )
    loss.backward()
    for k, p in model.named_parameters():
        assert torch.allclose(grads[k], p.grad, atol=1e-6), k
