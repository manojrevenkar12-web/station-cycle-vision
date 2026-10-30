"""The causal TCN one frame at a time, with explicit state, for the edge.

Each dilated layer keeps the last 2*d hidden vectors it has seen (a ring buffer). Starting from zero
buffers is exactly the zero left-padding of the offline model, so streaming a recording frame by
frame gives the same logits as running the whole recording at once (tests/test_stream.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..temporal.models import CausalTCN


class StreamingTCN(nn.Module):
    def __init__(self, tcn: CausalTCN) -> None:
        super().__init__()
        self.tcn = tcn
        self.shapes = [(s.inp.out_channels, 2 * 2**i) for s in tcn.stages for i in range(s.layers)]

    def init_state(self, batch: int = 1) -> list[torch.Tensor]:
        return [torch.zeros(batch, c, n) for c, n in self.shapes]

    def forward(self, x: torch.Tensor, *state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """x (B, D) features of one frame; state per layer (B, hidden, 2d). Returns logits and new state."""
        new: list[torch.Tensor] = []
        k = 0
        inp = x
        logits = x
        for si, stage in enumerate(self.tcn.stages):
            h = F.conv1d(inp[:, :, None], stage.inp.weight, stage.inp.bias)[:, :, 0]
            for i, (dil, pw) in enumerate(zip(stage.dil, stage.pw, strict=True)):
                d = 2**i
                buf = state[k]
                taps = torch.stack([buf[:, :, 0], buf[:, :, d], h], dim=2)  # t-2d, t-d, t
                y = F.conv1d(taps, dil.weight, dil.bias)[:, :, 0]
                new.append(torch.cat([buf[:, :, 1:], h[:, :, None]], dim=2))
                h = h + F.conv1d(F.relu(y)[:, :, None], pw.weight, pw.bias)[:, :, 0]
                k += 1
            logits = F.conv1d(h[:, :, None], stage.out.weight, stage.out.bias)[:, :, 0]
            if si + 1 < len(self.tcn.stages):
                inp = F.softmax(logits, dim=1)
        return (logits, *new)
