"""Data-parallel training of the temporal model with torch.distributed (gloo on CPU, nccl on GPUs).

Each rank trains on its own shard of recordings; DistributedDataParallel averages gradients, so N
ranks with equal shards take the same step as one process on the whole batch
(tests/test_ddp.py checks it). The committed run uses one process; this is the path for a cluster.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ..temporal.models import CausalTCN, segmentation_loss


def _worker(
    rank: int, world: int, init: str, state: dict, batches: list[tuple[np.ndarray, np.ndarray]], cfg: dict[str, Any], out: str
) -> None:
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    torch.manual_seed(0)
    model = CausalTCN(cfg["classes"], cfg["hidden"], cfg["layers"], cfg["stages"])
    model.load_state_dict(state)
    ddp = DDP(model)
    f, y = batches[rank]
    loss = segmentation_loss(ddp(torch.from_numpy(f)), torch.from_numpy(y), cfg["smooth_weight"])
    loss.backward()
    if rank == 0:
        torch.save({k: p.grad.clone() for k, p in model.named_parameters()}, out)
    dist.barrier()
    dist.destroy_process_group()


def ddp_gradients(state: dict, batches: list[tuple[np.ndarray, np.ndarray]], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Gradients after one DDP backward pass, one batch per rank (as rank 0 sees them)."""
    world = len(batches)
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{Path(d) / 'rendezvous'}"
        out = str(Path(d) / "grads.pt")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        mp.spawn(_worker, args=(world, init, state, batches, cfg, out), nprocs=world, join=True)
        return torch.load(out)
