"""Isolated deterministic Torch RNG streams compatible with Transformers 4.31."""

from __future__ import annotations

from contextlib import contextmanager

import torch


class GenerationSeedStream:
    """Give each generation call a deterministic seed without perturbing training RNG."""

    def __init__(self, seed: int, device: torch.device):
        self.seed = int(seed)
        self.device = torch.device(device)
        self.counter = 0

    @contextmanager
    def activate(self):
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            call_seed = self.seed + self.counter
            self.counter += 1
            torch.manual_seed(call_seed)
            if devices:
                torch.cuda.manual_seed(call_seed)
            yield
