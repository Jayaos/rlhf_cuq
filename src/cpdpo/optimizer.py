"""Shared optimizer safeguards for the three controlled experiment branches."""

from __future__ import annotations

import math
from types import MethodType


def install_gradient_clipping(trainer, max_grad_norm: float) -> None:
    """Clip immediately before every optimizer step and log the unclipped norm."""

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    original_step = trainer.opt.step

    def clipped_step(_optimizer, *args, **kwargs):
        norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), max_grad_norm)
        trainer.last_gradient_norm = float(norm.detach().float().item())
        if not math.isfinite(trainer.last_gradient_norm):
            trainer.opt.zero_grad()
            raise FloatingPointError(
                "Non-finite policy gradient norm before optimizer step "
                f"{getattr(trainer, 'iter_count', 0) + 1}; optimizer step was not applied"
            )
        trainer.accelerator.log(
            {"optimization/gradient_norm": trainer.last_gradient_norm},
            step=getattr(trainer, "iter_count", 0) + 1,
        )
        return original_step(*args, **kwargs)

    trainer.last_gradient_norm = 0.0
    trainer.opt.step = MethodType(clipped_step, trainer.opt)


def validate_training_precision(trainer, expected: str) -> None:
    """Reject a launcher whose mixed precision disagrees with run metadata."""

    actual = str(trainer.accelerator.mixed_precision).lower()
    normalized_actual = "fp32" if actual in {"no", "none", "fp32"} else actual
    if expected not in {"bf16", "fp32"}:
        raise ValueError(f"Unsupported declared training precision: {expected!r}")
    if normalized_actual != expected:
        raise RuntimeError(
            "Accelerate precision mismatch: "
            f"run declares {expected}, launcher resolved {normalized_actual}"
        )
