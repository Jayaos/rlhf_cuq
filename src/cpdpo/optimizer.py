"""Shared optimizer safeguards for the three controlled experiment branches."""

from __future__ import annotations

from types import MethodType


def install_gradient_clipping(trainer, max_grad_norm: float) -> None:
    """Clip immediately before every optimizer step and log the unclipped norm."""

    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    original_step = trainer.opt.step

    def clipped_step(_optimizer, *args, **kwargs):
        norm = trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), max_grad_norm)
        trainer.last_gradient_norm = float(norm.detach().float().item())
        trainer.accelerator.log(
            {"optimization/gradient_norm": trainer.last_gradient_norm},
            step=getattr(trainer, "iter_count", 0) + 1,
        )
        return original_step(*args, **kwargs)

    trainer.last_gradient_norm = 0.0
    trainer.opt.step = MethodType(clipped_step, trainer.opt)
