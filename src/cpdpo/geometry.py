"""Pair-difference geometry and fixed conformal calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from src.cpdpo.math_spec import finite_sample_quantile_rank
from src.cpdpo.spec import ALPHA, CALIBRATION_EPSILON, RIDGE_SCALE, ZERO_TRACE_RIDGE


@dataclass(frozen=True)
class PairGeometry:
    """Full Cholesky-factorized CPDPO geometry."""

    cholesky: torch.Tensor
    ridge: float
    n_rm: int

    @property
    def dimension(self) -> int:
        return int(self.cholesky.shape[0])

    def uncertainty(self, differences: torch.Tensor) -> torch.Tensor:
        if differences.shape[-1] != self.dimension:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.dimension}, found {differences.shape[-1]}"
            )
        original_dtype = differences.dtype
        vectors = differences.detach().to(device=self.cholesky.device, dtype=self.cholesky.dtype)
        flat = vectors.reshape(-1, self.dimension)
        # L z = d, so ||z||_2 = sqrt(d^T V^-1 d). No inverse is formed.
        solved = torch.linalg.solve_triangular(self.cholesky, flat.T, upper=False).T
        result = torch.linalg.vector_norm(solved, dim=-1).reshape(vectors.shape[:-1])
        return result.to(dtype=original_dtype)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "geometry": "full_pair_difference_gram",
            "geometry_mode": "full",
            "cholesky": self.cholesky.detach().cpu(),
            "ridge": self.ridge,
            "n_rm": self.n_rm,
            "dimension": self.dimension,
            "dtype": str(self.cholesky.dtype).replace("torch.", ""),
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "PairGeometry":
        if value.get("schema_version") != "1.0.0":
            raise ValueError("Unsupported pair geometry schema")
        factor = value.get("cholesky")
        if not isinstance(factor, torch.Tensor) or factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
            raise ValueError("Pair geometry contains an invalid Cholesky factor")
        if value.get("dimension") != factor.shape[0]:
            raise ValueError("Pair geometry dimension metadata is inconsistent")
        return cls(factor, float(value["ridge"]), int(value["n_rm"]))

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "PairGeometry":
        return cls.from_state_dict(torch.load(path, map_location=map_location))


def build_pair_geometry(
    differences: Iterable[torch.Tensor],
    *,
    ridge_scale: float = RIDGE_SCALE,
    zero_trace_ridge: float = ZERO_TRACE_RIDGE,
) -> tuple[PairGeometry, torch.Tensor]:
    """Accumulate normalized pair Gram matrix in float64 and factorize it."""

    gram_sum: torch.Tensor | None = None
    count = 0
    for batch in differences:
        if batch.ndim != 2:
            raise ValueError("Pair-difference batches must have shape [batch, feature]")
        values = batch.detach().to(device="cpu", dtype=torch.float64)
        gram_sum = values.T @ values if gram_sum is None else gram_sum + values.T @ values
        count += values.shape[0]
    if gram_sum is None or count == 0:
        raise ValueError("D_rm_train produced no pair differences")
    gram = gram_sum / count
    trace = float(torch.trace(gram).item())
    ridge = ridge_scale * trace / gram.shape[0] if trace != 0.0 else zero_trace_ridge
    matrix = gram + ridge * torch.eye(gram.shape[0], dtype=torch.float64)
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info.max().item()) != 0:
        raise RuntimeError(
            "Specified CPDPO ridge did not yield a positive-definite geometry; "
            "do not silently alter the frozen ridge rule"
        )
    return PairGeometry(factor, ridge, count), gram


def calibration_scores(
    labels: torch.Tensor,
    margins: torch.Tensor,
    uncertainties: torch.Tensor,
    *,
    epsilon: float = CALIBRATION_EPSILON,
) -> torch.Tensor:
    labels = labels.to(device=margins.device)
    if labels.shape != margins.shape or margins.shape != uncertainties.shape:
        raise ValueError("labels, margins, and uncertainties must have the same shape")
    if not torch.all((labels == 1) | (labels == -1)):
        raise ValueError("Calibration labels must be +1 or -1")
    if torch.any(uncertainties < 0):
        raise ValueError("Uncertainties cannot be negative")
    return torch.relu(-labels * margins) / (uncertainties + epsilon)


def conformal_quantile(scores: torch.Tensor, *, alpha: float = ALPHA) -> tuple[torch.Tensor, int]:
    flat = scores.detach().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("Calibration scores cannot be empty")
    if not torch.all(torch.isfinite(flat)) or torch.any(flat < 0):
        raise ValueError("Calibration scores must be finite and nonnegative")
    rank = finite_sample_quantile_rank(flat.numel(), alpha)
    # kthvalue implements the required one-based higher order statistic without interpolation.
    return torch.kthvalue(flat, rank).values, rank


def pair_signals(
    margins: torch.Tensor,
    uncertainties: torch.Tensor | None,
    *,
    method: str,
    q_alpha: float = 0.0,
    epsilon: float = CALIBRATION_EPSILON,
    reward_variant: str = "robust_margin",
) -> dict[str, torch.Tensor]:
    """Return detached PairPPO or CPDPO pair coefficients and diagnostics."""

    margins = margins.detach()
    if not torch.all(torch.isfinite(margins)):
        raise ValueError("Pair margins must be finite")
    if method == "pairppo":
        if uncertainties is None:
            uncertainties = torch.zeros_like(margins)
        # Normalized margins are not part of PairPPO.  Keep a finite internal
        # placeholder so optional rollout export and tensor collation cannot
        # introduce NaNs into otherwise valid PairPPO artifacts.
        normalized = torch.zeros_like(margins)
        certified = torch.ones_like(margins, dtype=torch.bool)
        gamma = margins.abs()
        reward = margins
    elif method == "cpdpo":
        if uncertainties is None:
            raise ValueError("CPDPO requires pair uncertainties")
        uncertainties = uncertainties.detach().to(margins)
        if (
            not torch.all(torch.isfinite(uncertainties))
            or torch.any(uncertainties < 0)
            or not math.isfinite(q_alpha)
            or q_alpha < 0
        ):
            raise ValueError("CPDPO uncertainty and q_alpha must be finite and nonnegative")
        normalized = margins.abs() / (uncertainties + epsilon)
        certified = (normalized > q_alpha) & (margins != 0)
        gamma = torch.relu(margins.abs() - q_alpha * uncertainties)
        if reward_variant == "robust_margin":
            reward = torch.sign(margins) * gamma * certified.to(gamma.dtype)
        elif reward_variant == "sign_only":
            reward = torch.sign(margins) * certified.to(gamma.dtype)
        else:
            raise ValueError("reward_variant must be robust_margin or sign_only")
    else:
        raise ValueError("method must be pairppo or cpdpo")
    return {
        "margin": margins,
        "uncertainty": uncertainties.detach(),
        "normalized_margin": normalized.detach(),
        "certified": certified.detach(),
        "gamma": gamma.detach(),
        "pair_reward": reward.detach(),
    }
