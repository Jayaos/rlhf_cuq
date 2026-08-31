"""AdvPO confidence matrix and closed-form adversarial reward-head update."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


CONFIDENCE_SCHEMA = "1.0.0"


@dataclass(frozen=True)
class AdvPOConfidenceGeometry:
    """Cholesky representation of M_D=lambda I+sum e e^T from paper Eq. (4)."""

    cholesky: torch.Tensor
    ridge_lambda: float
    n_responses: int

    @property
    def dimension(self) -> int:
        return int(self.cholesky.shape[0])

    def solve(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.shape[-1] != self.dimension:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.dimension}, found {vectors.shape[-1]}"
            )
        values = vectors.detach().to(device=self.cholesky.device, dtype=self.cholesky.dtype)
        flat = values.reshape(-1, self.dimension)
        first = torch.linalg.solve_triangular(self.cholesky, flat.T, upper=False)
        solved = torch.linalg.solve_triangular(self.cholesky.T, first, upper=True).T
        return solved.reshape(values.shape)

    def quadratic_norm(self, vectors: torch.Tensor) -> torch.Tensor:
        solved = self.solve(vectors)
        squared = (vectors.to(solved) * solved).sum(dim=-1).clamp_min(0.0)
        return torch.sqrt(squared)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIDENCE_SCHEMA,
            "geometry": "advpo_individual_feature_outer_product_sum",
            "cholesky": self.cholesky.detach().cpu(),
            "ridge_lambda": self.ridge_lambda,
            "n_responses": self.n_responses,
            "dimension": self.dimension,
            "dtype": str(self.cholesky.dtype).replace("torch.", ""),
        }

    @classmethod
    def from_state_dict(cls, value: dict[str, Any]) -> "AdvPOConfidenceGeometry":
        if value.get("schema_version") != CONFIDENCE_SCHEMA:
            raise ValueError("Unsupported AdvPO confidence-matrix schema")
        if value.get("geometry") != "advpo_individual_feature_outer_product_sum":
            raise ValueError("Artifact is not an AdvPO confidence matrix")
        factor = value.get("cholesky")
        if not isinstance(factor, torch.Tensor) or factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
            raise ValueError("AdvPO confidence artifact contains an invalid Cholesky factor")
        if int(value.get("dimension", -1)) != factor.shape[0]:
            raise ValueError("AdvPO confidence dimension metadata is inconsistent")
        ridge = float(value["ridge_lambda"])
        count = int(value["n_responses"])
        if (
            not math.isfinite(ridge)
            or ridge <= 0.0
            or count < 1
            or factor.dtype != torch.float64
            or not torch.isfinite(factor).all()
            or not torch.equal(factor, torch.tril(factor))
            or bool(torch.any(torch.diag(factor) <= 0.0))
        ):
            raise ValueError("AdvPO confidence artifact has invalid ridge, count, or values")
        return cls(factor, ridge, count)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "AdvPOConfidenceGeometry":
        return cls.from_state_dict(torch.load(path, map_location=map_location))


def build_advpo_confidence_geometry(
    feature_batches: Iterable[torch.Tensor],
    *,
    ridge_lambda: float,
) -> tuple[AdvPOConfidenceGeometry, torch.Tensor]:
    """Accumulate the paper's unnormalised individual-feature sum in float64."""

    if ridge_lambda <= 0.0:
        raise ValueError("AdvPO ridge_lambda must be positive")
    gram_sum: torch.Tensor | None = None
    count = 0
    for batch in feature_batches:
        if batch.ndim != 2:
            raise ValueError("AdvPO feature batches must have shape [batch, feature]")
        values = batch.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise ValueError("AdvPO confidence features must be finite")
        gram_sum = values.T @ values if gram_sum is None else gram_sum + values.T @ values
        count += values.shape[0]
    if gram_sum is None or count < 1:
        raise ValueError("D_rm_train produced no response features")
    matrix = gram_sum + ridge_lambda * torch.eye(gram_sum.shape[0], dtype=torch.float64)
    factor, info = torch.linalg.cholesky_ex(matrix)
    if int(info.max().item()) != 0:
        raise RuntimeError("Declared AdvPO ridge did not yield a positive-definite confidence matrix")
    return AdvPOConfidenceGeometry(factor, float(ridge_lambda), count), gram_sum


def advpo_batch_signal(
    *,
    current_rewards: torch.Tensor,
    reference_rewards: torch.Tensor,
    current_features: torch.Tensor,
    reference_features: torch.Tensor,
    geometry: AdvPOConfidenceGeometry,
    confidence_radius_squared: float,
    epsilon: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    """Evaluate the closed-form shared adversarial head from paper Eq. (7)."""

    if confidence_radius_squared <= 0.0 or epsilon <= 0.0:
        raise ValueError("AdvPO B and epsilon must be positive")
    if current_rewards.shape != reference_rewards.shape or current_rewards.ndim != 1:
        raise ValueError("Current/reference reward batches must be equal one-dimensional tensors")
    if current_features.shape != reference_features.shape or current_features.ndim != 2:
        raise ValueError("Current/reference feature batches must have equal [batch, feature] shapes")
    if current_features.shape[0] != current_rewards.numel():
        raise ValueError("AdvPO rewards and features have inconsistent batch sizes")
    for name, value in {
        "current rewards": current_rewards,
        "reference rewards": reference_rewards,
        "current features": current_features,
        "reference features": reference_features,
    }.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"AdvPO {name} must be finite")

    work_dtype = geometry.cholesky.dtype
    current = current_features.detach().to(device=geometry.cholesky.device, dtype=work_dtype)
    reference = reference_features.detach().to(device=geometry.cholesky.device, dtype=work_dtype)
    g = (current - reference).mean(dim=0)
    solved = geometry.solve(g).to(dtype=work_dtype)
    norm_squared = (g * solved).sum().clamp_min(0.0)
    norm = torch.sqrt(norm_squared)
    radius = confidence_radius_squared**0.5
    if float(norm.item()) <= epsilon:
        direction = torch.zeros_like(g)
        # The analytical multiplier tends to infinity as g -> 0. Store a
        # finite sentinel because the adversarial displacement is exactly zero
        # and scientific metrics must remain JSON-serialisable.
        lambda_star = torch.zeros((), dtype=work_dtype, device=g.device)
        degenerate_direction = torch.ones((), dtype=torch.bool, device=g.device)
    else:
        direction = radius * solved / norm
        lambda_star = norm / radius
        degenerate_direction = torch.zeros((), dtype=torch.bool, device=g.device)
    current_penalty = current @ direction
    reference_penalty = reference @ direction
    current_adv = current_rewards.detach().to(current_penalty) - current_penalty
    reference_adv = reference_rewards.detach().to(reference_penalty) - reference_penalty
    robust_objective = current_adv.mean() - reference_adv.mean()
    closed_form_objective = (
        current_rewards.detach().to(work_dtype).mean()
        - reference_rewards.detach().to(work_dtype).mean()
        - radius * norm
    )
    if not torch.allclose(robust_objective, closed_form_objective, atol=1e-8, rtol=1e-7):
        raise RuntimeError("AdvPO closed-form objective identity failed")
    return {
        "mean_feature_difference": g.detach(),
        "inverse_weighted_difference": solved.detach(),
        "mahalanobis_mean_difference": norm.detach(),
        "lambda_star": lambda_star.detach(),
        "degenerate_direction": degenerate_direction.detach(),
        "adversarial_direction": direction.detach(),
        "current_penalty": current_penalty.detach(),
        "reference_penalty": reference_penalty.detach(),
        "current_adversarial_reward": current_adv.detach(),
        "reference_adversarial_reward": reference_adv.detach(),
        "robust_objective": robust_objective.detach(),
        "closed_form_objective": closed_form_objective.detach(),
    }
