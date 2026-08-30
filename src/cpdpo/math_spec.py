"""Dependency-free reference equations used by tests and validators."""

from __future__ import annotations

import math
from collections.abc import Iterable


def finite_sample_quantile_rank(n_cal: int, alpha: float) -> int:
    """Return the one-based higher order-statistic rank from the specification."""

    if n_cal < 1:
        raise ValueError("n_cal must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return min(n_cal, math.ceil((n_cal + 1) * (1.0 - alpha)))


def finite_sample_quantile(scores: Iterable[float], alpha: float) -> tuple[float, int]:
    ordered = sorted(float(score) for score in scores)
    if not ordered:
        raise ValueError("calibration scores cannot be empty")
    if not all(math.isfinite(score) and score >= 0.0 for score in ordered):
        raise ValueError("calibration scores must be finite and nonnegative")
    rank = finite_sample_quantile_rank(len(ordered), alpha)
    return ordered[rank - 1], rank


def calibration_score(label: int, margin: float, uncertainty: float, epsilon: float) -> float:
    if label not in {-1, 1}:
        raise ValueError("label must be +1 or -1")
    if uncertainty < 0.0 or epsilon <= 0.0:
        raise ValueError("uncertainty must be nonnegative and epsilon positive")
    return max(-label * margin, 0.0) / (uncertainty + epsilon)


def pair_signal(
    method: str,
    margin: float,
    uncertainty: float,
    q_alpha: float,
    epsilon: float,
) -> dict[str, float | bool]:
    """Scalar reference for PairPPO and CPDPO reward identities."""

    if method not in {"pairppo", "cpdpo"}:
        raise ValueError("method must be pairppo or cpdpo")
    if uncertainty < 0.0 or q_alpha < 0.0 or epsilon <= 0.0:
        raise ValueError("uncertainty/q must be nonnegative and epsilon positive")
    normalized_margin = abs(margin) / (uncertainty + epsilon)
    if method == "pairppo":
        certified = True
        gamma = abs(margin)
        reward = margin
    else:
        certified = normalized_margin > q_alpha and margin != 0.0
        gamma = max(abs(margin) - q_alpha * uncertainty, 0.0)
        reward = math.copysign(gamma, margin) if certified else 0.0
    return {
        "normalized_margin": normalized_margin,
        "certified": certified,
        "gamma": gamma,
        "reward": reward,
    }


def reference_anchored_signal(
    current_reward: float,
    reference_reward: float,
    uncertainty: float,
    q_alpha: float,
    epsilon: float,
) -> dict[str, float | bool]:
    """Scalar reference equation for the exploratory CPDPOv2 reward."""

    values = (current_reward, reference_reward, uncertainty, q_alpha, epsilon)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("CPDPOv2 signal inputs must be finite")
    if uncertainty < 0.0 or q_alpha < 0.0 or epsilon <= 0.0:
        raise ValueError("uncertainty/q must be nonnegative and epsilon positive")
    margin = current_reward - reference_reward
    normalized_margin = abs(margin) / (uncertainty + epsilon)
    return {
        "margin": margin,
        "uncertainty": uncertainty,
        "normalized_margin": normalized_margin,
        "certified_current_better": margin > 0.0 and normalized_margin > q_alpha,
        "reward": margin - q_alpha * uncertainty,
    }
