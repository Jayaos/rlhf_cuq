"""Frozen v1 constants and validated runtime configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ALPHA = 0.10
CALIBRATION_EPSILON = 1.0e-8
RIDGE_SCALE = 1.0e-3
ZERO_TRACE_RIDGE = 1.0e-6
RESPONSES_PER_PROMPT = 2

PAIR_METHODS = frozenset({"pairppo", "cpdpo"})
ALL_METHODS = frozenset({"ppo", *PAIR_METHODS})


def is_main_alpha(alpha: float) -> bool:
    """Return whether alpha is the specification-defined main value."""

    return float(alpha) == ALPHA


def alpha_tag(alpha: float) -> str:
    """Return a stable filesystem-safe tag for a validated alpha value."""

    parsed = float(alpha)
    if not 0.0 < parsed < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return format(parsed, ".15g").replace("-", "m").replace("+", "").replace(".", "p")


def method_run_name(method: str, alpha: float = ALPHA) -> str:
    """Keep main paths stable while isolating non-default CPDPO ablations."""

    if method not in ALL_METHODS:
        raise ValueError(f"Unknown experiment method: {method}")
    if method != "cpdpo":
        return method
    return "cpdpo" if is_main_alpha(alpha) else f"cpdpo_alpha_{alpha_tag(alpha)}"


@dataclass(frozen=True)
class CPDPOConfig:
    """Configuration whose defaults exactly match the frozen PDF specification."""

    method: str
    alpha: float = ALPHA
    epsilon: float = CALIBRATION_EPSILON
    clip_epsilon: float = 0.2
    kl_beta: float = 0.0
    responses_per_prompt: int = RESPONSES_PER_PROMPT
    reward_variant: str = "robust_margin"
    geometry_mode: str = "full"
    normalize_pair_rewards: bool = False
    log_pair_records: bool = True
    certification_warning_threshold: float = 0.10
    certification_warning_patience: int = 3

    def __post_init__(self) -> None:
        if self.method not in PAIR_METHODS:
            raise ValueError(f"Pair trainer method must be one of {sorted(PAIR_METHODS)}")
        if self.responses_per_prompt != RESPONSES_PER_PROMPT:
            raise ValueError("CPDPO v1 requires exactly two responses per prompt")
        if self.reward_variant not in {"robust_margin", "sign_only"}:
            raise ValueError("reward_variant must be robust_margin or sign_only")
        if self.geometry_mode not in {"full", "unit"}:
            raise ValueError("geometry_mode must be full or unit")
        if self.method == "pairppo" and (
            self.reward_variant != "robust_margin" or self.geometry_mode != "full"
        ):
            raise ValueError("PairPPO is the q=0 control and does not accept CPDPO ablation flags")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must be in (0, 1)")
        if self.kl_beta < 0.0:
            raise ValueError("kl_beta cannot be negative")
        if self.normalize_pair_rewards:
            raise ValueError("pair-reward normalization is disabled in the v1 main experiment")
        if not self.log_pair_records:
            raise ValueError("pair-record persistence is required in the v1 main experiment")
        if not 0.0 <= self.certification_warning_threshold <= 1.0:
            raise ValueError("certification warning threshold must be in [0, 1]")
        if self.certification_warning_patience < 1:
            raise ValueError("certification warning patience must be positive")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CPDPOConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown CPDPO configuration fields: {', '.join(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
