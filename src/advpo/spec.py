"""Validated configuration for the paper-equation AdvPO branch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ADVPO_PAPER_B_GRID = (1.0, 5.0, 10.0, 15.0)
ADVPO_EPSILON = 1.0e-12


def number_tag(value: float) -> str:
    parsed = float(value)
    if not parsed > 0.0:
        raise ValueError("value must be positive")
    return format(parsed, ".15g").replace("-", "m").replace("+", "").replace(".", "p")


def advpo_run_name(confidence_radius_squared: float) -> str:
    """Give each paper hyperparameter B=b^2 an immutable run identity."""

    return f"advpo_B_{number_tag(confidence_radius_squared)}"


@dataclass(frozen=True)
class AdvPOConfig:
    """AdvPO settings disclosed by the paper or explicitly recorded here."""

    confidence_radius_squared: float
    method: str = "advpo"
    kl_beta: float = 0.0
    epsilon: float = ADVPO_EPSILON
    dynamic_reward_scaling: bool = True
    responses_per_prompt: int = 2
    adversarial_batch_responses: int = 64

    def __post_init__(self) -> None:
        if self.method != "advpo":
            raise ValueError("AdvPO method identity must be advpo")
        if not self.confidence_radius_squared > 0.0:
            raise ValueError("AdvPO B=b^2 must be positive")
        if self.kl_beta < 0.0:
            raise ValueError("kl_beta cannot be negative")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if not self.dynamic_reward_scaling:
            raise ValueError("Paper-equation AdvPO requires dynamic reward scaling")
        if self.responses_per_prompt != 2:
            raise ValueError("The fair comparison requires two current responses per prompt")
        if self.adversarial_batch_responses < 2 or self.adversarial_batch_responses % 2:
            raise ValueError("AdvPO adversarial batches must contain a positive number of prompt pairs")

    @property
    def confidence_radius(self) -> float:
        return self.confidence_radius_squared**0.5

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "AdvPOConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown AdvPO configuration fields: {', '.join(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
