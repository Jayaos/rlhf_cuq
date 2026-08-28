"""Conformal Pair-Difference Policy Optimization (CPDPO)."""

from src.cpdpo.spec import (
    ALPHA,
    CALIBRATION_EPSILON,
    RESPONSES_PER_PROMPT,
    RIDGE_SCALE,
    ZERO_TRACE_RIDGE,
    CPDPOConfig,
)

__all__ = [
    "ALPHA",
    "CALIBRATION_EPSILON",
    "RESPONSES_PER_PROMPT",
    "RIDGE_SCALE",
    "ZERO_TRACE_RIDGE",
    "CPDPOConfig",
]
