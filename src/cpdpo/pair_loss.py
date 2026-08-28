"""Exact CPDPO/PairPPO clipped token surrogate and separate KL term."""

from __future__ import annotations

from typing import Any

import torch


def pairwise_clipped_loss(
    *,
    logprobs_a: torch.Tensor,
    logprobs_b: torch.Tensor,
    old_logprobs_a: torch.Tensor,
    old_logprobs_b: torch.Tensor,
    ref_logprobs_a: torch.Tensor,
    ref_logprobs_b: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    pair_rewards: torch.Tensor,
    clip_epsilon: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute response-sum, orientation-average, prompt-pair-mean loss."""

    batch = pair_rewards.shape[0]
    expected_a = (batch, logprobs_a.shape[1])
    expected_b = (batch, logprobs_b.shape[1])
    for name, tensor, shape in (
        ("old_logprobs_a", old_logprobs_a, expected_a),
        ("ref_logprobs_a", ref_logprobs_a, expected_a),
        ("mask_a", mask_a, expected_a),
        ("old_logprobs_b", old_logprobs_b, expected_b),
        ("ref_logprobs_b", ref_logprobs_b, expected_b),
        ("mask_b", mask_b, expected_b),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match {shape}")
    rewards = pair_rewards.detach().to(logprobs_a)
    coefficients_a = rewards[:, None]
    coefficients_b = -rewards[:, None]

    def orientation(new, old, coefficient, mask):
        ratio = torch.exp(new - old)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate = torch.minimum(ratio * coefficient, clipped_ratio * coefficient)
        boolean_mask = mask.bool()
        summed = (surrogate * boolean_mask).sum(dim=1)
        clip_fraction = (((ratio != clipped_ratio) & boolean_mask).sum() / boolean_mask.sum().clamp_min(1)).float()
        return summed, clip_fraction, ratio

    sum_a, clip_fraction_a, ratio_a = orientation(logprobs_a, old_logprobs_a, coefficients_a, mask_a)
    sum_b, clip_fraction_b, ratio_b = orientation(logprobs_b, old_logprobs_b, coefficients_b, mask_b)
    pair_objective = (0.5 * (sum_a + sum_b)).mean()

    def nonnegative_kl(new, reference, mask):
        log_ratio = new - reference
        per_token = torch.exp(log_ratio) - 1.0 - log_ratio
        return (per_token * mask.bool()).sum(dim=1)

    kl_per_pair = 0.5 * (
        nonnegative_kl(logprobs_a, ref_logprobs_a, mask_a)
        + nonnegative_kl(logprobs_b, ref_logprobs_b, mask_b)
    )
    kl_loss = kl_per_pair.mean()
    loss = -pair_objective + kl_beta * kl_loss
    stats = {
        "loss/total": loss.detach(),
        "loss/pair": (-pair_objective).detach(),
        "loss/kl": kl_loss.detach(),
        "policy/pair_objective": pair_objective.detach(),
        "policy/clip_fraction_a": clip_fraction_a.detach(),
        "policy/clip_fraction_b": clip_fraction_b.detach(),
        "policy/ratio_mean_a": ((ratio_a * mask_a).sum() / mask_a.sum().clamp_min(1)).detach(),
        "policy/ratio_mean_b": ((ratio_b * mask_b).sum() / mask_b.sum().clamp_min(1)).detach(),
    }
    return loss, stats
