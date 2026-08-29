"""Common response formatting, KL, and aggregation for checkpoint evaluation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable


ALPACA_PROMPT_NOINPUTS = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
)
ALPACA_PROMPT_INPUTS = (
    "Below is an instruction that describes a task, paired with an input that provides further "
    "context. Write a response that appropriately completes the request.\n\n### Instruction:\n"
    "{instruction}\n\n### Input:\n{input}\n\n### Response:"
)


def format_alpaca_gold_sample(instruction: str, input_text: str, output: str) -> str:
    template = ALPACA_PROMPT_INPUTS if input_text else ALPACA_PROMPT_NOINPUTS
    return template.format(instruction=instruction, input=input_text) + output


def hydra_policy_logits(policy, input_ids, **forward_kwargs):
    """Return policy LM logits without evaluating the unused PPO value head."""
    return policy.base_model(input_ids, **forward_kwargs).logits


def mean_and_sample_std(values: Iterable[float]) -> tuple[float, float]:
    items = [float(value) for value in values]
    if not items or not all(math.isfinite(value) for value in items):
        raise ValueError("Metric values must be nonempty and finite")
    return statistics.fmean(items), statistics.stdev(items) if len(items) > 1 else 0.0


def checkpoint_summary(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty checkpoint")
    proxy_mean, proxy_std = mean_and_sample_std(row["proxy_reward"] for row in records)
    gold_mean, gold_std = mean_and_sample_std(row["gold_reward"] for row in records)
    kl_mean, kl_std = mean_and_sample_std(row["sampled_kl"] for row in records)
    return {
        "proxy_reward_mean": proxy_mean,
        "proxy_reward_std": proxy_std,
        "gold_reward_mean": gold_mean,
        "gold_reward_std": gold_std,
        "eval_kl_mean": kl_mean,
        "eval_kl_std": kl_std,
        "sqrt_eval_kl": math.sqrt(max(kl_mean, 0.0)),
    }
