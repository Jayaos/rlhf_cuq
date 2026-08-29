"""Common response formatting, KL, and aggregation for checkpoint evaluation."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable
from pathlib import Path

from src.cpdpo.artifacts import canonical_json_hash


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


def load_validated_checkpoint_responses(
    path: str | Path,
    *,
    prompts: list[dict],
    prompt_id_field: str,
    method: str,
    seed: int,
    rollout_step: int,
    optimizer_step: int,
    checkpoint: str | Path,
    checkpoint_fingerprint: str,
) -> list[dict]:
    """Load a complete persisted response set and reject mismatched resume data."""

    source = Path(path)
    try:
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load persisted evaluation responses: {source}") from exc
    if len(records) != len(prompts):
        raise ValueError(
            f"Persisted evaluation response count mismatch in {source}: "
            f"expected {len(prompts)}, found {len(records)}"
        )

    checkpoint_text = str(checkpoint)
    response_ids = []
    for index, (record, prompt) in enumerate(zip(records, prompts)):
        expected = {
            "prompt_id": prompt[prompt_id_field],
            "instruction": prompt["instruction"],
            "input": prompt["input"],
            "method": method,
            "seed": seed,
            "rollout_step": rollout_step,
            "optimizer_step": optimizer_step,
            "policy_checkpoint": checkpoint_text,
            "policy_checkpoint_fingerprint": checkpoint_fingerprint,
        }
        mismatches = {key: (value, record.get(key)) for key, value in expected.items() if record.get(key) != value}
        if mismatches:
            raise ValueError(f"Persisted evaluation response {index} metadata mismatch in {source}: {mismatches}")
        token_ids = record.get("response_token_ids")
        if not isinstance(record.get("output"), str) or not isinstance(token_ids, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in token_ids
        ):
            raise ValueError(f"Persisted evaluation response {index} has invalid output tokens in {source}")
        if record.get("generated_tokens") != len(token_ids):
            raise ValueError(f"Persisted evaluation response {index} has an invalid token count in {source}")
        sampled_kl = record.get("sampled_kl")
        if not isinstance(sampled_kl, (int, float)) or isinstance(sampled_kl, bool) or not math.isfinite(sampled_kl):
            raise ValueError(f"Persisted evaluation response {index} has invalid sampled KL in {source}")
        expected_response_id = canonical_json_hash(
            [method, seed, rollout_step, prompt[prompt_id_field], token_ids]
        )
        if record.get("response_id") != expected_response_id:
            raise ValueError(f"Persisted evaluation response {index} has an invalid response ID in {source}")
        response_ids.append(expected_response_id)
    if len(response_ids) != len(set(response_ids)):
        raise ValueError(f"Persisted evaluation response IDs are not unique in {source}")
    return records


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
