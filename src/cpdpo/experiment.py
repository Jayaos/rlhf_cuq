"""Dependency-free fairness checks and checkpoint record helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.cpdpo.spec import ALL_METHODS, TRAINING_METHODS


FORBIDDEN_POLICY_QUALITY_FIELDS = frozenset(
    {"pair_reward", "pair_reward_mean", "robust_margin", "gamma", "internal_training_reward"}
)


@dataclass(frozen=True)
class TrainingBudget:
    method: str
    prompt_pairs_per_rollout: int
    response_count_per_rollout: int
    proxy_rm_calls_per_rollout: int
    trainer_rollout_units: int
    trainer_batch_units: int
    optimizer_updates_per_rollout: int


def resolve_training_budget(
    method: str,
    *,
    prompts_per_rollout: int,
    pair_batch_size: int,
    ppo_epochs: int,
) -> TrainingBudget:
    if method not in TRAINING_METHODS:
        raise ValueError(f"Unknown method: {method}")
    if prompts_per_rollout < 1 or pair_batch_size < 1 or ppo_epochs < 1:
        raise ValueError("Budget values must be positive")
    responses = 2 * prompts_per_rollout
    if method in {"ppo", "cpdpo_v2"}:
        rollout_units = responses
        batch_units = 2 * pair_batch_size
    else:
        rollout_units = prompts_per_rollout
        batch_units = pair_batch_size
    if rollout_units % batch_units:
        raise ValueError("Rollout units must be divisible by batch units")
    return TrainingBudget(
        method=method,
        prompt_pairs_per_rollout=prompts_per_rollout,
        response_count_per_rollout=responses,
        proxy_rm_calls_per_rollout=responses,
        trainer_rollout_units=rollout_units,
        trainer_batch_units=batch_units,
        optimizer_updates_per_rollout=(rollout_units // batch_units) * ppo_epochs,
    )


def assert_equal_method_budgets(budgets: list[TrainingBudget]) -> None:
    if {budget.method for budget in budgets} != ALL_METHODS:
        raise ValueError("Budget audit requires exactly PPO, PairPPO, and CPDPO")
    comparable = {
        (
            budget.prompt_pairs_per_rollout,
            budget.response_count_per_rollout,
            budget.proxy_rm_calls_per_rollout,
            budget.optimizer_updates_per_rollout,
        )
        for budget in budgets
    }
    if len(comparable) != 1:
        raise ValueError(f"Method budgets are not equal: {[asdict(value) for value in budgets]}")


def validate_policy_quality_record(record: dict) -> None:
    forbidden = FORBIDDEN_POLICY_QUALITY_FIELDS.intersection(record)
    if forbidden:
        raise ValueError(
            "Internal pair rewards cannot be plotted as policy quality: " + ", ".join(sorted(forbidden))
        )
    required = {"method", "seed", "rollout_step", "proxy_reward_mean", "gold_reward_mean", "eval_kl_mean"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Checkpoint record is missing: {', '.join(missing)}")
