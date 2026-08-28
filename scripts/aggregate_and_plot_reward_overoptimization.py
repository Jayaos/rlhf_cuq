#!/usr/bin/env python
"""Aggregate checkpoint metrics and create the two frozen target figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cpdpo.experiment import validate_policy_quality_record
from src.cpdpo.spec import ALL_METHODS


def mean_se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def load_records(root: Path, split: str) -> list[dict]:
    records = []
    for path in sorted(root.glob(f"seed_*/*/evaluation/{split}/checkpoint_metrics.jsonl")):
        run_metadata = json.loads((path.parents[2] / "run_metadata.json").read_text(encoding="utf-8"))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    validate_policy_quality_record(record)
                    record["_prompt_schedule_sha256"] = run_metadata["prompt_schedule_sha256"]
                    record["_prompt_id_sequence_sha256"] = run_metadata["prompt_id_sequence_sha256"]
                    records.append(record)
    if not records:
        raise FileNotFoundError(f"No checkpoint metrics found under {root}")
    methods = {record["method"] for record in records}
    if methods != ALL_METHODS:
        raise ValueError(f"Expected {sorted(ALL_METHODS)}, found {sorted(methods)}")
    return records


def aggregate(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["method"], int(record["rollout_step"]))].append(record)
    seeds_by_method = {
        method: {int(record["seed"]) for record in records if record["method"] == method}
        for method in ALL_METHODS
    }
    if len({tuple(sorted(values)) for values in seeds_by_method.values()}) != 1:
        raise ValueError(f"Methods do not share identical seed sets: {seeds_by_method}")
    if min(map(len, seeds_by_method.values())) < 3:
        raise ValueError("Reportable aggregation requires at least three seeds")
    steps_by_method = {
        method: {int(record["rollout_step"]) for record in records if record["method"] == method}
        for method in ALL_METHODS
    }
    if len({tuple(sorted(values)) for values in steps_by_method.values()}) != 1:
        raise ValueError(f"Methods do not share an identical checkpoint schedule: {steps_by_method}")
    for seed in sorted({int(record["seed"]) for record in records}):
        selected = [record for record in records if int(record["seed"]) == seed]
        schedules = {
            (record["_prompt_schedule_sha256"], record["_prompt_id_sequence_sha256"])
            for record in selected
        }
        if len(schedules) != 1:
            raise ValueError(f"Methods do not share the same training prompt schedule for seed {seed}")
    for field in (
        "initial_policy_fingerprint",
        "reference_policy_fingerprint",
        "proxy_rm_fingerprint",
        "gold_rm_fingerprint",
        "evaluation_manifest_sha256",
        "evaluation_prompt_ids_sha256",
    ):
        values = {record[field] for record in records}
        if len(values) != 1:
            raise ValueError(f"Runs do not share one {field}: {sorted(values)}")
    for record in records:
        if int(record["rollout_step"]) == 0 and (
            record["policy_checkpoint_fingerprint"] != record["initial_policy_fingerprint"]
        ):
            raise ValueError("Checkpoint zero is not the shared initial policy")
    budget_groups = defaultdict(list)
    for record in records:
        budget_groups[(int(record["seed"]), int(record["rollout_step"]))].append(record)
    for (seed, step), rows in budget_groups.items():
        for field in ("generated_responses", "proxy_rm_calls"):
            values = {int(row[field]) for row in rows}
            if len(values) != 1:
                raise ValueError(
                    f"Methods have unequal {field} at seed={seed}, rollout_step={step}: {sorted(values)}"
                )

    output = []
    fields = ("proxy_reward_mean", "gold_reward_mean", "eval_kl_mean", "sqrt_eval_kl")
    for (method, step), rows in sorted(groups.items()):
        if len({row["seed"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate seed record for {method} rollout {step}")
        aggregated = {"method": method, "rollout_step": step, "n_seeds": len(rows)}
        for field in fields:
            mean, se = mean_se([float(row[field]) for row in rows])
            aggregated[f"{field}_across_seeds"] = mean
            aggregated[f"{field}_se"] = se
        output.append(aggregated)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def plot(rows: list[dict], figures: Path) -> None:
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    colors = {"ppo": "#4C78A8", "pairppo": "#F58518", "cpdpo": "#54A24B"}
    reward_styles = {"proxy": "--", "gold": "-"}

    def make_figure(x_field: str, x_label: str, filename: str):
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for method in ("ppo", "pairppo", "cpdpo"):
            selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["rollout_step"])
            x = [row[x_field] for row in selected]
            for reward in ("proxy", "gold"):
                y_field = f"{reward}_reward_mean_across_seeds"
                se_field = f"{reward}_reward_mean_se"
                y = [row[y_field] for row in selected]
                se = [row[se_field] for row in selected]
                axis.plot(x, y, color=colors[method], linestyle=reward_styles[reward], label=f"{method.upper()} {reward}")
                axis.fill_between(x, [a - b for a, b in zip(y, se)], [a + b for a, b in zip(y, se)], color=colors[method], alpha=0.12)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Reward score")
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        for suffix in ("pdf", "png"):
            figure.savefig(figures / f"{filename}.{suffix}", dpi=200)
        plt.close(figure)

    make_figure("rollout_step", "Policy optimization (rollout) step", "figure_2a_reward_vs_training")
    make_figure("sqrt_eval_kl_across_seeds", r"$\sqrt{\mathrm{evaluation\ KL}}$", "figure_2b_reward_vs_kl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/reward_overoptimization")
    parser.add_argument("--split", choices=["D_rl_val_prompts", "D_rl_test_prompts"], default="D_rl_val_prompts")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    records = load_records(root, args.split)
    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    evaluation_dir = root / "evaluations"
    write_jsonl(evaluation_dir / "checkpoint_metrics.jsonl", public_records)
    write_csv(evaluation_dir / "checkpoint_metrics.csv", public_records)
    aggregated = aggregate(records)
    aggregate_dir = root / "aggregated"
    write_csv(aggregate_dir / "mean_by_checkpoint.csv", aggregated)
    write_csv(
        aggregate_dir / "mean_by_kl.csv",
        sorted(aggregated, key=lambda row: (row["method"], row["sqrt_eval_kl_across_seeds"])),
    )
    plot(aggregated, root / "figures")
    print(f"PASS aggregated {len(records)} checkpoint records across {aggregated[0]['n_seeds']} seeds")
    print(f"PASS figures: {root / 'figures'}")


if __name__ == "__main__":
    main()
