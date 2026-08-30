#!/usr/bin/env python
"""Aggregate checkpoint metrics and create the two frozen target figures."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
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
from src.cpdpo.spec import ALPHA, ALL_METHODS, alpha_tag, is_main_alpha, method_run_name


def require_plotting_dependency() -> None:
    if importlib.util.find_spec("matplotlib") is None:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib==3.7.2. Install the constrained runtime dependency "
            "before rerunning; no plot artifacts have been written by this invocation."
        )


def mean_se(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se


def load_records(
    root: Path,
    split: str,
    *,
    cpdpo_alpha: float = ALPHA,
    include_cpdpo_v2: bool = False,
) -> list[dict]:
    records = []
    run_names = {
        "ppo": method_run_name("ppo"),
        "pairppo": method_run_name("pairppo"),
        "cpdpo": method_run_name("cpdpo", cpdpo_alpha),
    }
    if include_cpdpo_v2:
        run_names["cpdpo_v2"] = method_run_name("cpdpo_v2", cpdpo_alpha)
    paths = [
        seed_dir / run_name / "evaluation" / split / "checkpoint_metrics.jsonl"
        for seed_dir in sorted(root.glob("seed_*"))
        for run_name in run_names.values()
        if (seed_dir / run_name / "evaluation" / split / "checkpoint_metrics.jsonl").is_file()
    ]
    for path in paths:
        run_metadata = json.loads((path.parents[2] / "run_metadata.json").read_text(encoding="utf-8"))
        method = run_metadata["method"]
        if path.parents[2].name != run_names.get(method):
            raise ValueError(f"Unexpected run directory for {method}: {path.parents[2]}")
        recorded_alpha = run_metadata.get("cpdpo_alpha")
        if method in {"cpdpo", "cpdpo_v2"} and recorded_alpha is None:
            recorded_alpha = (run_metadata.get("pair_method") or {}).get("alpha", ALPHA)
            if method == "cpdpo_v2":
                recorded_alpha = (run_metadata.get("cpdpo_v2") or {}).get("alpha", ALPHA)
        if method in {"cpdpo", "cpdpo_v2"} and float(recorded_alpha) != float(cpdpo_alpha):
            raise ValueError(
                f"Requested CPDPO alpha {cpdpo_alpha}, but {path.parents[2]} records {recorded_alpha}"
            )
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    validate_policy_quality_record(record)
                    record["_prompt_schedule_sha256"] = run_metadata["prompt_schedule_sha256"]
                    record["_prompt_id_sequence_sha256"] = run_metadata["prompt_id_sequence_sha256"]
                    record["_cpdpo_alpha"] = recorded_alpha if method in {"cpdpo", "cpdpo_v2"} else None
                    records.append(record)
    if not records:
        raise FileNotFoundError(f"No checkpoint metrics found under {root}")
    methods = {record["method"] for record in records}
    expected_methods = set(ALL_METHODS) | ({"cpdpo_v2"} if include_cpdpo_v2 else set())
    if methods != expected_methods:
        raise ValueError(f"Expected {sorted(expected_methods)}, found {sorted(methods)}")
    return records


def aggregate(records: list[dict], *, minimum_seeds: int = 3) -> list[dict]:
    if minimum_seeds < 1:
        raise ValueError("minimum_seeds must be positive")
    groups = defaultdict(list)
    for record in records:
        groups[(record["method"], int(record["rollout_step"]))].append(record)
    methods = {record["method"] for record in records}
    if not set(ALL_METHODS).issubset(methods) or not methods.issubset(set(ALL_METHODS) | {"cpdpo_v2"}):
        raise ValueError(f"Unsupported comparison method set: {sorted(methods)}")
    seeds_by_method = {
        method: {int(record["seed"]) for record in records if record["method"] == method}
        for method in methods
    }
    if len({tuple(sorted(values)) for values in seeds_by_method.values()}) != 1:
        raise ValueError(f"Methods do not share identical seed sets: {seeds_by_method}")
    if min(map(len, seeds_by_method.values())) < minimum_seeds:
        raise ValueError(f"Aggregation requires at least {minimum_seeds} seeds")
    steps_by_method = {
        method: {int(record["rollout_step"]) for record in records if record["method"] == method}
        for method in methods
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
        alpha_values = {row.get("_cpdpo_alpha") for row in rows}
        if len(alpha_values) != 1:
            raise ValueError(f"Rows disagree on CPDPO alpha for {method} rollout {step}")
        aggregated = {
            "method": method,
            "rollout_step": step,
            "n_seeds": len(rows),
            "cpdpo_alpha": next(iter(alpha_values)),
        }
        for field in fields:
            mean, se = mean_se([float(row[field]) for row in rows])
            aggregated[f"{field}_across_seeds"] = mean
            aggregated[f"{field}_se"] = se
        output.append(aggregated)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    # Evaluation records intentionally carry method-specific provenance.  In
    # particular, CPDPO rows include alpha/run-track fields that are absent
    # from the PPO and PairPPO controls.  Build a stable union instead of
    # assuming that the first row defines the complete schema.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def plot(
    rows: list[dict],
    figures: Path,
    *,
    show_uncertainty: bool = True,
    title: str | None = None,
    cpdpo_alpha: float = ALPHA,
    training_filename: str = "figure_2a_reward_vs_training",
    kl_filename: str = "figure_2b_reward_vs_kl",
) -> None:
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    colors = {
        "ppo": "#4C78A8",
        "pairppo": "#F58518",
        "cpdpo": "#54A24B",
        "cpdpo_v2": "#B279A2",
    }
    reward_styles = {"proxy": "--", "gold": "-"}

    def make_figure(x_field: str, x_label: str, filename: str):
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        methods = [method for method in ("ppo", "pairppo", "cpdpo", "cpdpo_v2") if any(
            row["method"] == method for row in rows
        )]
        for method in methods:
            selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["rollout_step"])
            x = [row[x_field] for row in selected]
            for reward in ("proxy", "gold"):
                y_field = f"{reward}_reward_mean_across_seeds"
                se_field = f"{reward}_reward_mean_se"
                y = [row[y_field] for row in selected]
                se = [row[se_field] for row in selected]
                method_label = method.upper()
                if method in {"cpdpo", "cpdpo_v2"}:
                    method_label = "CPDPOv2" if method == "cpdpo_v2" else "CPDPO"
                    if not is_main_alpha(cpdpo_alpha):
                        method_label = f"{method_label} alpha={cpdpo_alpha:g}"
                axis.plot(
                    x,
                    y,
                    color=colors[method],
                    linestyle=reward_styles[reward],
                    label=f"{method_label} {reward}",
                )
                if show_uncertainty:
                    axis.fill_between(
                        x,
                        [a - b for a, b in zip(y, se)],
                        [a + b for a, b in zip(y, se)],
                        color=colors[method],
                        alpha=0.12,
                    )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Reward score")
        if title:
            axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        for suffix in ("pdf", "png"):
            figure.savefig(figures / f"{filename}.{suffix}", dpi=200)
        plt.close(figure)

    make_figure("rollout_step", "Policy optimization (rollout) step", training_filename)
    make_figure("sqrt_eval_kl_across_seeds", r"$\sqrt{\mathrm{evaluation\ KL}}$", kl_filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/reward_overoptimization")
    parser.add_argument("--split", choices=["D_rl_val_prompts", "D_rl_test_prompts"], default="D_rl_val_prompts")
    parser.add_argument(
        "--cpdpo-alpha",
        type=float,
        default=ALPHA,
        help="Select the main CPDPO run (0.10) or a named alpha-ablation run",
    )
    parser.add_argument(
        "--include-cpdpo-v2",
        action="store_true",
        help="Add the separately trained exploratory CPDPOv2 run to the unchanged v1 controls",
    )
    parser.add_argument(
        "--diagnostic-seed",
        type=int,
        help="Create a clearly labelled single-seed diagnostic without an uncertainty band",
    )
    args = parser.parse_args()
    if not 0.0 < args.cpdpo_alpha < 1.0:
        raise ValueError("cpdpo-alpha must be in (0, 1)")
    root = Path(args.output_root).resolve()
    records = load_records(
        root,
        args.split,
        cpdpo_alpha=args.cpdpo_alpha,
        include_cpdpo_v2=args.include_cpdpo_v2,
    )
    result_root = (
        root
        if is_main_alpha(args.cpdpo_alpha)
        else root / "alpha_ablations" / f"alpha_{alpha_tag(args.cpdpo_alpha)}"
    )
    if args.include_cpdpo_v2:
        result_root = result_root / "cpdpo_v2_comparison"
    if args.diagnostic_seed is not None:
        selected = [record for record in records if int(record["seed"]) == args.diagnostic_seed]
        if not selected:
            raise ValueError(f"No evaluation records found for diagnostic seed {args.diagnostic_seed}")
        aggregated = aggregate(selected, minimum_seeds=1)
        require_plotting_dependency()
        diagnostic_dir = result_root / "diagnostics" / f"seed_{args.diagnostic_seed}"
        if diagnostic_dir.exists() and any(diagnostic_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite diagnostic directory: {diagnostic_dir}")
        public_records = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in selected
        ]
        write_jsonl(diagnostic_dir / "checkpoint_metrics.jsonl", public_records)
        write_csv(diagnostic_dir / "checkpoint_metrics.csv", public_records)
        write_csv(diagnostic_dir / "mean_by_checkpoint.csv", aggregated)
        write_csv(
            diagnostic_dir / "mean_by_sqrt_kl.csv",
            sorted(aggregated, key=lambda row: (row["method"], row["sqrt_eval_kl_across_seeds"])),
        )
        plot(
            aggregated,
            diagnostic_dir,
            show_uncertainty=False,
            title=(
                f"Single-seed diagnostic (seed {args.diagnostic_seed}; no uncertainty band"
                + ("; includes exploratory CPDPOv2" if args.include_cpdpo_v2 else "")
                + ")"
                if is_main_alpha(args.cpdpo_alpha)
                else (
                    f"Single-seed diagnostic (seed {args.diagnostic_seed}; no uncertainty band; "
                    f"CPDPO alpha={args.cpdpo_alpha:g})"
                )
            ),
            cpdpo_alpha=args.cpdpo_alpha,
            training_filename="reward_vs_rollout_step",
            kl_filename="reward_vs_sqrt_kl",
        )
        print(
            f"PASS diagnostic plots for seed {args.diagnostic_seed}: {diagnostic_dir} "
            "(single-seed result; not an across-seed estimate)"
        )
        return

    aggregated = aggregate(records)
    require_plotting_dependency()
    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    evaluation_dir = result_root / "evaluations"
    write_jsonl(evaluation_dir / "checkpoint_metrics.jsonl", public_records)
    write_csv(evaluation_dir / "checkpoint_metrics.csv", public_records)
    aggregate_dir = result_root / "aggregated"
    write_csv(aggregate_dir / "mean_by_checkpoint.csv", aggregated)
    write_csv(
        aggregate_dir / "mean_by_kl.csv",
        sorted(aggregated, key=lambda row: (row["method"], row["sqrt_eval_kl_across_seeds"])),
    )
    plot(aggregated, result_root / "figures", cpdpo_alpha=args.cpdpo_alpha)
    print(f"PASS aggregated {len(records)} checkpoint records across {aggregated[0]['n_seeds']} seeds")
    print(f"PASS figures: {result_root / 'figures'}")


if __name__ == "__main__":
    main()
