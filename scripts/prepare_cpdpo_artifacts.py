#!/usr/bin/env python
"""Build frozen pair geometry and calibration from the verified split manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.cpdpo.artifacts import (
    atomic_write_json,
    git_revision,
    model_fingerprint,
    sha256_file,
    tokenizer_fingerprint,
)
from src.cpdpo.geometry import build_pair_geometry, calibration_scores, conformal_quantile
from src.cpdpo.reward_features import load_proxy_feature_scorer
from src.cpdpo.spec import (
    ALPHA,
    CALIBRATION_EPSILON,
    RIDGE_SCALE,
    ZERO_TRACE_RIDGE,
    is_main_alpha,
)
from src.data_utils.split_manifest import load_split_records, verify_split_manifest
from src.reward_modeling.training.label_noise import load_optional_label_noise_metadata


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", required=True)
    value.add_argument("--proxy-rm", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--geometry-mode", choices=["full", "unit"], default="full")
    value.add_argument(
        "--alpha",
        type=alpha_value,
        default=ALPHA,
        help="Conformal miscoverage level; 0.10 is the frozen main run and other values are ablations",
    )
    value.add_argument(
        "--max-rm-pairs",
        type=positive_int,
        help="Use only this manifest-order prefix of D_rm_train and mark the artifacts smoke-only",
    )
    value.add_argument(
        "--max-cal-pairs",
        type=positive_int,
        help="Use only this manifest-order prefix of D_cal and mark the artifacts smoke-only",
    )
    value.add_argument("--overwrite", action="store_true")
    return value


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def alpha_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("alpha must be in (0, 1)")
    return parsed


def prompt_text(row: dict) -> str:
    return f"{row['instruction']}\n{row['input']}" if row["input"] else row["instruction"]


def score_pair_rows(scorer, rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [prompt_text(row) for row in batch for _ in range(2)]
        outputs = [answer for row in batch for answer in row["answers"]]
        if any(len(row["answers"]) != 2 for row in batch):
            raise ValueError("Every preference record must contain exactly two answers")
        # Static preference records are represented exactly as in RM
        # training/scoring: one terminal Pythia EOS token follows each answer.
        rewards, features = scorer.score(prompts, outputs, evaluation=True)
        yield batch, rewards[0::2], rewards[1::2], features[0::2] - features[1::2]


def atomic_torch_save(path: Path, value, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    args = parser().parse_args()
    manifest = Path(args.manifest).resolve()
    proxy_rm = Path(args.proxy_rm).resolve()
    output = Path(args.output_dir).resolve()
    verify_split_manifest(manifest)
    rm_rows = load_split_records(manifest, "D_rm_train", expected_kind="preference")
    cal_rows = load_split_records(manifest, "D_cal", expected_kind="preference")
    available_rm = len(rm_rows)
    available_cal = len(cal_rows)
    artifact_scope = "smoke" if args.max_rm_pairs is not None or args.max_cal_pairs is not None else "scientific"
    if args.max_rm_pairs is not None:
        rm_rows = rm_rows[: args.max_rm_pairs]
    if args.max_cal_pairs is not None:
        cal_rows = cal_rows[: args.max_cal_pairs]
    if artifact_scope == "smoke":
        print(
            "WARNING: building smoke-only CPDPO artifacts from deterministic manifest-order prefixes; "
            "these artifacts are not valid for scientific training"
        )
    scorer = load_proxy_feature_scorer(str(proxy_rm), device=args.device, batch_size=args.batch_size * 2)

    geometry_path = output / "pair_geometry.pt"
    if args.geometry_mode == "full":
        def differences():
            for _batch, _reward_a, _reward_b, difference in score_pair_rows(scorer, rm_rows, args.batch_size):
                yield difference

        geometry, gram = build_pair_geometry(
            differences(), ridge_scale=RIDGE_SCALE, zero_trace_ridge=ZERO_TRACE_RIDGE
        )
        geometry_state = geometry.state_dict()
    else:
        geometry = None
        gram = None
        geometry_state = {
            "schema_version": "1.0.0",
            "geometry": "unit",
            "geometry_mode": "unit",
            "dimension": int(scorer.model.config.hidden_size),
            "n_rm": 0,
            "dtype": "not_applicable",
        }
    geometry_state["artifact_scope"] = artifact_scope
    atomic_torch_save(geometry_path, geometry_state, args.overwrite)
    geometry_fingerprint = sha256_file(geometry_path)

    common = {
        "schema_version": "1.0.0",
        "proxy_rm_path": str(proxy_rm),
        "proxy_rm_fingerprint": model_fingerprint(proxy_rm),
        "proxy_rm_label_noise": load_optional_label_noise_metadata(proxy_rm),
        "tokenizer_fingerprint": tokenizer_fingerprint(proxy_rm),
        "data_manifest_path": str(manifest),
        "data_manifest_sha256": sha256_file(manifest),
        "feature_extraction": "input_to_GPTNeoXRewardModel.out_proj",
        "pair_orientation": "source_answer_a_minus_source_answer_b",
        "artifact_scope": artifact_scope,
        "data_selection": (
            "complete_manifest_roles"
            if artifact_scope == "scientific"
            else "deterministic_manifest_order_prefix_for_smoke_only"
        ),
        "available_n_rm": available_rm,
        "available_n_cal": available_cal,
        "code_revision": git_revision(ROOT),
    }
    geometry_metadata = {
        **common,
        "artifact": "pair_geometry.pt",
        "geometry_fingerprint": geometry_fingerprint,
        "source_role": "D_rm_train",
        "n_rm": geometry.n_rm if geometry is not None else 0,
        "dimension": geometry.dimension if geometry is not None else geometry_state["dimension"],
        "accumulation_dtype": "float64",
        "geometry": args.geometry_mode,
        "normalization": "one_over_n_rm",
        "ridge_scale": RIDGE_SCALE,
        "zero_trace_ridge": ZERO_TRACE_RIDGE,
        "ridge": geometry.ridge if geometry is not None else None,
        "gram_trace": float(torch.trace(gram).item()) if gram is not None else None,
        "solve": "cholesky_triangular" if geometry is not None else "u_equals_one_ablation",
    }
    atomic_write_json(output / "pair_geometry_metadata.json", geometry_metadata, overwrite=args.overwrite)

    labels, margins, uncertainties = [], [], []
    for batch, reward_a, reward_b, difference in score_pair_rows(scorer, cal_rows, args.batch_size):
        labels.append(torch.tensor([1 if row["preference"] == 0 else -1 for row in batch], dtype=torch.float64))
        margins.append((reward_a - reward_b).to(torch.float64))
        uncertainties.append(
            geometry.uncertainty(difference.to(torch.float64))
            if geometry is not None
            else torch.ones_like(reward_a, dtype=torch.float64)
        )
    labels_tensor = torch.cat(labels)
    margins_tensor = torch.cat(margins)
    uncertainty_tensor = torch.cat(uncertainties)
    scores = calibration_scores(
        labels_tensor, margins_tensor, uncertainty_tensor, epsilon=CALIBRATION_EPSILON
    )
    q_alpha, rank = conformal_quantile(scores, alpha=args.alpha)
    scores_path = output / "calibration_scores.pt"
    atomic_torch_save(
        scores_path,
        {
            "schema_version": "1.0.0",
            "artifact_scope": artifact_scope,
            "labels": labels_tensor,
            "margins": margins_tensor,
            "uncertainties": uncertainty_tensor,
            "scores": scores,
        },
        args.overwrite,
    )
    calibration = {
        **common,
        "artifact": "conformal_calibration.json",
        "source_role": "D_cal",
        "geometry_fingerprint": geometry_fingerprint,
        "geometry_mode": args.geometry_mode,
        "calibration_scores_fingerprint": sha256_file(scores_path),
        "alpha": args.alpha,
        "experiment_track": "main" if is_main_alpha(args.alpha) else "cpdpo_alpha_ablation",
        "epsilon": CALIBRATION_EPSILON,
        "n_cal": scores.numel(),
        "quantile_rank_one_based": rank,
        "quantile_index_zero_based": rank - 1,
        "quantile_convention": "higher_order_statistic_no_interpolation",
        "q_alpha": float(q_alpha.item()),
        "score_summary": {
            "min": float(scores.min().item()),
            "mean": float(scores.mean().item()),
            "median": float(scores.median().item()),
            "max": float(scores.max().item()),
            "positive_fraction": float((scores > 0).double().mean().item()),
        },
        "threshold_schedule": "fixed_for_entire_ppo_run",
    }
    atomic_write_json(output / "conformal_calibration.json", calibration, overwrite=args.overwrite)
    print(
        f"PASS geometry: {geometry_path} mode={args.geometry_mode} "
        f"scope={artifact_scope} "
        f"n={geometry.n_rm if geometry is not None else 0} "
        f"d={geometry.dimension if geometry is not None else geometry_state['dimension']} "
        f"ridge={geometry.ridge if geometry is not None else None}"
    )
    print(
        f"PASS calibration: scope={artifact_scope} alpha={args.alpha} n={scores.numel()} "
        f"rank={rank} q_alpha={q_alpha.item()}"
    )


if __name__ == "__main__":
    main()
