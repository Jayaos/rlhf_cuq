#!/usr/bin/env python
"""Build the frozen AdvPO confidence matrix from D_rm_train response features."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.advpo.geometry import build_advpo_confidence_geometry
from src.cpdpo.artifacts import (
    atomic_write_json,
    git_revision,
    model_fingerprint,
    sha256_file,
    tokenizer_fingerprint,
)
from src.cpdpo.reward_features import load_proxy_feature_scorer
from src.data_utils.split_manifest import load_split_records, verify_split_manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--proxy-rm", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ridge-lambda",
        type=float,
        required=True,
        help="Explicit M_D ridge; the AdvPO paper does not publish a numeric value",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-rm-pairs",
        type=int,
        help="Use a manifest-order prefix and tag the matrix smoke-only",
    )
    return parser.parse_args()


def prompt_text(row: dict) -> str:
    return f"{row['instruction']}\n{row['input']}" if row["input"] else row["instruction"]


def atomic_torch_save(path: Path, value) -> None:
    if path.exists():
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
    args = arguments()
    if args.batch_size < 1 or args.ridge_lambda <= 0.0:
        raise ValueError("batch-size and ridge-lambda must be positive")
    if args.max_rm_pairs is not None and args.max_rm_pairs < 1:
        raise ValueError("max-rm-pairs must be positive")
    manifest = Path(args.manifest).resolve()
    proxy_rm = Path(args.proxy_rm).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty AdvPO confidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    verify_split_manifest(manifest)
    rows = load_split_records(manifest, "D_rm_train", expected_kind="preference")
    available_pairs = len(rows)
    artifact_scope = "smoke" if args.max_rm_pairs is not None else "scientific"
    if args.max_rm_pairs is not None:
        rows = rows[: args.max_rm_pairs]
        print("WARNING: building a smoke-only AdvPO confidence matrix")
    scorer = load_proxy_feature_scorer(
        str(proxy_rm), device=args.device, batch_size=2 * args.batch_size
    )

    def feature_batches():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            if any(len(row["answers"]) != 2 for row in batch):
                raise ValueError("Every D_rm_train preference row must contain exactly two answers")
            prompts = [prompt_text(row) for row in batch for _ in range(2)]
            outputs = [answer for row in batch for answer in row["answers"]]
            _rewards, features = scorer.score(prompts, outputs, evaluation=True)
            yield features

    geometry, gram_sum = build_advpo_confidence_geometry(
        feature_batches(), ridge_lambda=args.ridge_lambda
    )
    confidence_path = output / "confidence_matrix.pt"
    state = geometry.state_dict()
    state["artifact_scope"] = artifact_scope
    atomic_torch_save(confidence_path, state)
    metadata = {
        "schema_version": "1.0.0",
        "method": "advpo",
        "paper_equation": "M_D=lambda_I_plus_sum_over_pairs_and_both_responses_e_eT",
        "artifact": "confidence_matrix.pt",
        "confidence_fingerprint": sha256_file(confidence_path),
        "artifact_scope": artifact_scope,
        "source_role": "D_rm_train",
        "data_selection": (
            "complete_manifest_role"
            if artifact_scope == "scientific"
            else "deterministic_manifest_order_prefix_for_smoke_only"
        ),
        "available_preference_pairs": available_pairs,
        "preference_pairs": len(rows),
        "feature_terms_per_preference": 2,
        "n_responses": geometry.n_responses,
        "dimension": geometry.dimension,
        "normalization": "unnormalized_sum",
        "accumulation_dtype": "float64",
        "ridge_lambda": geometry.ridge_lambda,
        "ridge_value_disclosed_by_paper": False,
        "gram_trace": float(torch.trace(gram_sum).item()),
        "solve": "cholesky_triangular_no_explicit_inverse",
        "feature_extraction": "input_to_GPTNeoXRewardModel.out_proj",
        "proxy_rm_path": str(proxy_rm),
        "proxy_rm_fingerprint": model_fingerprint(proxy_rm),
        "tokenizer_fingerprint": tokenizer_fingerprint(proxy_rm),
        "data_manifest_path": str(manifest),
        "data_manifest_sha256": sha256_file(manifest),
        "gold_access": False,
        "code_revision": git_revision(ROOT),
    }
    atomic_write_json(output / "confidence_matrix_metadata.json", metadata)
    print(
        f"PASS AdvPO confidence: {confidence_path} scope={artifact_scope} "
        f"responses={geometry.n_responses} d={geometry.dimension} ridge={geometry.ridge_lambda}"
    )


if __name__ == "__main__":
    main()
