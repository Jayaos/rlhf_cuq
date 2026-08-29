#!/usr/bin/env python
"""Reconstruct pinned AlpacaFarm SFT10k and human-RM artifacts locally.

This is an offline, fail-closed adaptation of AlpacaFarm's pinned
``pretrained_models.recover_model_weights`` utility.  The upstream utility
accepts Hub model IDs only; this wrapper consumes the manifest-pinned local
weight-difference directories produced by ``scripts/download_assets.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_assets import load_assets, verify_asset
from src.cpdpo.artifacts import (
    atomic_write_json,
    canonical_json_hash,
    git_revision,
    model_fingerprint,
    sha256_file,
    tokenizer_fingerprint,
)

SFT_ASSET_NAME = "alpaca_farm_sft10k_weight_diff"
GOLD_ASSET_NAME = "alpaca_farm_gold_rm_weight_diff"
MIN_TRANSFORMERS_VERSION = "4.29.2"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-7b-hf-dir", required=True, type=Path)
    parser.add_argument("--sft10k-weight-diff", required=True, type=Path)
    parser.add_argument("--gold-rm-weight-diff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "source_manifest.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--llama-license-acknowledged",
        action="store_true",
        help="Confirm that use of the supplied LLaMA base is authorized under its license.",
    )
    return parser.parse_args()


def require_empty_output_root(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output root exists but is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty reconstruction directory: {path}")


def verify_hf_checkpoint_layout(root: Path) -> list[Path]:
    indexes = [
        path
        for path in (root / "pytorch_model.bin.index.json", root / "model.safetensors.index.json")
        if path.is_file()
    ]
    declared: set[Path] = set()
    for index in indexes:
        payload = json.loads(index.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight map: {index}")
        declared.update(root / str(name) for name in weight_map.values())
    if not declared:
        declared.update(root.glob("pytorch_model*.bin"))
        declared.update(root.glob("model*.safetensors"))
    if not declared:
        raise FileNotFoundError(f"No LLaMA weight files or weight indexes found under {root}")
    missing = [path for path in sorted(declared) if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing or empty indexed LLaMA weights: " + ", ".join(map(str, missing)))
    return sorted(declared)


def read_expected_model_sum(path: Path) -> float:
    value = float(path.read_text(encoding="utf-8").strip())
    if not math.isfinite(value):
        raise ValueError(f"Non-finite model sum in {path}: {value}")
    return value


def manifest_asset(manifest: Path, name: str) -> dict[str, Any]:
    matches = [asset for asset in load_assets(manifest) if asset.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest asset named {name!r}, found {len(matches)}")
    return matches[0]


def verify_local_asset(manifest: Path, name: str, directory: Path) -> dict[str, Any]:
    asset = manifest_asset(manifest, name)
    failures = verify_asset(asset, directory)
    if failures:
        raise ValueError(f"Pinned asset verification failed for {name}: " + "; ".join(failures))
    print(f"PASS pinned input: {name} ({len(asset['required_files'])} files)")
    return asset


def reconstruct_tuned_model(model_tuned, model_raw, *, is_reward_model: bool) -> None:
    """Apply the pinned AlpacaFarm in-place float32 addition rule."""

    state_dict_diff = model_tuned.state_dict()
    state_dict_raw = model_raw.state_dict()
    if is_reward_model:
        state_dict_raw = {f"backbone_model.{key}": value for key, value in state_dict_raw.items()}
    for key, raw_value in state_dict_raw.items():
        if key not in state_dict_diff:
            raise KeyError(f"Weight difference is missing base parameter: {key}")
        if raw_value.size() != state_dict_diff[key].size():
            # The pinned upstream implementation intentionally does not diff
            # resized token embeddings whose shapes differ.
            continue
        state_dict_diff[key].add_(raw_value)


def reconstructed_model_sum(model) -> float:
    return float(sum(parameter.sum() for parameter in model.state_dict().values()).item())


def assert_integrity(model, expected_sum_path: Path, label: str) -> tuple[float, float]:
    import numpy as np

    expected = read_expected_model_sum(expected_sum_path)
    actual = reconstructed_model_sum(model)
    if not bool(np.isclose(expected, actual)):
        raise RuntimeError(
            f"{label} model_sum integrity check failed: expected={expected}, actual={actual}. "
            "The LLaMA base or pinned difference is incompatible."
        )
    print(f"PASS {label} model_sum: expected={expected} actual={actual}")
    return expected, actual


def save_reconstruction(model, tokenizer, temporary: Path, expected_sum_path: Path) -> None:
    temporary.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    shutil.copy2(expected_sum_path, temporary / "model_sum.txt")


def main() -> None:  # noqa: C901
    args = arguments()
    if not args.llama_license_acknowledged:
        raise ValueError(
            "Reconstruction requires authorized use of the original LLaMA-7B base; "
            "pass --llama-license-acknowledged only after confirming that authorization."
        )

    llama = args.llama_7b_hf_dir.expanduser().resolve()
    sft_diff = args.sft10k_weight_diff.expanduser().resolve()
    gold_diff = args.gold_rm_weight_diff.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    for label, path in (("LLaMA base", llama), ("SFT10k difference", sft_diff), ("gold-RM difference", gold_diff)):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")
    for required in (llama / "config.json", llama / "tokenizer.model"):
        if not required.is_file():
            raise FileNotFoundError(required)
    llama_weight_files = verify_hf_checkpoint_layout(llama)
    print(
        f"PASS LLaMA checkpoint layout: {len(llama_weight_files)} weight files, "
        f"{sum(path.stat().st_size for path in llama_weight_files)} bytes"
    )
    require_empty_output_root(output_root)

    sft_asset = verify_local_asset(manifest, SFT_ASSET_NAME, sft_diff)
    gold_asset = verify_local_asset(manifest, GOLD_ASSET_NAME, gold_diff)

    import torch
    import transformers
    from packaging.version import Version

    from alpaca_farm.models.reward_model import RewardConfig, RewardModel
    from alpaca_farm.utils import stable_resize_token_embeddings_and_tokenizer

    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS_VERSION):
        raise RuntimeError(
            f"AlpacaFarm reconstruction requires transformers>={MIN_TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

    base_config = json.loads((llama / "config.json").read_text(encoding="utf-8"))
    base_fingerprint = model_fingerprint(llama)
    base_tokenizer_fingerprint = tokenizer_fingerprint(llama)
    output_root.mkdir(parents=True, exist_ok=True)
    sft_temporary = output_root / f".sft10k.partial.{os.getpid()}"
    gold_temporary = output_root / f".reward-model-human.partial.{os.getpid()}"
    sft_output = output_root / "sft10k"
    gold_output = output_root / "reward-model-human"

    print(f"Loading authorized LLaMA base on {device}")
    model_raw = transformers.AutoModelForCausalLM.from_pretrained(
        llama, device_map={"": device}, torch_dtype=torch.float32
    ).eval()
    tokenizer_raw = transformers.AutoTokenizer.from_pretrained(llama)
    if tokenizer_raw.pad_token is None:
        stable_resize_token_embeddings_and_tokenizer(
            model=model_raw,
            tokenizer=tokenizer_raw,
            special_tokens_dict={"pad_token": "[PAD]"},
        )

    print("Reconstructing SFT10k from the pinned local difference")
    model_sft_diff = transformers.AutoModelForCausalLM.from_pretrained(
        sft_diff, device_map={"": device}, torch_dtype=torch.float32
    ).eval()
    tokenizer_sft = transformers.AutoTokenizer.from_pretrained(sft_diff)
    reconstruct_tuned_model(model_sft_diff, model_raw, is_reward_model=False)
    sft_expected, sft_actual = assert_integrity(model_sft_diff, sft_diff / "model_sum.txt", "SFT10k")
    save_reconstruction(model_sft_diff, tokenizer_sft, sft_temporary, sft_diff / "model_sum.txt")
    sft_temporary.replace(sft_output)
    del model_sft_diff, tokenizer_sft
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Reconstructing reward-model-human from the pinned local difference")
    model_gold_diff = RewardModel.from_pretrained(
        gold_diff,
        device_map={"": device},
        torch_dtype=torch.float32,
        flash_attn=False,
        config=RewardConfig(backbone_model_name_or_path=str(sft_output)),
    ).eval()
    tokenizer_gold = transformers.AutoTokenizer.from_pretrained(gold_diff)
    reconstruct_tuned_model(model_gold_diff, model_raw, is_reward_model=True)
    gold_expected, gold_actual = assert_integrity(
        model_gold_diff, gold_diff / "model_sum.txt", "reward-model-human"
    )
    save_reconstruction(model_gold_diff, tokenizer_gold, gold_temporary, gold_diff / "model_sum.txt")
    gold_temporary.replace(gold_output)
    del model_gold_diff, tokenizer_gold, model_raw, tokenizer_raw
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metadata = {
        "schema_version": "1.0.0",
        "artifact": "alpaca_farm_reward_model_human_reconstruction",
        "authorized_llama_use_acknowledged": True,
        "pinned_alpaca_farm_revision": "f92bd550130975436301ba02137b303d1eb59986",
        "code_revision": git_revision(ROOT),
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "base": {
            "path": str(llama),
            "declared_transformers_version": base_config.get("transformers_version"),
            "model_fingerprint": base_fingerprint,
            "tokenizer_fingerprint": base_tokenizer_fingerprint,
        },
        "sft10k": {
            "source_path": str(sft_diff),
            "source_revision": sft_asset["revision"],
            "source_manifest_entry_sha256": canonical_json_hash(sft_asset),
            "source_model_sum_sha256": sha256_file(sft_diff / "model_sum.txt"),
            "expected_model_sum": sft_expected,
            "actual_model_sum": sft_actual,
            "output_path": str(sft_output),
            "model_fingerprint": model_fingerprint(sft_output),
            "tokenizer_fingerprint": tokenizer_fingerprint(sft_output),
        },
        "reward_model_human": {
            "source_path": str(gold_diff),
            "source_revision": gold_asset["revision"],
            "source_manifest_entry_sha256": canonical_json_hash(gold_asset),
            "source_model_sum_sha256": sha256_file(gold_diff / "model_sum.txt"),
            "expected_model_sum": gold_expected,
            "actual_model_sum": gold_actual,
            "output_path": str(gold_output),
            "model_fingerprint": model_fingerprint(gold_output),
            "tokenizer_fingerprint": tokenizer_fingerprint(gold_output),
        },
    }
    atomic_write_json(output_root / "reconstruction_metadata.json", metadata)
    print(f"PASS reconstructed SFT10k: {sft_output}")
    print(f"PASS reconstructed reward-model-human: {gold_output}")
    print(f"PASS reconstruction metadata: {output_root / 'reconstruction_metadata.json'}")


if __name__ == "__main__":
    main()
