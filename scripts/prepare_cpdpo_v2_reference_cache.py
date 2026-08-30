#!/usr/bin/env python
"""Generate and score one immutable frozen-SFT response per scheduled prompt."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from model_training.custom_datasets.formatting import format_pairs
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cpdpo.artifacts import (
    atomic_write_json,
    canonical_json_hash,
    git_revision,
    model_fingerprint,
    sha256_file,
    tokenizer_fingerprint,
)
from src.cpdpo.pair_reward import PairRewardCallback
from src.cpdpo.reference_anchor import REFERENCE_CACHE_SCHEMA, REFERENCE_GENERATION_SEED_OFFSET
from src.cpdpo.spec import ALPHA, CPDPOConfig, is_main_alpha


GENERATION_SETTINGS = {
    "do_sample": True,
    "top_k": 0,
    "top_p": 1.0,
    "temperature": 1.0,
    "max_new_tokens": 128,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-schedule", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--proxy-rm", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--proxy-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--allow-smoke-artifacts", action="store_true")
    return parser.parse_args()


def load_unique_schedule(path: Path) -> tuple[list[dict], dict]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prompt schedule metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schedule_sha256") != sha256_file(path):
        raise ValueError("Prompt schedule fingerprint mismatch")
    unique: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = row["prompt_id"]
            content = {"prompt_id": prompt_id, "instruction": row["instruction"], "input": row["input"]}
            if prompt_id in unique and unique[prompt_id] != content:
                raise ValueError(f"Schedule reuses prompt ID with different content: {prompt_id}")
            unique.setdefault(prompt_id, content)
    if not unique:
        raise ValueError("Prompt schedule contains no prompts")
    return list(unique.values()), metadata


def generate_references(rows, *, policy_path: Path, batch_size: int, device: torch.device, dtype, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(policy_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|padding|>"
    model = AutoModelForCausalLM.from_pretrained(
        str(policy_path), torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval().requires_grad_(False).to(device)
    raw_prompts = [
        f"{row['instruction']}\n{row['input']}" if row["input"] else row["instruction"]
        for row in rows
    ]
    policy_prompts = [
        "".join(format_pairs([prompt], tokenizer.eos_token, add_initial_reply_token=True))
        for prompt in raw_prompts
    ]
    responses: list[str] = []
    response_token_ids: list[list[int]] = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        for start in range(0, len(rows), batch_size):
            prompts = policy_prompts[start : start + batch_size]
            tokenized = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=520,
                return_tensors="pt",
            ).to(device)
            prompt_width = tokenized.input_ids.shape[1]
            with torch.no_grad():
                generated = model.generate(
                    **tokenized,
                    **GENERATION_SETTINGS,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output_ids = generated[:, prompt_width:]
            for sample, ids in zip(generated, output_ids):
                text = tokenizer.decode(ids, skip_special_tokens=True)
                kept = ids[ids.ne(tokenizer.pad_token_id)].tolist()
                if int(sample[-1].item()) in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
                    text += tokenizer.eos_token
                responses.append(text)
                response_token_ids.append(kept)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tokenizer, raw_prompts, responses, response_token_ids


def atomic_torch_save(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    args = arguments()
    if args.base_seed < 0 or args.batch_size < 1 or args.proxy_batch_size < 1:
        raise ValueError("Seeds must be nonnegative and batch sizes positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    schedule = Path(args.prompt_schedule).resolve()
    manifest = Path(args.manifest).resolve()
    policy_path = Path(args.policy_model).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty reference cache directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows, schedule_metadata = load_unique_schedule(schedule)
    if schedule_metadata.get("base_seed") != args.base_seed:
        raise ValueError("Reference cache base seed does not match the prompt schedule")
    if schedule_metadata.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("Reference cache manifest does not match the prompt schedule")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    reference_seed = args.base_seed + REFERENCE_GENERATION_SEED_OFFSET
    _policy_tokenizer, raw_prompts, responses, response_token_ids = generate_references(
        rows,
        policy_path=policy_path,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        seed=reference_seed,
    )

    artifact_loader = PairRewardCallback(
        proxy_rm_path=args.proxy_rm,
        config=CPDPOConfig(method="cpdpo", alpha=args.alpha),
        device=device,
        batch_size=args.proxy_batch_size,
        geometry_path=args.geometry,
        calibration_path=args.calibration,
        data_manifest_path=str(manifest),
        allow_smoke_artifacts=args.allow_smoke_artifacts,
    )
    reference_rewards, reference_features = artifact_loader.scorer.score(raw_prompts, responses)
    prompt_ids = [row["prompt_id"] for row in rows]
    response_ids = [
        canonical_json_hash([prompt_id, token_ids])
        for prompt_id, token_ids in zip(prompt_ids, response_token_ids)
    ]
    response_path = output / "reference_responses.jsonl"
    with response_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row, prompt, response, token_ids, response_id in zip(
            rows, raw_prompts, responses, response_token_ids, response_ids
        ):
            handle.write(
                json.dumps(
                    {
                        **row,
                        "prompt": prompt,
                        "response": response,
                        "response_token_ids": token_ids,
                        "reference_response_id": response_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    cache_path = output / "reference_cache.pt"
    atomic_torch_save(
        cache_path,
        {
            "schema_version": REFERENCE_CACHE_SCHEMA,
            "prompt_ids": prompt_ids,
            "prompts": raw_prompts,
            "responses": responses,
            "response_token_ids": response_token_ids,
            "reference_rewards": reference_rewards.detach().float().cpu(),
            "reference_features": reference_features.detach().float().cpu(),
        },
    )
    metadata = {
        "schema_version": REFERENCE_CACHE_SCHEMA,
        "method": "cpdpo_v2",
        "experiment_track": "main" if is_main_alpha(args.alpha) else "cpdpo_v2_alpha_ablation",
        "source_role": "D_rl_train_prompts",
        "artifact_scope": artifact_loader.artifact_scope,
        "base_seed": args.base_seed,
        "reference_generation_seed": reference_seed,
        "generation_settings": GENERATION_SETTINGS,
        "generation_batch_size": args.batch_size,
        "dtype": args.dtype,
        "unique_prompt_count": len(prompt_ids),
        "offline_reference_policy_generations": len(prompt_ids),
        "offline_proxy_rm_calls": len(prompt_ids),
        "prompt_ids_sha256": canonical_json_hash(prompt_ids),
        "reference_response_ids_sha256": canonical_json_hash(response_ids),
        "reference_responses_fingerprint": sha256_file(response_path),
        "reference_cache_fingerprint": sha256_file(cache_path),
        "reference_policy_path": str(policy_path),
        "reference_policy_fingerprint": model_fingerprint(policy_path),
        "reference_policy_tokenizer_fingerprint": tokenizer_fingerprint(policy_path),
        "proxy_rm_path": artifact_loader.proxy_rm_path,
        "proxy_rm_fingerprint": artifact_loader.proxy_rm_fingerprint,
        "tokenizer_fingerprint": artifact_loader.tokenizer_fingerprint,
        "data_manifest_path": str(manifest),
        "data_manifest_sha256": sha256_file(manifest),
        "prompt_schedule_path": str(schedule),
        "prompt_schedule_sha256": sha256_file(schedule),
        "geometry_fingerprint": artifact_loader.geometry_fingerprint,
        "calibration_fingerprint": artifact_loader.calibration_fingerprint,
        "alpha": args.alpha,
        "q_alpha": artifact_loader.q_alpha,
        "exchangeability_assumption": "D_cal pair differences exchangeable with current/SFT differences",
        "gold_reward_bound": False,
        "gold_access": False,
        "code_revision": git_revision(ROOT),
    }
    atomic_write_json(output / "reference_cache_metadata.json", metadata)
    print(
        f"PASS CPDPOv2 reference cache: {cache_path} "
        f"unique_prompts={len(prompt_ids)} alpha={args.alpha:g} q_alpha={artifact_loader.q_alpha}"
    )


if __name__ == "__main__":
    main()
