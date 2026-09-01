#!/usr/bin/env python
"""Generate and proxy-score one immutable SFT reference per AdvPO prompt."""

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

from src.advpo.reference import (
    ADVPO_GENERATION_SETTINGS,
    ADVPO_PROMPT_CANONICALIZATION,
    ADVPO_REFERENCE_GENERATION_SEED_OFFSET,
    ADVPO_REFERENCE_SCHEMA,
)
from src.cpdpo.artifacts import (
    atomic_write_json,
    canonical_json_hash,
    git_revision,
    model_fingerprint,
    sha256_file,
    tokenizer_fingerprint,
)
from src.cpdpo.reward_features import load_proxy_feature_scorer
from src.data_utils.split_manifest import verify_split_manifest
from src.ppo.policy_variants import (
    DEFAULT_POLICY_VARIANT,
    POLICY_VARIANTS,
    validate_policy_checkpoint,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-schedule", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument(
        "--policy-variant", choices=tuple(POLICY_VARIANTS), default=DEFAULT_POLICY_VARIANT
    )
    parser.add_argument("--proxy-rm", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--proxy-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--artifact-scope", choices=["scientific", "smoke"], default="scientific")
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
            content = {
                "prompt_id": row["prompt_id"],
                "instruction": row["instruction"],
                "input": row["input"],
            }
            if row["prompt_id"] in unique and unique[row["prompt_id"]] != content:
                raise ValueError(f"Schedule reuses prompt ID with different content: {row['prompt_id']}")
            unique.setdefault(row["prompt_id"], content)
    if not unique:
        raise ValueError("Prompt schedule contains no prompts")
    return list(unique.values()), metadata


def atomic_torch_save(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:  # noqa: C901
    args = arguments()
    if args.base_seed < 0 or args.batch_size < 1 or args.proxy_batch_size < 1:
        raise ValueError("Seeds must be nonnegative and batch sizes positive")
    schedule = Path(args.prompt_schedule).resolve()
    manifest = Path(args.manifest).resolve()
    policy_path = Path(args.policy_model).resolve()
    policy_architecture = validate_policy_checkpoint(policy_path, args.policy_variant)
    proxy_path = Path(args.proxy_rm).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty AdvPO reference directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    verify_split_manifest(manifest)
    rows, schedule_metadata = load_unique_schedule(schedule)
    if schedule_metadata.get("base_seed") != args.base_seed:
        raise ValueError("AdvPO reference seed does not match the prompt schedule")
    if schedule_metadata.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("AdvPO reference manifest does not match the prompt schedule")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
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
    proxy_prompts: list[str] = []
    reference_seed = args.base_seed + ADVPO_REFERENCE_GENERATION_SEED_OFFSET
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(reference_seed)
        if devices:
            torch.cuda.manual_seed(reference_seed)
        for start in range(0, len(rows), args.batch_size):
            tokenized = tokenizer(
                policy_prompts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=520,
                return_tensors="pt",
            ).to(device)
            prompt_width = tokenized.input_ids.shape[1]
            proxy_prompts.extend(
                tokenizer.batch_decode(tokenized.input_ids, skip_special_tokens=True)
            )
            with torch.no_grad():
                generated = model.generate(
                    **tokenized,
                    **ADVPO_GENERATION_SETTINGS,
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

    scorer = load_proxy_feature_scorer(
        str(proxy_path), device=device, batch_size=args.proxy_batch_size
    )
    reference_rewards, reference_features = scorer.score(proxy_prompts, responses)
    prompt_ids = [row["prompt_id"] for row in rows]
    response_ids = [
        canonical_json_hash([prompt_id, token_ids])
        for prompt_id, token_ids in zip(prompt_ids, response_token_ids)
    ]
    responses_path = output / "reference_responses.jsonl"
    with responses_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row, prompt, response, token_ids, response_id in zip(
            rows, proxy_prompts, responses, response_token_ids, response_ids
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
            "schema_version": ADVPO_REFERENCE_SCHEMA,
            "prompt_ids": prompt_ids,
            "prompts": proxy_prompts,
            "responses": responses,
            "response_token_ids": response_token_ids,
            "reference_rewards": reference_rewards.detach().float().cpu(),
            "reference_features": reference_features.detach().float().cpu(),
        },
    )
    metadata = {
        "schema_version": ADVPO_REFERENCE_SCHEMA,
        "method": "advpo",
        "source_role": "D_rl_train_prompts",
        "reference_type": "fixed_sft_generation_per_prompt",
        "prompt_canonicalization": ADVPO_PROMPT_CANONICALIZATION,
        "artifact_scope": args.artifact_scope,
        "base_seed": args.base_seed,
        "reference_generation_seed": reference_seed,
        "generation_settings": ADVPO_GENERATION_SETTINGS,
        "generation_batch_size": args.batch_size,
        "dtype": args.dtype,
        "unique_prompt_count": len(prompt_ids),
        "offline_reference_policy_generations": len(prompt_ids),
        "offline_proxy_rm_calls": len(prompt_ids),
        "prompt_ids_sha256": canonical_json_hash(prompt_ids),
        "reference_response_ids_sha256": canonical_json_hash(response_ids),
        "reference_responses_fingerprint": sha256_file(responses_path),
        "reference_cache_fingerprint": sha256_file(cache_path),
        "reference_policy_path": str(policy_path),
        "policy_variant": args.policy_variant,
        "policy_architecture": policy_architecture,
        "reference_policy_fingerprint": model_fingerprint(policy_path),
        "reference_policy_tokenizer_fingerprint": tokenizer_fingerprint(policy_path),
        "proxy_rm_path": str(proxy_path),
        "proxy_rm_fingerprint": model_fingerprint(proxy_path),
        "tokenizer_fingerprint": tokenizer_fingerprint(proxy_path),
        "data_manifest_path": str(manifest),
        "data_manifest_sha256": sha256_file(manifest),
        "prompt_schedule_path": str(schedule),
        "prompt_schedule_sha256": sha256_file(schedule),
        "prompt_schedule_metadata_sha256": sha256_file(
            schedule.with_suffix(schedule.suffix + ".metadata.json")
        ),
        "gold_access": False,
        "code_revision": git_revision(ROOT),
    }
    atomic_write_json(output / "reference_cache_metadata.json", metadata)
    print(
        f"PASS AdvPO SFT references: {cache_path} scope={args.artifact_scope} "
        f"unique_prompts={len(prompt_ids)} seed={reference_seed}"
    )


if __name__ == "__main__":
    main()
