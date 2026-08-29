#!/usr/bin/env python
"""Generate once, then proxy/gold/KL-score the same held-out responses."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alpaca_farm.models.reward_model import RewardModel
from src.cpdpo.artifacts import canonical_json_hash, model_fingerprint, sha256_file
from src.cpdpo.evaluation import checkpoint_summary, format_alpaca_gold_sample, hydra_policy_logits
from src.cpdpo.reward_features import load_proxy_feature_scorer
from src.cpdpo.run_logging import load_rollout_records
from src.data_utils.split_manifest import PROMPT_ID_FIELD, load_split_records, verify_split_manifest
from trlx.utils.modeling import logprobs_of_labels
from trlx.models.modeling_ppo import AutoModelForCausalLMWithHydraValueHead


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--initial-policy", required=True)
    parser.add_argument("--reference-policy", required=True)
    parser.add_argument("--proxy-rm", required=True)
    parser.add_argument("--gold-rm", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["D_rl_val_prompts", "D_rl_test_prompts"], default="D_rl_val_prompts")
    parser.add_argument("--num-prompts", type=int, default=0, help="0 evaluates the complete split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--proxy-batch-size", type=int, default=64)
    parser.add_argument("--gold-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def policy_prompt(row: dict) -> str:
    raw = f"{row['instruction']}\n{row['input']}" if row["input"] else row["instruction"]
    return f"<|prompter|>{raw}<|endoftext|><|assistant|>"


def discover_checkpoints(run_dir: Path, initial_policy: Path, updates_per_rollout: int) -> list[tuple[int, int, Path]]:
    values = [(0, 0, initial_policy)]
    for directory in sorted((run_dir / "checkpoints").glob("checkpoint_*")):
        try:
            optimizer_step = int(directory.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        policy = directory / "hf_model"
        if policy.is_dir():
            if optimizer_step % updates_per_rollout:
                raise ValueError(f"Checkpoint is not on a rollout boundary: {directory}")
            state_path = directory / "experiment_state.json"
            if not state_path.is_file():
                raise FileNotFoundError(f"Checkpoint experiment state is missing: {state_path}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                int(state.get("optimizer_step", -1)) != optimizer_step
                or int(state.get("completed_rollouts", -1)) != optimizer_step // updates_per_rollout
            ):
                raise ValueError(f"Checkpoint experiment counters are inconsistent: {directory}")
            values.append((optimizer_step // updates_per_rollout, optimizer_step, policy))
    if len(values) == 1:
        raise FileNotFoundError(f"No policy checkpoints found under {run_dir / 'checkpoints'}")
    return values


def load_policy(path: Path, dtype, device):
    # trlx checkpoints prefix policy weights with `base_model.` and also carry
    # value/frozen-head state, so they must be loaded through the pinned wrapper.
    # The pinned trlx loader predates pathlib support and accepts only `str` or
    # a transformers model instance at this API boundary.
    model = AutoModelForCausalLMWithHydraValueHead.from_pretrained(
        str(path), num_layers_unfrozen=2, num_value_layers_unfrozen=0, torch_dtype=dtype
    )
    return model.eval().requires_grad_(False).to(device)


def load_reference_policy(path: Path, dtype, device):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).eval().requires_grad_(False).to(device)


def generate_and_kl(
    *,
    policy,
    reference,
    tokenizer,
    rows,
    batch_size,
    device,
    seed,
):
    outputs = []
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [policy_prompt(row) for row in batch]
            tokens = tokenizer(prompts, padding=True, truncation=True, max_length=520, return_tensors="pt").to(device)
            prompt_width = tokens.input_ids.shape[1]
            with torch.no_grad():
                generated = policy.generate(
                    **tokens,
                    do_sample=True,
                    top_k=0,
                    top_p=1.0,
                    temperature=1.0,
                    max_new_tokens=128,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                attention_mask = generated.ne(tokenizer.pad_token_id).long()
                position_ids = attention_mask.cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                policy_logits = hydra_policy_logits(
                    policy,
                    generated,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                reference_logits = reference(generated, attention_mask=attention_mask, position_ids=position_ids).logits
                labels = generated[:, 1:]
                policy_logprobs = logprobs_of_labels(policy_logits[:, :-1, :], labels)
                reference_logprobs = logprobs_of_labels(reference_logits[:, :-1, :], labels)
            response_ids = generated[:, prompt_width:]
            response_mask = response_ids.ne(tokenizer.pad_token_id)
            start_logprob = prompt_width - 1
            token_log_ratio = (
                policy_logprobs[:, start_logprob : start_logprob + response_ids.shape[1]]
                - reference_logprobs[:, start_logprob : start_logprob + response_ids.shape[1]]
            )
            sampled_kl = (token_log_ratio * response_mask).sum(dim=1).cpu()
            decoded = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            for row, text, ids, mask, kl in zip(batch, decoded, response_ids.cpu(), response_mask.cpu(), sampled_kl):
                kept_ids = ids[mask].tolist()
                outputs.append(
                    {
                        "prompt_id": row[PROMPT_ID_FIELD],
                        "instruction": row["instruction"],
                        "input": row["input"],
                        "output": text,
                        "response_token_ids": kept_ids,
                        "generated_tokens": len(kept_ids),
                        "sampled_kl": float(kl.item()),
                    }
                )
    return outputs


def score_gold(rows: list[dict], gold_path: str, device, dtype, batch_size: int) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(gold_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    model = RewardModel.from_pretrained(gold_path, flash_attn=False, bf16=dtype == torch.bfloat16)
    model = model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    samples = [format_alpaca_gold_sample(row["instruction"], row["input"], row["output"]) for row in rows]
    values = []
    for start in range(0, len(samples), batch_size):
        tokenized = tokenizer(
            samples[start : start + batch_size], padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            rewards = model(**tokenized).rewards.float().reshape(-1).cpu()
        if rewards.numel() != len(samples[start : start + batch_size]):
            raise RuntimeError("Gold RM returned an unexpected reward shape")
        values.extend(rewards.tolist())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return values


def main() -> None:  # noqa: C901
    args = arguments()
    run_dir = Path(args.run_dir).resolve()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("gold_training_access") is not False:
        raise ValueError("Run metadata does not prove gold isolation")
    manifest = Path(args.manifest).resolve()
    verify_split_manifest(manifest)
    manifest_fingerprint = sha256_file(manifest)
    if manifest_fingerprint != metadata.get("data_manifest_sha256"):
        raise ValueError("Evaluation manifest does not match the training run")
    rows = load_split_records(manifest, args.split, expected_kind="prompt")
    if args.num_prompts:
        rows = rows[: args.num_prompts]
    if not rows:
        raise ValueError("Evaluation prompt set is empty")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    initial = Path(args.initial_policy).resolve()
    reference_path = Path(args.reference_policy).resolve()
    expected_policy = metadata["initial_policy_fingerprint"]
    if model_fingerprint(initial) != expected_policy or model_fingerprint(reference_path) != metadata["reference_policy_fingerprint"]:
        raise ValueError("Initial/reference policy fingerprint mismatch")
    if model_fingerprint(args.proxy_rm) != metadata["proxy_rm_fingerprint"]:
        raise ValueError("Proxy RM fingerprint mismatch")
    gold_rm_fingerprint = model_fingerprint(args.gold_rm)
    checkpoints = discover_checkpoints(run_dir, initial, int(metadata["updates_per_rollout"]))
    for _rollout_step, _optimizer_step, checkpoint in checkpoints[1:]:
        state = json.loads((checkpoint.parent / "experiment_state.json").read_text(encoding="utf-8"))
        context = state["experiment_context"]
        expected = {
            "method": metadata["method"],
            "base_seed": metadata["base_seed"],
            "rollout_steps": metadata["rollout_steps"],
            "updates_per_rollout": metadata["updates_per_rollout"],
            "prompt_schedule_sha256": metadata["prompt_schedule_sha256"],
            "prompt_id_sequence_sha256": metadata["prompt_id_sequence_sha256"],
            "data_manifest_sha256": metadata["data_manifest_sha256"],
            "initial_policy_fingerprint": metadata["initial_policy_fingerprint"],
            "reference_policy_fingerprint": metadata["reference_policy_fingerprint"],
            "proxy_rm_fingerprint": metadata["proxy_rm_fingerprint"],
            "pair_artifacts": metadata["pair_artifacts"],
            "code_revision": metadata["code_revision"],
        }
        if any(context.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Checkpoint immutable context does not match run metadata: {checkpoint.parent}")
    rollout_records = load_rollout_records(run_dir)
    output_dir = run_dir / "evaluation" / args.split
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(initial)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|padding|>"
    reference = load_reference_policy(reference_path, dtype, device)
    all_records = []
    for rollout_step, optimizer_step, checkpoint in checkpoints:
        policy = load_policy(checkpoint, dtype, device)
        generated = generate_and_kl(
            policy=policy,
            reference=reference,
            tokenizer=tokenizer,
            rows=rows,
            batch_size=args.batch_size,
            device=device,
            seed=int(metadata["resolved_seeds"]["evaluation_generation"]) + rollout_step,
        )
        checkpoint_fingerprint = model_fingerprint(checkpoint)
        for row in generated:
            row.update(
                {
                    "method": metadata["method"],
                    "seed": metadata["base_seed"],
                    "rollout_step": rollout_step,
                    "optimizer_step": optimizer_step,
                    "policy_checkpoint": str(checkpoint),
                    "policy_checkpoint_fingerprint": checkpoint_fingerprint,
                }
            )
            row["response_id"] = canonical_json_hash(
                [row["method"], row["seed"], rollout_step, row["prompt_id"], row["response_token_ids"]]
            )
        response_file = output_dir / f"responses_rollout_{rollout_step}.jsonl"
        with response_file.open("x", encoding="utf-8", newline="\n") as handle:
            for row in generated:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        all_records.extend(generated)
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    proxy = load_proxy_feature_scorer(args.proxy_rm, device=device, batch_size=args.proxy_batch_size)
    proxy_rewards, _features = proxy.score(
        [f"{row['instruction']}\n{row['input']}" if row["input"] else row["instruction"] for row in all_records],
        [row["output"] for row in all_records],
        evaluation=True,
    )
    for row, reward in zip(all_records, proxy_rewards.tolist()):
        row["proxy_reward"] = float(reward)
    del proxy
    gc.collect()
    torch.cuda.empty_cache()
    gold_rewards = score_gold(all_records, args.gold_rm, device, dtype, args.gold_batch_size)
    for row, reward in zip(all_records, gold_rewards):
        row["gold_reward"] = float(reward)

    scored_path = output_dir / "scored_responses.jsonl"
    with scored_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in all_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = output_dir / "checkpoint_metrics.jsonl"
    metric_records = []
    with metrics_path.open("x", encoding="utf-8", newline="\n") as handle:
        for rollout_step, optimizer_step, checkpoint in checkpoints:
            selected = [row for row in all_records if row["rollout_step"] == rollout_step]
            checkpoint_fingerprints = {row["policy_checkpoint_fingerprint"] for row in selected}
            if len(checkpoint_fingerprints) != 1:
                raise RuntimeError("Evaluation rows disagree on the policy checkpoint fingerprint")
            summary = checkpoint_summary(selected)
            training_counts = (
                {"generated_responses": 0, "generated_tokens": 0, "proxy_rm_calls": 0}
                if rollout_step == 0
                else rollout_records[rollout_step]
            )
            record = {
                "schema_version": "1.0.0",
                "experiment": "reward_overoptimization",
                "method": metadata["method"],
                "seed": metadata["base_seed"],
                "rollout_step": rollout_step,
                "optimizer_step": optimizer_step,
                "generated_responses": training_counts["generated_responses"],
                "generated_tokens": training_counts["generated_tokens"],
                "proxy_rm_calls": training_counts["proxy_rm_calls"],
                "evaluation_responses": len(selected),
                "evaluation_generated_tokens": sum(row["generated_tokens"] for row in selected),
                "evaluation_proxy_rm_calls": len(selected),
                "policy_checkpoint": str(checkpoint),
                "policy_checkpoint_fingerprint": next(iter(checkpoint_fingerprints)),
                "initial_policy_fingerprint": metadata["initial_policy_fingerprint"],
                "reference_policy_fingerprint": metadata["reference_policy_fingerprint"],
                "proxy_rm_fingerprint": metadata["proxy_rm_fingerprint"],
                "gold_rm_fingerprint": gold_rm_fingerprint,
                "geometry_fingerprint": (metadata.get("pair_artifacts") or {}).get("geometry_fingerprint"),
                "calibration_fingerprint": (metadata.get("pair_artifacts") or {}).get("calibration_fingerprint"),
                "q_alpha": (metadata.get("pair_artifacts") or {}).get("q_alpha"),
                "certification_rate": (
                    training_counts.get("pair/certification_rate")
                    if metadata["method"] == "cpdpo" and rollout_step > 0
                    else None
                ),
                "evaluation_split": args.split,
                "evaluation_manifest_sha256": manifest_fingerprint,
                "evaluation_prompt_ids_sha256": canonical_json_hash(
                    [row["prompt_id"] for row in selected]
                ),
                "response_ids_sha256": canonical_json_hash([row["response_id"] for row in selected]),
                **summary,
            }
            metric_records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    metrics_csv_path = output_dir / "checkpoint_metrics.csv"
    with metrics_csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_records[0]))
        writer.writeheader()
        writer.writerows(metric_records)
    print(f"PASS common checkpoint evaluation: {metrics_path}")
    print(f"PASS checkpoint CSV: {metrics_csv_path}")
    print(f"PASS same-response scored records: {scored_path}")


if __name__ == "__main__":
    main()
