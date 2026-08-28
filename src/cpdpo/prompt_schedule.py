"""Deterministic shared prompt schedules and seed namespaces."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.cpdpo.artifacts import canonical_json_hash
from src.data_utils.split_manifest import PROMPT_ID_FIELD, load_split_records, sha256_file


@dataclass(frozen=True)
class ExperimentSeeds:
    training: int
    rollout_generation: int
    evaluation_generation: int
    prompt_schedule: int

    @classmethod
    def from_base(cls, base_seed: int) -> "ExperimentSeeds":
        if base_seed < 0:
            raise ValueError("base seed cannot be negative")
        return cls(base_seed, base_seed + 10_000, base_seed + 20_000, base_seed + 30_000)


def build_prompt_schedule(
    records: list[dict[str, Any]],
    *,
    rollout_steps: int,
    prompts_per_rollout: int,
    seed: int,
) -> list[dict[str, Any]]:
    if rollout_steps < 1 or prompts_per_rollout < 1:
        raise ValueError("rollout_steps and prompts_per_rollout must be positive")
    if not records:
        raise ValueError("Prompt pool is empty")
    rng = random.Random(seed)
    schedule = []
    order = list(range(len(records)))
    cursor = len(order)
    for rollout_step in range(rollout_steps):
        for within_rollout in range(prompts_per_rollout):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            record = records[order[cursor]]
            cursor += 1
            schedule.append(
                {
                    "rollout_step": rollout_step,
                    "within_rollout": within_rollout,
                    "prompt_id": record[PROMPT_ID_FIELD],
                    "instruction": record["instruction"],
                    "input": record["input"],
                }
            )
    return schedule


def materialize_prompt_schedule(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    base_seed: int,
    rollout_steps: int,
    prompts_per_rollout: int,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    records = load_split_records(manifest, "D_rl_train_prompts", expected_kind="prompt")
    seeds = ExperimentSeeds.from_base(base_seed)
    schedule = build_prompt_schedule(
        records,
        rollout_steps=rollout_steps,
        prompts_per_rollout=prompts_per_rollout,
        seed=seeds.prompt_schedule,
    )
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite prompt schedule: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for row in schedule:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "schema_version": "1.0.0",
        "base_seed": base_seed,
        "resolved_seeds": asdict(seeds),
        "rollout_steps": rollout_steps,
        "prompts_per_rollout": prompts_per_rollout,
        "responses_per_prompt": 2,
        "row_count": len(schedule),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "schedule_sha256": sha256_file(target),
        "prompt_id_sequence_sha256": canonical_json_hash([row["prompt_id"] for row in schedule]),
    }
    metadata_path = target.with_suffix(target.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_schedule_as_duplicated_prompts(
    schedule_path: str | Path, *, rollout_steps: int | None = None
) -> list[dict[str, Any]]:
    rows = []
    with Path(schedule_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if rollout_steps is None or row["rollout_step"] < rollout_steps:
                    rows.append(row)
    prompts = []
    for row in rows:
        prompt = f"{row['instruction']}\n{row['input']}" if row["input"] else row["instruction"]
        for orientation in ("a", "b"):
            prompts.append(
                {
                    "prompt": prompt,
                    "prompt_id": row["prompt_id"],
                    "rollout_step": row["rollout_step"],
                    "within_rollout": row["within_rollout"],
                    "orientation": orientation,
                }
            )
    return prompts
