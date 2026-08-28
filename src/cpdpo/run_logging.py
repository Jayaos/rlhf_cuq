"""Append-only rollout accounting shared by all experiment trainers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


def append_rollout_record(output_dir: str | Path, record: dict[str, Any]) -> None:
    path = Path(output_dir) / "rollout_metrics.jsonl"
    prior = []
    if path.is_file():
        prior = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    expected_step = len(prior) + 1
    if record.get("rollout_step") != expected_step:
        raise RuntimeError(
            f"Rollout metric sequence mismatch: expected {expected_step}, found {record.get('rollout_step')}"
        )
    previous_responses = prior[-1]["generated_responses"] if prior else 0
    previous_tokens = prior[-1]["generated_tokens"] if prior else 0
    previous_calls = prior[-1]["proxy_rm_calls"] if prior else 0
    row = dict(record)
    row["generated_responses_this_rollout"] = int(row.pop("response_count"))
    row["generated_tokens_this_rollout"] = int(row.pop("generated_token_count"))
    row["proxy_rm_calls_this_rollout"] = int(row.pop("proxy_call_count"))
    row["generated_responses"] = previous_responses + row["generated_responses_this_rollout"]
    row["generated_tokens"] = previous_tokens + row["generated_tokens_this_rollout"]
    row["proxy_rm_calls"] = previous_calls + row["proxy_rm_calls_this_rollout"]
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def load_rollout_records(output_dir: str | Path) -> dict[int, dict[str, Any]]:
    path = Path(output_dir) / "rollout_metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Training rollout metrics are missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {int(row["rollout_step"]): row for row in rows}


def rewind_rollout_records(output_dir: str | Path, completed_rollouts: int) -> Path | None:
    """Recoverably discard only records newer than a restored checkpoint."""

    path = Path(output_dir) / "rollout_metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Training rollout metrics are missing: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in lines]
    if len(rows) < completed_rollouts:
        raise ValueError("Rollout log ends before the requested resume checkpoint")
    for index, row in enumerate(rows, start=1):
        if int(row["rollout_step"]) != index:
            raise ValueError("Rollout log is not a contiguous one-based sequence")
    if len(rows) == completed_rollouts:
        return None

    backup = path.with_name(f"rollout_metrics.before_resume_{time.time_ns()}.jsonl")
    shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines[:completed_rollouts]:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return backup


def archive_pair_rollouts_after_checkpoint(
    output_dir: str | Path, completed_rollouts: int
) -> Path | None:
    """Move post-checkpoint pair snapshots to a recoverable archive directory."""

    source = Path(output_dir) / "pair_rollouts"
    if not source.is_dir():
        return None
    newer = []
    for path in source.glob("rollout_*.pt"):
        try:
            step = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Malformed pair rollout snapshot name: {path}") from exc
        if step > completed_rollouts:
            newer.append(path)
    if not newer:
        return None
    archive = Path(output_dir) / f"pair_rollouts.before_resume_{time.time_ns()}"
    archive.mkdir()
    for path in sorted(newer):
        shutil.move(str(path), archive / path.name)
    return archive
