"""Read and verify immutable logical dataset split manifests.

This module intentionally uses only the Python standard library. Training
adapters can therefore validate split membership and payload hashes before
importing Torch, Hugging Face Datasets, or the legacy Open-Assistant stack.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
REQUIRED_LOGICAL_SPLITS = {
    "D_rm_train": "preference",
    "D_rm_val": "preference",
    "D_cal": "preference",
    "D_rl_train_prompts": "prompt",
    "D_rl_val_prompts": "prompt",
    "D_rl_test_prompts": "prompt",
}
RECORD_ID_FIELD = "_split_record_id"
PROMPT_ID_FIELD = "_split_prompt_id"
ROLE_FIELD = "_split_role"
DUPLICATE_ORDINAL_FIELD = "_split_duplicate_ordinal"


class SplitManifestError(RuntimeError):
    """Raised when split provenance, membership, or a materialized file is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SplitManifestError(f"Split manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise SplitManifestError(f"Split manifest is not valid JSON: {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise SplitManifestError("Split manifest root must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SplitManifestError(
            f"Unsupported split manifest schema: {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise SplitManifestError("Split manifest must contain a splits object")
    missing = sorted(set(REQUIRED_LOGICAL_SPLITS) - set(splits))
    if missing:
        raise SplitManifestError(f"Split manifest is missing required roles: {', '.join(missing)}")
    return manifest_path, manifest


def _resolve_manifest_file(manifest_path: Path, relative_value: Any, label: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise SplitManifestError(f"{label} must be a non-empty relative path")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SplitManifestError(f"Unsafe {label}: {relative_value!r}")
    root = manifest_path.parent.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SplitManifestError(f"{label} escapes the split bundle: {relative_value!r}") from exc
    return candidate


def _read_ids(path: Path) -> list[str]:
    try:
        ids = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except FileNotFoundError as exc:
        raise SplitManifestError(f"Split ID file does not exist: {path}") from exc
    if len(ids) != len(set(ids)):
        raise SplitManifestError(f"Split ID file contains duplicates: {path}")
    return ids


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SplitManifestError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise SplitManifestError(f"Expected an object at {path}:{line_number}")
                records.append(record)
    except FileNotFoundError as exc:
        raise SplitManifestError(f"Split data file does not exist: {path}") from exc
    return records


def load_split_records(
    manifest_path: str | Path,
    role: str,
    *,
    expected_kind: str | None = None,
) -> list[dict[str, Any]]:
    resolved_manifest, manifest = load_split_manifest(manifest_path)
    entry = manifest["splits"].get(role)
    if not isinstance(entry, dict):
        raise SplitManifestError(f"Unknown split role: {role}")
    kind = entry.get("kind")
    if expected_kind is not None and kind != expected_kind:
        raise SplitManifestError(f"Split {role} has kind {kind!r}; expected {expected_kind!r}")

    data_path = _resolve_manifest_file(resolved_manifest, entry.get("data_file"), f"{role}.data_file")
    ids_path = _resolve_manifest_file(resolved_manifest, entry.get("ids_file"), f"{role}.ids_file")
    for path, expected_hash, label in (
        (data_path, entry.get("data_sha256"), "data"),
        (ids_path, entry.get("ids_sha256"), "IDs"),
    ):
        if not path.is_file():
            raise SplitManifestError(f"Split {label} file does not exist: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise SplitManifestError(
                f"Split {role} {label} hash mismatch: expected {expected_hash}, found {actual_hash}"
            )

    ids = _read_ids(ids_path)
    records = _read_jsonl(data_path)
    expected_count = entry.get("count")
    if len(records) != expected_count or len(ids) != expected_count:
        raise SplitManifestError(
            f"Split {role} count mismatch: manifest={expected_count}, "
            f"records={len(records)}, ids={len(ids)}"
        )

    record_ids = [record.get(RECORD_ID_FIELD) for record in records]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise SplitManifestError(f"Split {role} contains a missing or invalid record ID")
    if record_ids != ids:
        raise SplitManifestError(f"Split {role} data and ID membership files disagree")
    if any(record.get(ROLE_FIELD) != role for record in records):
        raise SplitManifestError(f"Split {role} contains a record assigned to another role")
    if any(not isinstance(record.get(PROMPT_ID_FIELD), str) for record in records):
        raise SplitManifestError(f"Split {role} contains a missing prompt ID")
    return records


def verify_split_manifest(path: str | Path) -> dict[str, int]:
    manifest_path, manifest = load_split_manifest(path)
    companion = manifest_path.with_name("manifest.sha256")
    if not companion.is_file():
        raise SplitManifestError(f"Manifest hash companion does not exist: {companion}")
    companion_fields = companion.read_text(encoding="utf-8").strip().split()
    if not companion_fields:
        raise SplitManifestError(f"Manifest hash companion is empty: {companion}")
    expected = companion_fields[0]
    actual = sha256_file(manifest_path)
    if expected != actual:
        raise SplitManifestError(
            f"Manifest hash mismatch: expected {expected}, found {actual}"
        )

    records_by_role: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for role, entry in manifest["splits"].items():
        if not isinstance(entry, dict):
            raise SplitManifestError(f"Split entry {role} must be an object")
        if entry.get("kind") not in {"preference", "prompt"}:
            raise SplitManifestError(f"Split {role} has unsupported kind {entry.get('kind')!r}")
        records = load_split_records(
            manifest_path,
            role,
            expected_kind=REQUIRED_LOGICAL_SPLITS.get(role),
        )
        records_by_role[role] = records
        counts[role] = len(records)

    for kind in ("preference", "prompt"):
        roles = tuple(
            role
            for role, entry in manifest["splits"].items()
            if isinstance(entry, dict) and entry.get("kind") == kind
        )
        seen_records: set[str] = set()
        seen_prompts: set[str] = set()
        for role in roles:
            records = records_by_role[role]
            record_ids = {record[RECORD_ID_FIELD] for record in records}
            prompt_ids = {record[PROMPT_ID_FIELD] for record in records}
            if seen_records.intersection(record_ids):
                raise SplitManifestError(f"Record leakage detected at {role}")
            if seen_prompts.intersection(prompt_ids):
                raise SplitManifestError(f"Prompt leakage detected at {role}")
            seen_records.update(record_ids)
            seen_prompts.update(prompt_ids)

    rm_prompts = {
        record[PROMPT_ID_FIELD]
        for role in ("D_rm_train", "D_rm_val", "D_cal")
        for record in records_by_role[role]
    }
    rl_prompts = {
        record[PROMPT_ID_FIELD]
        for role in ("D_rl_train_prompts", "D_rl_val_prompts", "D_rl_test_prompts")
        for record in records_by_role[role]
    }
    actual_overlap = len(rm_prompts.intersection(rl_prompts))
    recorded_overlap = manifest.get("overlap_audit", {}).get("logical_cross_source_prompt_count")
    if recorded_overlap != actual_overlap:
        raise SplitManifestError(
            f"Cross-source overlap audit mismatch: manifest={recorded_overlap}, actual={actual_overlap}"
        )
    all_rm_prompts = {
        record[PROMPT_ID_FIELD]
        for role, records in records_by_role.items()
        if manifest["splits"][role].get("kind") == "preference"
        for record in records
    }
    all_rl_prompts = {
        record[PROMPT_ID_FIELD]
        for role, records in records_by_role.items()
        if manifest["splits"][role].get("kind") == "prompt"
        for record in records
    }
    actual_all_overlap = len(all_rm_prompts.intersection(all_rl_prompts))
    recorded_all_overlap = manifest.get("overlap_audit", {}).get(
        "all_preserved_cross_source_prompt_count"
    )
    if recorded_all_overlap != actual_all_overlap:
        raise SplitManifestError(
            "All-role cross-source overlap audit mismatch: "
            f"manifest={recorded_all_overlap}, actual={actual_all_overlap}"
        )
    return counts
