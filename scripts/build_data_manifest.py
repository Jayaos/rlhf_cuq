#!/usr/bin/env python3
"""Build deterministic, disjoint RM/calibration/PPO split artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_assets import load_assets, verify_asset  # noqa: E402
from src.data_utils.split_manifest import (  # noqa: E402
    DUPLICATE_ORDINAL_FIELD,
    PROMPT_ID_FIELD,
    RECORD_ID_FIELD,
    ROLE_FIELD,
    SCHEMA_VERSION,
    SplitManifestError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_split_manifest,
)


DEFAULT_CONFIG = ROOT / "configs" / "data_split_coste_v1.yaml"
METADATA_FIELDS = {
    DUPLICATE_ORDINAL_FIELD,
    RECORD_ID_FIELD,
    PROMPT_ID_FIELD,
    ROLE_FIELD,
    "_split_source_asset",
    "_split_source_revision",
    "_split_source_split",
}


class SplitBuildError(RuntimeError):
    """Raised when source data cannot satisfy the frozen split contract."""


def compute_target_counts(total: int, allocations: Mapping[str, int | float]) -> dict[str, int]:
    """Return exact integer quotas using the largest-remainder convention."""

    if total < 0:
        raise SplitBuildError("Total record count cannot be negative")
    if not allocations:
        raise SplitBuildError("At least one allocation is required")

    weighted_roles: list[tuple[str, Fraction]] = []
    for role, value in allocations.items():
        try:
            weight = Fraction(str(value))
        except (ValueError, ZeroDivisionError) as exc:
            raise SplitBuildError(f"Invalid allocation weight for {role}: {value!r}") from exc
        if weight <= 0:
            raise SplitBuildError(f"Allocation weight for {role} must be positive")
        weighted_roles.append((role, weight))

    total_weight = sum((weight for _, weight in weighted_roles), Fraction(0))
    exact = [(role, Fraction(total) * weight / total_weight) for role, weight in weighted_roles]
    counts = {role: value.numerator // value.denominator for role, value in exact}
    remainder = total - sum(counts.values())
    ranked = sorted(
        enumerate(exact),
        key=lambda item: (-(item[1][1] - counts[item[1][0]]), item[0]),
    )
    for _, (role, _) in ranked[:remainder]:
        counts[role] += 1
    return counts


def _validate_source_row(row: Mapping[str, Any], kind: str, index: int) -> None:
    for field in METADATA_FIELDS:
        if field in row:
            raise SplitBuildError(f"Source row {index} already contains reserved field {field}")
    if not isinstance(row.get("instruction"), str) or not isinstance(row.get("input", ""), str):
        raise SplitBuildError(f"Source row {index} has an invalid instruction/input")
    if kind == "preference":
        answers = row.get("answers")
        preference = row.get("preference")
        if not isinstance(answers, list) or len(answers) != 2 or not all(
            isinstance(answer, str) for answer in answers
        ):
            raise SplitBuildError(f"Preference row {index} must contain exactly two string answers")
        if type(preference) is not int or preference not in {0, 1}:
            raise SplitBuildError(f"Preference row {index} must have preference 0 or 1")
    elif kind == "prompt":
        if not isinstance(row.get("output"), str):
            raise SplitBuildError(f"PPO prompt row {index} must contain its source output string")
    else:
        raise SplitBuildError(f"Unsupported source kind: {kind}")


def annotate_source_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    kind: str,
    asset_name: str,
    repo_id: str,
    revision: str,
    source_split: str,
) -> list[dict[str, Any]]:
    """Attach content-derived record and prompt IDs without using row order."""

    records: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    content_occurrences: dict[str, int] = defaultdict(int)
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        _validate_source_row(row, kind, index)
        prompt_payload = {"instruction": row["instruction"], "input": row.get("input", "")}
        prompt_id = f"prompt_{sha256_bytes(canonical_json_bytes(prompt_payload))}"
        source_record_payload = {
            "repo_id": repo_id,
            "revision": revision,
            "source_split": source_split,
            "kind": kind,
            "row": row,
        }
        content_key = sha256_bytes(canonical_json_bytes(source_record_payload))
        duplicate_ordinal = content_occurrences[content_key]
        content_occurrences[content_key] += 1
        record_payload = {
            **source_record_payload,
            "duplicate_ordinal": duplicate_ordinal,
        }
        record_id = f"{kind}_{sha256_bytes(canonical_json_bytes(record_payload))}"
        if record_id in seen_record_ids:
            raise SplitBuildError(
                f"Duplicate content-addressed record in {asset_name}/{source_split}: {record_id}"
            )
        seen_record_ids.add(record_id)
        row.update(
            {
                RECORD_ID_FIELD: record_id,
                PROMPT_ID_FIELD: prompt_id,
                DUPLICATE_ORDINAL_FIELD: duplicate_ordinal,
                "_split_source_asset": asset_name,
                "_split_source_revision": revision,
                "_split_source_split": source_split,
            }
        )
        records.append(row)
    return records


def _assignment_key(seed: str, prompt_id: str) -> str:
    return sha256_bytes(f"{seed}\0{prompt_id}".encode("utf-8"))


def allocate_grouped_records(
    records: Iterable[Mapping[str, Any]],
    allocations: Mapping[str, int | float],
    *,
    seed: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Hash-sort prompt groups and fill exact largest-remainder record quotas."""

    materialized = [dict(record) for record in records]
    targets = compute_target_counts(len(materialized), allocations)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in materialized:
        prompt_id = record.get(PROMPT_ID_FIELD)
        if not isinstance(prompt_id, str) or not prompt_id:
            raise SplitBuildError("Cannot allocate a record without a prompt ID")
        groups[prompt_id].append(record)

    ordered_groups = sorted(
        (
            sorted(group, key=lambda record: record[RECORD_ID_FIELD])
            for group in groups.values()
        ),
        key=lambda group: (_assignment_key(seed, group[0][PROMPT_ID_FIELD]), group[0][PROMPT_ID_FIELD]),
    )
    remaining = ordered_groups
    assigned: dict[str, list[dict[str, Any]]] = {}
    roles = list(allocations)
    for role in roles[:-1]:
        target = targets[role]
        selected: list[list[dict[str, Any]]] = []
        deferred: list[list[dict[str, Any]]] = []
        selected_count = 0
        for group in remaining:
            if selected_count < target and selected_count + len(group) <= target:
                selected.append(group)
                selected_count += len(group)
            else:
                deferred.append(group)
        if selected_count != target:
            raise SplitBuildError(
                f"Prompt grouping prevents exact quota for {role}: target={target}, "
                f"reachable={selected_count}. Resolve duplicate-prompt policy explicitly."
            )
        assigned[role] = [dict(record, **{ROLE_FIELD: role}) for group in selected for record in group]
        remaining = deferred

    last_role = roles[-1]
    last_records = [dict(record, **{ROLE_FIELD: last_role}) for group in remaining for record in group]
    if len(last_records) != targets[last_role]:
        raise SplitBuildError(
            f"Prompt grouping prevents exact quota for {last_role}: "
            f"target={targets[last_role]}, found={len(last_records)}"
        )
    assigned[last_role] = last_records
    return assigned, targets


def assign_preserved_role(records: Iterable[Mapping[str, Any]], role: str, *, seed: str) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda record: (
            _assignment_key(f"{seed}:{role}", record[PROMPT_ID_FIELD]),
            record[RECORD_ID_FIELD],
        ),
    )
    return [dict(record, **{ROLE_FIELD: role}) for record in ordered]


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record))
            handle.write(b"\n")


def _write_ids(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    ids = [str(record[RECORD_ID_FIELD]) for record in records]
    path.write_text("".join(f"{record_id}\n" for record_id in ids), encoding="utf-8")


def write_split_bundle(
    output_root: Path,
    *,
    split_records: Mapping[str, list[dict[str, Any]]],
    split_kinds: Mapping[str, str],
    split_targets: Mapping[str, int | None],
    provenance: Mapping[str, Any],
    overlap_audit: Mapping[str, Any],
) -> Path:
    data_directory = output_root / "splits"
    ids_directory = output_root / "ids"
    data_directory.mkdir(parents=True)
    ids_directory.mkdir(parents=True)

    split_entries: dict[str, dict[str, Any]] = {}
    for role, records in split_records.items():
        data_path = data_directory / f"{role}.jsonl"
        ids_path = ids_directory / f"{role}.txt"
        _write_jsonl(data_path, records)
        _write_ids(ids_path, records)
        split_entries[role] = {
            "kind": split_kinds[role],
            "count": len(records),
            "target_count": split_targets.get(role),
            "prompt_count": len({record[PROMPT_ID_FIELD] for record in records}),
            "data_file": data_path.relative_to(output_root).as_posix(),
            "data_sha256": sha256_file(data_path),
            "ids_file": ids_path.relative_to(output_root).as_posix(),
            "ids_sha256": sha256_file(ids_path),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        **dict(provenance),
        "splits": split_entries,
        "overlap_audit": dict(overlap_audit),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_hash = sha256_file(manifest_path)
    (output_root / "manifest.sha256").write_text(
        f"{manifest_hash}  manifest.json\n",
        encoding="utf-8",
    )
    return manifest_path


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SplitBuildError("PyYAML is required; install the cluster environment first") from exc
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SplitBuildError(f"Split config does not exist: {path}") from exc
    if not isinstance(config, dict):
        raise SplitBuildError("Split config root must be a mapping")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SplitBuildError(f"Split config schema must be {SCHEMA_VERSION}")
    return config


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _asset_entry(source_manifest_path: Path, name: str) -> dict[str, Any]:
    entries = {entry["name"]: entry for entry in load_assets(source_manifest_path)}
    try:
        return entries[name]
    except KeyError as exc:
        raise SplitBuildError(f"Unknown asset in split config: {name}") from exc


def _verify_local_asset(asset: dict[str, Any], path: Path) -> None:
    errors = verify_asset(asset, path)
    if errors:
        detail = "\n".join(f"  - {error}" for error in errors)
        raise SplitBuildError(f"Local asset verification failed for {asset['name']}:\n{detail}")


def _resolve_split_data_files(
    asset: Mapping[str, Any],
    asset_root: Path,
    configured_files: Any,
    split_names: Iterable[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Resolve only manifest-verified files declared for each source split."""

    if not isinstance(configured_files, Mapping):
        raise SplitBuildError(f"{asset['name']} data_files must be a split-to-files mapping")

    verified_paths = {
        Path(required_file["path"]).as_posix()
        for required_file in asset.get("required_files", [])
    }
    root = asset_root.resolve()
    resolved: dict[str, list[str]] = {}
    provenance: dict[str, list[str]] = {}
    seen_paths: set[Path] = set()
    for split in split_names:
        entries = configured_files.get(split)
        if isinstance(entries, str):
            entries = [entries]
        if not isinstance(entries, list) or not entries:
            raise SplitBuildError(f"{asset['name']} data_files.{split} must be a non-empty list")

        local_paths: list[str] = []
        relative_paths: list[str] = []
        for entry in entries:
            if not isinstance(entry, str) or not entry:
                raise SplitBuildError(
                    f"{asset['name']} data_files.{split} contains an invalid path: {entry!r}"
                )
            relative = Path(entry)
            if relative.is_absolute() or ".." in relative.parts:
                raise SplitBuildError(
                    f"{asset['name']} data_files.{split} contains an unsafe path: {entry}"
                )
            relative_posix = relative.as_posix()
            if relative_posix not in verified_paths:
                raise SplitBuildError(
                    f"{asset['name']} data_files.{split} is not in the verified source manifest: "
                    f"{relative_posix}"
                )
            local_path = (root / relative).resolve()
            if not local_path.is_relative_to(root) or not local_path.is_file():
                raise SplitBuildError(
                    f"{asset['name']} data_files.{split} does not exist under the asset root: "
                    f"{relative_posix}"
                )
            if local_path in seen_paths:
                raise SplitBuildError(
                    f"{asset['name']} data file is assigned more than once: {relative_posix}"
                )
            seen_paths.add(local_path)
            local_paths.append(str(local_path))
            relative_paths.append(relative_posix)

        resolved[split] = local_paths
        provenance[split] = relative_paths
    return resolved, provenance


def _validate_expected_source_counts(
    source_config: Mapping[str, Any],
    rows_by_split: Mapping[str, list[dict]],
    *,
    asset_name: str,
) -> None:
    expected = source_config.get("expected_source_counts")
    if expected is None:
        return
    if not isinstance(expected, Mapping):
        raise SplitBuildError(f"{asset_name} expected_source_counts must be a mapping")
    for split, rows in rows_by_split.items():
        expected_count = expected.get(split)
        if type(expected_count) is not int or expected_count < 0:
            raise SplitBuildError(
                f"{asset_name} expected_source_counts.{split} must be a non-negative integer"
            )
        if len(rows) != expected_count:
            raise SplitBuildError(
                f"{asset_name}/{split} row-count mismatch: expected {expected_count}, "
                f"found {len(rows)}. Check the pinned revision and explicit data_files; "
                "do not continue with duplicated or unrelated JSON files."
            )


def _filter_ppo_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    try:
        from model_training.custom_datasets.utils import _filter_by_words
    except ImportError as exc:
        raise SplitBuildError("Pinned Open-Assistant must be installed before building PPO splits") from exc

    accepted: list[dict[str, Any]] = []
    rejected = 0
    for row in rows:
        materialized = dict(row)
        instruction = materialized["instruction"]
        input_text = materialized.get("input", "")
        prompt = f"{instruction}\n{input_text}" if input_text else instruction
        if _filter_by_words(prompt) is None or _filter_by_words(materialized["output"]) is None:
            rejected += 1
        else:
            accepted.append(materialized)
    return accepted, rejected


def _load_source_rows(config: dict[str, Any]) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SplitBuildError("Hugging Face datasets is required; install the cluster environment first") from exc

    source_manifest_path = _resolve_project_path(config["source_manifest"])
    preference_config = config["preference"]
    ppo_config = config["ppo"]
    preference_asset = _asset_entry(source_manifest_path, preference_config["asset_name"])
    ppo_asset = _asset_entry(source_manifest_path, ppo_config["asset_name"])
    preference_path = _resolve_project_path(preference_config["asset_path"])
    ppo_path = _resolve_project_path(ppo_config["asset_path"])
    _verify_local_asset(preference_asset, preference_path)
    _verify_local_asset(ppo_asset, ppo_path)

    preference_split_names = [
        preference_config["allocation_source_split"],
        *preference_config["preserved_splits"].values(),
    ]
    preference_data_files, preference_file_provenance = _resolve_split_data_files(
        preference_asset,
        preference_path,
        preference_config.get("data_files"),
        preference_split_names,
    )
    preference_loaded = load_dataset(
        "json",
        data_files=preference_data_files,
        split=preference_split_names,
    )
    preference_rows = {
        split: [dict(row) for row in dataset]
        for split, dataset in zip(preference_split_names, preference_loaded)
    }
    _validate_expected_source_counts(
        preference_config,
        preference_rows,
        asset_name=preference_asset["name"],
    )

    ppo_split_names = [ppo_config["allocation_source_split"], *ppo_config["preserved_splits"].values()]
    ppo_data_files, ppo_file_provenance = _resolve_split_data_files(
        ppo_asset,
        ppo_path,
        ppo_config.get("data_files"),
        ppo_split_names,
    )
    ppo_loaded = load_dataset(
        "json",
        data_files=ppo_data_files,
        split=ppo_split_names,
    )
    ppo_raw_rows = {
        split: [dict(row) for row in dataset]
        for split, dataset in zip(ppo_split_names, ppo_loaded)
    }
    _validate_expected_source_counts(
        ppo_config,
        ppo_raw_rows,
        asset_name=ppo_asset["name"],
    )
    ppo_rows: dict[str, list[dict]] = {}
    ppo_rejections: dict[str, int] = {}
    for split in ppo_split_names:
        filtered, rejected = _filter_ppo_rows(ppo_raw_rows[split])
        ppo_rows[split] = filtered
        ppo_rejections[split] = rejected

    source_metadata = {
        "source_manifest": {
            "path": Path(config["source_manifest"]).as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "source_assets": {
            "preference": {
                "name": preference_asset["name"],
                "repo_id": preference_asset["repo_id"],
                "revision": preference_asset["revision"],
                "data_files": preference_file_provenance,
                "source_counts": {split: len(rows) for split, rows in preference_rows.items()},
                "filter": "none",
            },
            "ppo": {
                "name": ppo_asset["name"],
                "repo_id": ppo_asset["repo_id"],
                "revision": ppo_asset["revision"],
                "dataset_config": ppo_config["dataset_config"],
                "data_files": ppo_file_provenance,
                "source_counts_before_legacy_filter": {
                    split: len(rows) for split, rows in ppo_raw_rows.items()
                },
                "source_counts_after_legacy_filter": {
                    split: len(rows) for split, rows in ppo_rows.items()
                },
                "legacy_filter_rejections": ppo_rejections,
                "filter": "Open-Assistant _filter_by_words, matching Coste load_alpaca_dataset",
            },
        },
    }
    return {"preference": preference_rows, "ppo": ppo_rows}, source_metadata


def build_from_config(config_path: Path, output_override: Path | None = None) -> Path:
    config = _load_yaml_config(config_path)
    seed = config.get("assignment_seed")
    if not isinstance(seed, str) or not seed:
        raise SplitBuildError("assignment_seed must be a non-empty string")
    output_root = output_override.resolve() if output_override else _resolve_project_path(config["output_root"])
    if output_root.exists():
        raise SplitBuildError(
            f"Output already exists: {output_root}. Verify it or choose a new --output-root; "
            "the builder never overwrites an experiment split."
        )

    source_rows, source_metadata = _load_source_rows(config)
    preference_config = config["preference"]
    ppo_config = config["ppo"]
    preference_asset = source_metadata["source_assets"]["preference"]
    ppo_asset = source_metadata["source_assets"]["ppo"]

    preference_by_source = {
        split: annotate_source_records(
            rows,
            kind="preference",
            asset_name=preference_asset["name"],
            repo_id=preference_asset["repo_id"],
            revision=preference_asset["revision"],
            source_split=split,
        )
        for split, rows in source_rows["preference"].items()
    }
    ppo_by_source = {
        split: annotate_source_records(
            rows,
            kind="prompt",
            asset_name=ppo_asset["name"],
            repo_id=ppo_asset["repo_id"],
            revision=ppo_asset["revision"],
            source_split=split,
        )
        for split, rows in source_rows["ppo"].items()
    }
    source_metadata["source_assets"]["preference"]["exact_duplicate_occurrences"] = {
        split: sum(record[DUPLICATE_ORDINAL_FIELD] > 0 for record in records)
        for split, records in preference_by_source.items()
    }
    source_metadata["source_assets"]["ppo"]["exact_duplicate_occurrences"] = {
        split: sum(record[DUPLICATE_ORDINAL_FIELD] > 0 for record in records)
        for split, records in ppo_by_source.items()
    }

    preference_allocated, preference_targets = allocate_grouped_records(
        preference_by_source[preference_config["allocation_source_split"]],
        preference_config["allocations"],
        seed=f"{seed}:preference",
    )
    ppo_allocated, ppo_targets = allocate_grouped_records(
        ppo_by_source[ppo_config["allocation_source_split"]],
        ppo_config["allocations"],
        seed=f"{seed}:ppo",
    )

    split_records: dict[str, list[dict[str, Any]]] = {**preference_allocated, **ppo_allocated}
    split_kinds = {role: "preference" for role in preference_allocated}
    split_kinds.update({role: "prompt" for role in ppo_allocated})
    split_targets: dict[str, int | None] = {**preference_targets, **ppo_targets}
    for role, source_split in preference_config["preserved_splits"].items():
        split_records[role] = assign_preserved_role(
            preference_by_source[source_split], role, seed=seed
        )
        split_kinds[role] = "preference"
        split_targets[role] = None
    for role, source_split in ppo_config["preserved_splits"].items():
        split_records[role] = assign_preserved_role(ppo_by_source[source_split], role, seed=seed)
        split_kinds[role] = "prompt"
        split_targets[role] = None

    logical_rm_prompts = {
        record[PROMPT_ID_FIELD]
        for role in preference_allocated
        for record in split_records[role]
    }
    logical_ppo_prompts = {
        record[PROMPT_ID_FIELD] for role in ppo_allocated for record in split_records[role]
    }
    logical_overlap = logical_rm_prompts.intersection(logical_ppo_prompts)
    all_rm_prompts = {
        record[PROMPT_ID_FIELD]
        for role, records in split_records.items()
        if split_kinds[role] == "preference"
        for record in records
    }
    all_ppo_prompts = {
        record[PROMPT_ID_FIELD]
        for role, records in split_records.items()
        if split_kinds[role] == "prompt"
        for record in records
    }
    all_overlap = all_rm_prompts.intersection(all_ppo_prompts)
    overlap_policy = config.get("cross_source_prompt_overlap_policy", "forbid")
    if overlap_policy not in {"forbid", "allow_coste_native_and_report"}:
        raise SplitBuildError(
            "cross_source_prompt_overlap_policy must be 'forbid' or "
            "'allow_coste_native_and_report'"
        )
    if overlap_policy == "forbid" and all_overlap:
        examples = ", ".join(sorted(all_overlap)[:5])
        raise SplitBuildError(
            f"Detected {len(all_overlap)} preference/PPO prompt overlaps; examples: {examples}"
        )

    preference_roles = sorted(
        role for role, kind in split_kinds.items() if kind == "preference"
    )
    ppo_roles = sorted(role for role, kind in split_kinds.items() if kind == "prompt")
    prompt_ids_by_role = {
        role: {record[PROMPT_ID_FIELD] for record in split_records[role]}
        for role in split_records
    }
    overlap_by_role_pair = {
        preference_role: {
            ppo_role: len(
                prompt_ids_by_role[preference_role].intersection(prompt_ids_by_role[ppo_role])
            )
            for ppo_role in ppo_roles
        }
        for preference_role in preference_roles
    }

    provenance = {
        "name": config["name"],
        "assignment": {
            "seed": seed,
            "algorithm": "content-plus-duplicate-ordinal IDs; SHA-256 prompt-group ordering; exact largest-remainder quotas",
            "preference_allocations": dict(preference_config["allocations"]),
            "ppo_allocations": dict(ppo_config["allocations"]),
            "original_validation_policy": "preserved as external splits",
        },
        **source_metadata,
        "split_config": {
            "path": config_path.resolve().relative_to(ROOT).as_posix()
            if config_path.resolve().is_relative_to(ROOT)
            else str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "creation_code": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    overlap_audit = {
        "policy": overlap_policy,
        "logical_cross_source_prompt_count": len(logical_overlap),
        "all_preserved_cross_source_prompt_count": len(all_overlap),
        "by_role_pair": overlap_by_role_pair,
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.building-", dir=output_root.parent) as temp:
        temporary_root = Path(temp)
        manifest_path = write_split_bundle(
            temporary_root,
            split_records=split_records,
            split_kinds=split_kinds,
            split_targets=split_targets,
            provenance=provenance,
            overlap_audit=overlap_audit,
        )
        verify_split_manifest(manifest_path)
        temporary_root.replace(output_root)

    return output_root / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing bundle selected by config/--output-root without loading source data.",
    )
    return parser


def _print_bundle_summary(manifest_path: Path, counts: Mapping[str, int], *, built: bool) -> None:
    prefix = "PASS built" if built else "PASS"
    print(f"{prefix} {manifest_path}")
    for role, count in counts.items():
        print(f"  {role}: {count}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlap = manifest["overlap_audit"]
    print(f"  cross_source_prompt_overlap_policy: {overlap.get('policy', 'forbid')}")
    print(
        "  logical_cross_source_prompt_count: "
        f"{overlap['logical_cross_source_prompt_count']}"
    )
    print(
        "  all_preserved_cross_source_prompt_count: "
        f"{overlap['all_preserved_cross_source_prompt_count']}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config.expanduser().resolve()
        config = _load_yaml_config(config_path)
        output_root = (
            args.output_root.expanduser().resolve()
            if args.output_root
            else _resolve_project_path(config["output_root"])
        )
        if args.verify_only:
            manifest_path = output_root / "manifest.json"
            counts = verify_split_manifest(manifest_path)
            _print_bundle_summary(manifest_path, counts, built=False)
            return 0

        manifest_path = build_from_config(config_path, output_root)
        counts = verify_split_manifest(manifest_path)
        _print_bundle_summary(manifest_path, counts, built=True)
        return 0
    except (SplitBuildError, SplitManifestError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
