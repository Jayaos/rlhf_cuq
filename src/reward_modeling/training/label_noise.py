"""Deterministic, auditable preference-label corruption for RM ablations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data_utils.split_manifest import RECORD_ID_FIELD


SCHEMA_VERSION = "1.0.0"
SELECTION_ALGORITHM = "sha256_rank_exact_floor_v1"
TRAIN_ROLE = "D_rm_train"
METADATA_FILENAME = "rm_label_noise_metadata.json"
FLIPPED_IDS_FILENAME = "rm_label_noise_flipped_record_ids.txt"


@dataclass(frozen=True)
class LabelNoiseResult:
    """Corrupted row copies plus compact and full selection provenance."""

    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    flipped_record_ids: tuple[str, ...]


def validate_label_noise_config(rate: Any, seed: Any) -> tuple[float, int]:
    """Normalize and validate the user-declared corruption configuration."""

    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ValueError("rm_label_noise_rate must be a finite number in [0, 1]")
    normalized_rate = float(rate)
    if not math.isfinite(normalized_rate) or not 0.0 <= normalized_rate <= 1.0:
        raise ValueError("rm_label_noise_rate must be a finite number in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("rm_label_noise_seed must be a nonnegative integer")
    return normalized_rate, seed


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_ids_payload(record_ids: Iterable[str]) -> bytes:
    return "".join(f"{record_id}\n" for record_id in sorted(record_ids)).encode("utf-8")


def _selection_key(record_id: str, seed: int) -> tuple[bytes, str]:
    payload = f"{SELECTION_ALGORITHM}\0{seed}\0{record_id}".encode("utf-8")
    return hashlib.sha256(payload).digest(), record_id


def apply_preference_label_noise(
    rows: Sequence[Mapping[str, Any]],
    *,
    rate: Any,
    seed: Any,
    manifest_sha256: str,
    role: str = TRAIN_ROLE,
) -> LabelNoiseResult:
    """Flip an exact seeded subset of binary preferences without mutation.

    Selection is independent of input iteration order because it ranks stable
    split-record IDs. Only the binary ``preference`` field changes.
    """

    normalized_rate, normalized_seed = validate_label_noise_config(rate, seed)
    if role != TRAIN_ROLE:
        raise ValueError(f"Label noise is restricted to {TRAIN_ROLE}, received {role!r}")
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("manifest_sha256 must be a hexadecimal SHA-256 digest")
    try:
        bytes.fromhex(manifest_sha256)
    except ValueError as exc:
        raise ValueError("manifest_sha256 must be a hexadecimal SHA-256 digest") from exc

    copied_rows: list[dict[str, Any]] = []
    record_ids: list[str] = []
    for row in rows:
        record_id = row.get(RECORD_ID_FIELD)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Every {role} row must have a stable {RECORD_ID_FIELD}")
        preference = row.get("preference")
        if isinstance(preference, bool) or not isinstance(preference, int) or preference not in (0, 1):
            raise ValueError(
                f"Every {role} preference must be binary for label flipping: "
                f"{record_id} has {preference!r}"
            )
        copied_rows.append(dict(row))
        record_ids.append(record_id)

    if len(set(record_ids)) != len(record_ids):
        raise ValueError(f"{role} contains duplicate {RECORD_ID_FIELD} values")

    flip_count = math.floor(normalized_rate * len(copied_rows))
    selected = {
        record_id
        for record_id in sorted(record_ids, key=lambda value: _selection_key(value, normalized_seed))[
            :flip_count
        ]
    }
    for row in copied_rows:
        if row[RECORD_ID_FIELD] in selected:
            row["preference"] = 1 - row["preference"]

    canonical_ids = tuple(sorted(selected))
    ids_payload = _canonical_ids_payload(canonical_ids)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "enabled": bool(flip_count),
        "role": role,
        "requested_rate": normalized_rate,
        "realized_rate": flip_count / len(copied_rows) if copied_rows else 0.0,
        "seed": normalized_seed,
        "selection_algorithm": SELECTION_ALGORITHM,
        "rounding": "floor",
        "train_record_count": len(copied_rows),
        "flip_count": flip_count,
        "manifest_sha256": manifest_sha256,
        "flipped_record_ids_file": FLIPPED_IDS_FILENAME,
        "flipped_record_ids_sha256": hashlib.sha256(ids_payload).hexdigest(),
    }
    return LabelNoiseResult(copied_rows, metadata, canonical_ids)


def _write_or_validate(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"Existing label-noise provenance does not match this run: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def persist_label_noise_provenance(
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    flipped_record_ids: Iterable[str],
) -> None:
    """Atomically create, or exactly validate, noisy-RM provenance files."""

    root = Path(output_dir)
    ids_payload = _canonical_ids_payload(flipped_record_ids)
    if hashlib.sha256(ids_payload).hexdigest() != metadata.get("flipped_record_ids_sha256"):
        raise ValueError("Flipped-record ID payload does not match label-noise metadata")
    metadata_payload = (
        json.dumps(dict(metadata), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_or_validate(root / FLIPPED_IDS_FILENAME, ids_payload)
    _write_or_validate(root / METADATA_FILENAME, metadata_payload)


def validate_persisted_label_noise_provenance(
    output_dir: str | Path,
    *,
    expected_rate: float | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Validate root artifacts and the metadata embedded in model config."""

    root = Path(output_dir)
    metadata_path = root / METADATA_FILENAME
    ids_path = root / FLIPPED_IDS_FILENAME
    config_path = root / "config.json"
    for path in (metadata_path, ids_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Noisy RM provenance is missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if model_config.get("rm_label_noise") != metadata:
        raise RuntimeError("Model config and root label-noise metadata disagree")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported RM label-noise metadata schema")
    if metadata.get("selection_algorithm") != SELECTION_ALGORITHM:
        raise RuntimeError("Unsupported RM label-noise selection algorithm")
    if metadata.get("role") != TRAIN_ROLE or not metadata.get("enabled"):
        raise RuntimeError("Persisted RM metadata is not an enabled D_rm_train noise run")

    record_ids = tuple(ids_path.read_text(encoding="utf-8").splitlines())
    if record_ids != tuple(sorted(set(record_ids))):
        raise RuntimeError("Flipped-record ID file must be sorted and unique")
    ids_payload = _canonical_ids_payload(record_ids)
    if hashlib.sha256(ids_payload).hexdigest() != metadata.get("flipped_record_ids_sha256"):
        raise RuntimeError("Flipped-record ID hash does not match metadata")
    if len(record_ids) != metadata.get("flip_count"):
        raise RuntimeError("Flipped-record ID count does not match metadata")
    expected_flips = math.floor(
        float(metadata["requested_rate"]) * int(metadata["train_record_count"])
    )
    if expected_flips != metadata.get("flip_count"):
        raise RuntimeError("Persisted flip count does not implement exact floor rounding")

    if expected_rate is not None and float(metadata.get("requested_rate")) != float(expected_rate):
        raise RuntimeError(
            f"RM label-noise rate mismatch: expected {expected_rate}, "
            f"found {metadata.get('requested_rate')}"
        )
    if expected_seed is not None and metadata.get("seed") != expected_seed:
        raise RuntimeError(
            f"RM label-noise seed mismatch: expected {expected_seed}, found {metadata.get('seed')}"
        )
    return metadata


def load_optional_label_noise_metadata(model_dir: str | Path) -> dict[str, Any] | None:
    """Return validated noisy-RM provenance, or ``None`` for a clean RM.

    Historical clean checkpoints predate this ablation and have no dedicated
    provenance files. Any partial indication of label noise is treated as a
    corrupt checkpoint rather than silently classified as clean.
    """

    root = Path(model_dir)
    config_path = root / "config.json"
    metadata_path = root / METADATA_FILENAME
    ids_path = root / FLIPPED_IDS_FILENAME
    config_metadata = None
    if config_path.is_file():
        config_metadata = json.loads(config_path.read_text(encoding="utf-8")).get(
            "rm_label_noise"
        )

    if config_metadata is None and not metadata_path.exists() and not ids_path.exists():
        return None
    return validate_persisted_label_noise_provenance(root)
