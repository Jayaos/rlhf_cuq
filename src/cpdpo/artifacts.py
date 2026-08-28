"""Fingerprint and atomic-write helpers for scientific artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(root: str | Path, relative_paths: list[str]) -> str:
    """Hash names and bytes of a declared, ordered file set."""

    base = Path(root).resolve()
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Fingerprint path escapes root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def model_fingerprint(path: str | Path) -> str:
    root = Path(path)
    weight_files = sorted(
        item.name for item in root.iterdir() if item.is_file() and item.suffix in {".bin", ".safetensors"}
    )
    required = ["config.json", *weight_files]
    if len(required) == 1:
        raise FileNotFoundError(f"No model weights found under {root}")
    return fingerprint_files(root, required)


def tokenizer_fingerprint(path: str | Path) -> str:
    root = Path(path)
    candidates = [
        name
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        )
        if (root / name).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No tokenizer files found under {root}")
    return fingerprint_files(root, candidates)


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_revision(root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(root), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
