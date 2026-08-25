#!/usr/bin/env python3
"""Audit pinned research sources without downloading model or dataset payloads.

The command reads ``artifacts/source_manifest.json``, verifies the vendored
Coste baseline against its pinned content fingerprints, and (unless
``--offline`` is used) queries only GitHub and Hugging Face metadata APIs.  It
never mutates a remote asset and it does not download weights or dataset rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


SCHEMA_VERSION = "1.0.0"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "artifacts" / "source_manifest.json"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
USER_AGENT = "rlhf-uq-source-audit/1.0"
UTF8_LF_CANONICALIZATION = "utf8_lf"


class AuditError(RuntimeError):
    """Raised when an audit input or metadata request cannot be validated."""


def _result(name: str, status: str, detail: str, **evidence: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"name": name, "status": status, "detail": detail}
    if evidence:
        item["evidence"] = evidence
    return item


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"Manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"Manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AuditError("Manifest root must be a JSON object")
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required_sections = (
        "schema_version",
        "repositories",
        "huggingface_assets",
        "dataset_contracts",
        "legacy_baseline_files",
    )
    missing = [key for key in required_sections if key not in manifest]
    if missing:
        checks.append(_result("manifest.sections", "fail", f"Missing sections: {', '.join(missing)}"))
        return checks

    if manifest["schema_version"] != SCHEMA_VERSION:
        checks.append(
            _result(
                "manifest.schema_version",
                "fail",
                f"Expected {SCHEMA_VERSION}, found {manifest['schema_version']!r}",
            )
        )
    else:
        checks.append(_result("manifest.schema_version", "pass", SCHEMA_VERSION))

    names: list[str] = []
    repositories = manifest.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        checks.append(_result("manifest.repositories", "fail", "Section must be a non-empty list"))
        repositories = []
    for index, entry in enumerate(repositories):
        prefix = f"manifest.repositories[{index}]"
        if not isinstance(entry, dict):
            checks.append(_result(prefix, "fail", "Entry must be an object"))
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            checks.append(_result(f"{prefix}.name", "fail", "Missing non-empty name"))
        else:
            names.append(name)
        if entry.get("provider") != "github":
            checks.append(_result(f"{prefix}.provider", "fail", "Only the audited github provider is supported"))
        repository = entry.get("repository")
        if not isinstance(repository, str) or repository.count("/") != 1:
            checks.append(_result(f"{prefix}.repository", "fail", "Expected an owner/repository identifier"))
        _validate_revision(checks, f"{prefix}.revision", entry.get("revision"))
        _validate_non_empty_string(checks, f"{prefix}.license", entry.get("license"))
        package_versions = entry.get("package_versions")
        if not isinstance(package_versions, dict) or not package_versions or any(
            not isinstance(package, str)
            or not package
            or not isinstance(version, str)
            or not version
            for package, version in (package_versions.items() if isinstance(package_versions, dict) else [])
        ):
            checks.append(_result(f"{prefix}.package_versions", "fail", "Package version metadata is required"))

    assets = manifest.get("huggingface_assets")
    if not isinstance(assets, list) or not assets:
        checks.append(_result("manifest.huggingface_assets", "fail", "Section must be a non-empty list"))
        assets = []
    assets_by_name: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(assets):
        prefix = f"manifest.huggingface_assets[{index}]"
        if not isinstance(entry, dict):
            checks.append(_result(prefix, "fail", "Entry must be an object"))
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            checks.append(_result(f"{prefix}.name", "fail", "Missing non-empty name"))
        else:
            names.append(name)
            assets_by_name[name] = entry
        if entry.get("kind") not in {"model", "dataset"}:
            checks.append(_result(f"{prefix}.kind", "fail", "Kind must be model or dataset"))
        repo_id = entry.get("repo_id")
        if not isinstance(repo_id, str) or repo_id.count("/") != 1:
            checks.append(_result(f"{prefix}.repo_id", "fail", "Expected an owner/repository identifier"))
        _validate_revision(checks, f"{prefix}.revision", entry.get("revision"))
        _validate_non_empty_string(checks, f"{prefix}.license", entry.get("license"))

        files = entry.get("required_files")
        if not isinstance(files, list) or not files:
            checks.append(_result(f"{prefix}.required_files", "fail", "At least one required file is needed"))
            continue
        seen_paths: set[str] = set()
        for file_index, required_file in enumerate(files):
            file_prefix = f"{prefix}.required_files[{file_index}]"
            if not isinstance(required_file, dict):
                checks.append(_result(file_prefix, "fail", "File entry must be an object"))
                continue
            path = required_file.get("path")
            if not _safe_relative_path(path):
                checks.append(_result(f"{file_prefix}.path", "fail", "Expected a safe relative file path"))
            elif path in seen_paths:
                checks.append(_result(f"{file_prefix}.path", "fail", f"Duplicate required file: {path}"))
            else:
                seen_paths.add(path)
            size = required_file.get("size")
            if type(size) is not int or size <= 0:
                checks.append(_result(f"{file_prefix}.size", "fail", "Size must be a positive integer"))
            blob = required_file.get("git_blob_sha1")
            if not isinstance(blob, str) or not SHA1_RE.fullmatch(blob):
                checks.append(_result(f"{file_prefix}.git_blob_sha1", "fail", "Invalid Git blob SHA-1"))
            lfs_hash = required_file.get("lfs_sha256")
            if lfs_hash is not None and (not isinstance(lfs_hash, str) or not SHA256_RE.fullmatch(lfs_hash)):
                checks.append(_result(f"{file_prefix}.lfs_sha256", "fail", "Invalid LFS SHA-256"))

    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        checks.append(_result("manifest.unique_names", "fail", f"Duplicate names: {', '.join(duplicates)}"))
    else:
        checks.append(_result("manifest.unique_names", "pass", f"{len(names)} unique source names"))

    contracts = manifest.get("dataset_contracts")
    if not isinstance(contracts, list):
        checks.append(_result("manifest.dataset_contracts", "fail", "Section must be a list"))
        contracts = []
    for index, contract in enumerate(contracts):
        prefix = f"manifest.dataset_contracts[{index}]"
        if not isinstance(contract, dict):
            checks.append(_result(prefix, "fail", "Entry must be an object"))
            continue
        source_name = contract.get("source_asset_name")
        source_asset = assets_by_name.get(source_name)
        if source_asset is None or source_asset.get("kind") != "dataset":
            checks.append(_result(f"{prefix}.source_asset_name", "fail", "Must name a dataset asset"))
        else:
            for key in ("repo_id", "revision"):
                if contract.get(key) != source_asset.get(key):
                    checks.append(_result(f"{prefix}.{key}", "fail", f"Does not match {source_name}"))
        _validate_revision(checks, f"{prefix}.revision", contract.get("revision"))
        _validate_non_empty_string(checks, f"{prefix}.config", contract.get("config"))
        splits = contract.get("splits")
        if not isinstance(splits, dict) or not splits or any(
            not isinstance(split, str) or not split or type(rows) is not int or rows <= 0
            for split, rows in (splits.items() if isinstance(splits, dict) else [])
        ):
            checks.append(_result(f"{prefix}.splits", "fail", "Expected non-empty split names and row counts"))
        total_rows = contract.get("total_rows")
        if type(total_rows) is not int or total_rows <= 0:
            checks.append(_result(f"{prefix}.total_rows", "fail", "Total rows must be a positive integer"))
        elif isinstance(splits, dict) and all(type(rows) is int for rows in splits.values()):
            if total_rows != sum(splits.values()):
                checks.append(_result(f"{prefix}.total_rows", "fail", "Total does not equal split row counts"))
        features = contract.get("features")
        if not isinstance(features, dict) or not features or any(
            not isinstance(name, str) or not name or not isinstance(dtype, str) or not dtype
            for name, dtype in (features.items() if isinstance(features, dict) else [])
        ):
            checks.append(_result(f"{prefix}.features", "fail", "Expected a non-empty name-to-type mapping"))
        observation = contract.get("observation")
        if not isinstance(observation, dict) or not observation.get("date") or not observation.get("method"):
            checks.append(_result(f"{prefix}.observation", "fail", "Recorded observation provenance is required"))

    baseline_entries = manifest.get("legacy_baseline_files")
    if not isinstance(baseline_entries, list) or not baseline_entries:
        checks.append(_result("manifest.legacy_baseline_files", "fail", "At least one file is required"))
    else:
        baseline_failures = 0
        seen_baseline_paths: set[str] = set()
        for index, entry in enumerate(baseline_entries):
            prefix = f"manifest.legacy_baseline_files[{index}]"
            if not isinstance(entry, dict):
                checks.append(_result(prefix, "fail", "Entry must be an object"))
                baseline_failures += 1
                continue
            path = entry.get("path")
            if not _safe_relative_path(path):
                checks.append(_result(f"{prefix}.path", "fail", "Expected a safe relative file path"))
                baseline_failures += 1
            elif path in seen_baseline_paths:
                checks.append(_result(f"{prefix}.path", "fail", f"Duplicate baseline file: {path}"))
                baseline_failures += 1
            else:
                seen_baseline_paths.add(path)
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                checks.append(_result(f"{prefix}.sha256", "fail", "Invalid SHA-256"))
                baseline_failures += 1
            size = entry.get("size")
            if type(size) is not int or size <= 0:
                checks.append(_result(f"{prefix}.size", "fail", "Size must be a positive integer"))
                baseline_failures += 1
            canonicalization = entry.get("canonicalization")
            if canonicalization not in {None, UTF8_LF_CANONICALIZATION}:
                checks.append(
                    _result(
                        f"{prefix}.canonicalization",
                        "fail",
                        f"Expected {UTF8_LF_CANONICALIZATION!r} or no canonicalization",
                    )
                )
                baseline_failures += 1
        if baseline_failures == 0:
            checks.append(_result("manifest.legacy_hashes", "pass", f"{len(baseline_entries)} hashes"))
    return checks


def _validate_revision(checks: list[dict[str, Any]], name: str, revision: Any) -> None:
    if not isinstance(revision, str) or not SHA1_RE.fullmatch(revision):
        checks.append(_result(name, "fail", "Revision must be a lowercase 40-character Git SHA"))


def _validate_non_empty_string(checks: list[dict[str, Any]], name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        checks.append(_result(name, "fail", "Expected a non-empty string"))


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def fingerprint_local_file(path: Path, canonicalization: str | None = None) -> tuple[int, str]:
    """Return the manifest size and SHA-256 for a local baseline file.

    ``utf8_lf`` makes text fingerprints stable across Git checkouts on Windows
    (CRLF) and Linux (LF). It changes only line-ending bytes; all other source
    content remains protected by the digest.
    """

    payload = path.read_bytes()
    if canonicalization == UTF8_LF_CANONICALIZATION:
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AuditError(f"Cannot apply utf8_lf to non-UTF-8 file: {path}") from exc
        payload = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    elif canonicalization is not None:
        raise AuditError(f"Unsupported baseline canonicalization: {canonicalization!r}")
    return len(payload), hashlib.sha256(payload).hexdigest()


def audit_local_baseline(manifest: dict[str, Any], workspace_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    resolved_root = workspace_root.resolve()
    for entry in manifest.get("legacy_baseline_files", []):
        relative = Path(entry["path"])
        candidate = (resolved_root / relative).resolve()
        name = f"local:{relative.as_posix()}"
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            checks.append(_result(name, "fail", "Path escapes the workspace root"))
            continue
        if not candidate.is_file():
            checks.append(_result(name, "fail", "File is missing"))
            continue
        canonicalization = entry.get("canonicalization")
        try:
            actual_size, actual_hash = fingerprint_local_file(candidate, canonicalization)
        except AuditError as exc:
            checks.append(_result(name, "fail", str(exc)))
            continue
        expected_size = entry.get("size")
        expected_hash = entry["sha256"]
        if actual_hash != expected_hash or (expected_size is not None and actual_size != expected_size):
            checks.append(
                _result(
                    name,
                    "fail",
                    "Vendored baseline differs from the pinned Coste snapshot",
                    expected_sha256=expected_hash,
                    actual_sha256=actual_hash,
                    expected_size=expected_size,
                    actual_size=actual_size,
                    canonicalization=canonicalization,
                )
            )
        else:
            checks.append(
                _result(
                    name,
                    "pass",
                    actual_hash,
                    size=actual_size,
                    canonicalization=canonicalization,
                )
            )
    return checks


def fetch_json(url: str, *, timeout: float = 20.0, attempts: int = 3) -> dict[str, Any]:
    """Fetch a JSON metadata document with bounded retries."""

    last_error: Exception | None = None
    for attempt in range(attempts):
        request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS metadata URLs
                payload = response.read()
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise AuditError(f"Expected a JSON object from {url}")
            return parsed
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError, AuditError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    raise AuditError(f"Metadata request failed after {attempts} attempts: {url}: {last_error}")


def _github_commit_url(repository: str, revision: str) -> str:
    return f"https://api.github.com/repos/{quote(repository, safe='/')}/commits/{revision}"


def audit_repositories(
    repositories: Iterable[dict[str, Any]], client: Callable[[str], dict[str, Any]]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for source in repositories:
        name = f"github:{source['name']}"
        if source.get("provider") != "github":
            checks.append(_result(name, "skip", f"Unsupported provider: {source.get('provider')!r}"))
            continue
        url = _github_commit_url(source["repository"], source["revision"])
        try:
            metadata = client(url)
            actual = metadata.get("sha")
            if actual == source["revision"]:
                checks.append(_result(name, "pass", actual, url=url))
            else:
                checks.append(
                    _result(
                        name,
                        "fail",
                        "GitHub did not resolve the requested immutable commit",
                        expected=source["revision"],
                        actual=actual,
                        url=url,
                    )
                )
        except AuditError as exc:
            checks.append(_result(name, "fail", str(exc), url=url))
    return checks


def _hf_metadata_url(asset: dict[str, Any]) -> str:
    collection = "models" if asset["kind"] == "model" else "datasets"
    repo_id = quote(asset["repo_id"], safe="/")
    revision = quote(asset["revision"], safe="")
    return f"https://huggingface.co/api/{collection}/{repo_id}/revision/{revision}?blobs=true"


def _audit_hf_file(asset_name: str, expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    check_name = f"huggingface:{asset_name}:{expected['path']}"
    mismatches: dict[str, Any] = {}
    comparisons = (
        ("size", "size"),
        ("git_blob_sha1", "blobId"),
    )
    for expected_key, actual_key in comparisons:
        if expected.get(expected_key) is not None and expected[expected_key] != actual.get(actual_key):
            mismatches[expected_key] = {"expected": expected[expected_key], "actual": actual.get(actual_key)}
    expected_lfs = expected.get("lfs_sha256")
    actual_lfs = (actual.get("lfs") or {}).get("sha256")
    if expected_lfs is not None and expected_lfs != actual_lfs:
        mismatches["lfs_sha256"] = {"expected": expected_lfs, "actual": actual_lfs}
    if mismatches:
        return _result(check_name, "fail", "File metadata mismatch", mismatches=mismatches)
    return _result(check_name, "pass", "Pinned file metadata matches")


def audit_huggingface_assets(
    assets: Iterable[dict[str, Any]], client: Callable[[str], dict[str, Any]]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for asset in assets:
        name = f"huggingface:{asset['name']}"
        url = _hf_metadata_url(asset)
        try:
            metadata = client(url)
        except AuditError as exc:
            checks.append(_result(name, "fail", str(exc), url=url))
            continue
        actual_revision = metadata.get("sha")
        if actual_revision != asset["revision"]:
            checks.append(
                _result(
                    name,
                    "fail",
                    "Resolved Hugging Face revision differs from the manifest",
                    expected=asset["revision"],
                    actual=actual_revision,
                    url=url,
                )
            )
            continue
        checks.append(_result(name, "pass", actual_revision, url=url))
        siblings = {item.get("rfilename"): item for item in metadata.get("siblings", [])}
        for expected_file in asset.get("required_files", []):
            actual_file = siblings.get(expected_file["path"])
            if actual_file is None:
                checks.append(
                    _result(
                        f"{name}:{expected_file['path']}",
                        "fail",
                        "Required file is absent at the pinned revision",
                    )
                )
            else:
                checks.append(_audit_hf_file(asset["name"], expected_file, actual_file))
    return checks


def audit_dataset_contracts(
    contracts: Iterable[dict[str, Any]], assets: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Report revision binding without fetching live viewer rows.

    Row counts and schemas are recorded observations. The Hub metadata audit
    independently verifies the immutable source files that observation names.
    """

    checks: list[dict[str, Any]] = []
    assets_by_name = {asset["name"]: asset for asset in assets}
    for contract in contracts:
        asset = assets_by_name[contract["source_asset_name"]]
        checks.append(
            _result(
                f"dataset:{contract['repo_id']}:revision_binding",
                "pass",
                "Contract matches the pinned dataset asset",
                revision=asset["revision"],
            )
        )
        checks.append(
            _result(
                f"dataset:{contract['repo_id']}:recorded_contract",
                "warn",
                "Counts/schema are a dated observation; online audit intentionally fetches no rows",
                total_rows=contract["total_rows"],
                splits=contract["splits"],
                features=contract["features"],
                observation=contract["observation"],
            )
        )
    return checks


def audit_manifest(
    manifest: dict[str, Any],
    *,
    workspace_root: Path,
    online: bool,
    client: Callable[[str], dict[str, Any]] = fetch_json,
) -> dict[str, Any]:
    checks = validate_manifest(manifest)
    if any(check["status"] == "fail" for check in checks):
        return _report(online, checks)

    checks.extend(audit_local_baseline(manifest, workspace_root))
    if online:
        checks.extend(audit_repositories(manifest["repositories"], client))
        checks.extend(audit_huggingface_assets(manifest["huggingface_assets"], client))
        checks.extend(audit_dataset_contracts(manifest["dataset_contracts"], manifest["huggingface_assets"]))
    else:
        checks.append(_result("remote_metadata", "skip", "Offline mode requested"))
    return _report(online, checks)


def _report(online: bool, checks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(check["status"] for check in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "online" if online else "offline",
        "ok": counts.get("fail", 0) == 0,
        "summary": {status: counts.get(status, 0) for status in ("pass", "warn", "fail", "skip")},
        "checks": checks,
    }


def _render_text(report: dict[str, Any]) -> str:
    lines = [f"{item['status'].upper():4} {item['name']}: {item['detail']}" for item in report["checks"]]
    summary = report["summary"]
    lines.append(
        "Summary: "
        + ", ".join(f"{status}={summary[status]}" for status in ("pass", "warn", "fail", "skip"))
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--offline", action="store_true", help="Validate only the manifest and vendored hashes")
    parser.add_argument("--json", action="store_true", help="Print the audit report as JSON")
    parser.add_argument("--output", type=Path, help="Optionally write the same report to a local file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        manifest = load_manifest(manifest_path)
        workspace_root = manifest_path.parent.parent
        report = audit_manifest(manifest, workspace_root=workspace_root, online=not args.offline)
    except AuditError as exc:
        print(f"Asset audit could not start: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True) if args.json else _render_text(report)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
