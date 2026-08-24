#!/usr/bin/env python3
"""Download revision-pinned Hugging Face assets and verify local payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "source_manifest.json"
DEFAULT_ASSET_ROOT = ROOT / "assets"
DEFAULT_ASSET_NAMES = (
    "initial_sft_policy",
    "proxy_rm_sft_base",
    "coste_preference_dataset",
    "alpaca_farm_prompt_dataset",
)
GOLD_ASSET_NAMES = (
    "alpaca_farm_sft10k_weight_diff",
    "alpaca_farm_gold_rm_weight_diff",
)


class DownloadError(RuntimeError):
    """Raised when the manifest, selection, download, or checksum is invalid."""


def load_assets(manifest_path: Path) -> list[dict]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DownloadError(f"Manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise DownloadError(f"Manifest is not valid JSON: {manifest_path}: {exc}") from exc

    assets = manifest.get("huggingface_assets")
    if not isinstance(assets, list) or not assets:
        raise DownloadError("Manifest has no Hugging Face assets")
    return assets


def select_assets(
    assets: Iterable[dict], requested_names: Iterable[str] | None, include_gold: bool
) -> list[dict]:
    assets_by_name = {asset.get("name"): asset for asset in assets}
    names = list(requested_names) if requested_names else list(DEFAULT_ASSET_NAMES)
    if include_gold:
        names.extend(GOLD_ASSET_NAMES)

    selected_names = list(dict.fromkeys(names))
    unknown = [name for name in selected_names if name not in assets_by_name]
    if unknown:
        raise DownloadError(f"Unknown asset name(s): {', '.join(unknown)}")
    return [assets_by_name[name] for name in selected_names]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 - Git object identity, not security
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(asset: dict, asset_directory: Path) -> list[str]:
    errors: list[str] = []
    for required_file in asset.get("required_files", []):
        relative_path = Path(required_file["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe manifest path: {relative_path}")
            continue

        local_path = asset_directory / relative_path
        if not local_path.is_file():
            errors.append(f"missing: {local_path}")
            continue

        actual_size = local_path.stat().st_size
        expected_size = required_file["size"]
        if actual_size != expected_size:
            errors.append(f"size: {local_path}: expected {expected_size}, found {actual_size}")
            continue

        expected_lfs_hash = required_file.get("lfs_sha256")
        if expected_lfs_hash:
            actual_hash = sha256_file(local_path)
            if actual_hash != expected_lfs_hash:
                errors.append(
                    f"sha256: {local_path}: expected {expected_lfs_hash}, found {actual_hash}"
                )
        else:
            expected_blob = required_file["git_blob_sha1"]
            actual_blob = git_blob_sha1(local_path)
            if actual_blob != expected_blob:
                errors.append(
                    f"git-blob-sha1: {local_path}: expected {expected_blob}, found {actual_blob}"
                )
    return errors


def expected_payload_size(asset: dict) -> int:
    return sum(required_file["size"] for required_file in asset.get("required_files", []))


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument(
        "--asset",
        action="append",
        dest="assets",
        help="Manifest asset name to fetch; repeat to select multiple. Defaults to online PPO assets.",
    )
    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Also fetch both large AlpacaFarm weight-difference snapshots.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not use the network; verify files already under --asset-root.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        assets = select_assets(load_assets(args.manifest), args.assets, args.include_gold)
        asset_root = args.asset_root.expanduser().resolve()
        if args.max_workers < 1:
            raise DownloadError("--max-workers must be at least 1")

        total_size = sum(expected_payload_size(asset) for asset in assets)
        mode = "Verifying" if args.verify_only else "Downloading"
        print(f"{mode} {len(assets)} asset(s); required payload is {format_bytes(total_size)}")

        snapshot_download = None
        if not args.verify_only:
            try:
                from huggingface_hub import snapshot_download as hub_snapshot_download
            except ImportError as exc:
                raise DownloadError(
                    "huggingface_hub is not installed; finish the cluster environment setup first"
                ) from exc
            snapshot_download = hub_snapshot_download
            asset_root.mkdir(parents=True, exist_ok=True)

        failed = False
        for asset in assets:
            name = asset["name"]
            if Path(name).name != name:
                raise DownloadError(f"Unsafe asset name in manifest: {name!r}")
            destination = asset_root / name
            print(f"[{name}] {asset['repo_id']}@{asset['revision']}")
            if snapshot_download is not None:
                snapshot_download(
                    repo_id=asset["repo_id"],
                    repo_type="dataset" if asset["kind"] == "dataset" else None,
                    revision=asset["revision"],
                    local_dir=str(destination),
                    local_dir_use_symlinks=False,
                    force_download=args.force_download,
                    max_workers=args.max_workers,
                )

            errors = verify_asset(asset, destination)
            if errors:
                failed = True
                for error in errors:
                    print(f"  FAIL {error}", file=sys.stderr)
            else:
                print(f"  PASS {len(asset['required_files'])} required file(s)")

        return 1 if failed else 0
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
