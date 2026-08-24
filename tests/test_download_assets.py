from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.download_assets import (
    DEFAULT_ASSET_NAMES,
    GOLD_ASSET_NAMES,
    DownloadError,
    select_assets,
    verify_asset,
)


def asset(name: str) -> dict:
    return {"name": name, "required_files": []}


class AssetSelectionTests(unittest.TestCase):
    def test_default_selection_excludes_large_gold_weight_differences(self) -> None:
        available = [asset(name) for name in (*DEFAULT_ASSET_NAMES, *GOLD_ASSET_NAMES)]
        selected = select_assets(available, requested_names=None, include_gold=False)
        self.assertEqual([entry["name"] for entry in selected], list(DEFAULT_ASSET_NAMES))

    def test_gold_selection_is_explicit_and_deduplicated(self) -> None:
        available = [asset(name) for name in (*DEFAULT_ASSET_NAMES, *GOLD_ASSET_NAMES)]
        selected = select_assets(
            available,
            requested_names=[DEFAULT_ASSET_NAMES[0], GOLD_ASSET_NAMES[0]],
            include_gold=True,
        )
        self.assertEqual(
            [entry["name"] for entry in selected],
            [DEFAULT_ASSET_NAMES[0], *GOLD_ASSET_NAMES],
        )

    def test_unknown_selection_fails(self) -> None:
        with self.assertRaisesRegex(DownloadError, "Unknown asset"):
            select_assets([], requested_names=["missing"], include_gold=False)


class LocalPayloadVerificationTests(unittest.TestCase):
    def test_regular_git_blob_and_lfs_payloads_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regular_payload = b"manifest file\n"
            lfs_payload = b"large model payload"
            (root / "config.json").write_bytes(regular_payload)
            (root / "weights.bin").write_bytes(lfs_payload)

            regular_blob = hashlib.sha1(
                f"blob {len(regular_payload)}\0".encode("ascii") + regular_payload
            ).hexdigest()
            local_asset = {
                "required_files": [
                    {
                        "path": "config.json",
                        "size": len(regular_payload),
                        "git_blob_sha1": regular_blob,
                    },
                    {
                        "path": "weights.bin",
                        "size": len(lfs_payload),
                        "git_blob_sha1": "unused-for-lfs",
                        "lfs_sha256": hashlib.sha256(lfs_payload).hexdigest(),
                    },
                ]
            }
            self.assertEqual(verify_asset(local_asset, root), [])

            (root / "weights.bin").write_bytes(b"corrupt model bytes")
            failures = verify_asset(local_asset, root)
            self.assertTrue(any("sha256" in failure for failure in failures))

    def test_missing_required_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local_asset = {
                "required_files": [
                    {"path": "missing.bin", "size": 1, "git_blob_sha1": "0" * 40}
                ]
            }
            self.assertIn("missing:", verify_asset(local_asset, Path(directory))[0])


if __name__ == "__main__":
    unittest.main()
