from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.audit_assets import (
    AuditError,
    audit_dataset_contracts,
    audit_huggingface_assets,
    audit_local_baseline,
    audit_manifest,
    audit_repositories,
    load_manifest,
    validate_manifest,
)


def minimal_manifest(file_path: str, digest: str, size: int) -> dict:
    return {
        "schema_version": "1.0.0",
        "repositories": [
            {
                "name": "example_source",
                "provider": "github",
                "repository": "owner/repository",
                "revision": "a" * 40,
                "license": "MIT",
                "package_versions": {"example": "1.0.0"},
            }
        ],
        "huggingface_assets": [
            {
                "name": "example_model",
                "kind": "model",
                "repo_id": "owner/model",
                "revision": "b" * 40,
                "license": "MIT",
                "required_files": [
                    {
                        "path": "weights.bin",
                        "size": 12,
                        "git_blob_sha1": "c" * 40,
                        "lfs_sha256": "d" * 64,
                    }
                ],
            }
        ],
        "dataset_contracts": [],
        "legacy_baseline_files": [{"path": file_path, "sha256": digest, "size": size}],
    }


class ManifestValidationTests(unittest.TestCase):
    def test_load_manifest_rejects_non_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaises(AuditError):
                load_manifest(path)

    def test_validation_accepts_lowercase_immutable_revisions_and_sha256(self) -> None:
        manifest = minimal_manifest("baseline.py", "0" * 64, 1)
        failures = [check for check in validate_manifest(manifest) if check["status"] == "fail"]
        self.assertEqual(failures, [])

    def test_validation_rejects_moving_or_short_revisions(self) -> None:
        manifest = minimal_manifest("baseline.py", "0" * 64, 1)
        manifest["repositories"][0]["revision"] = "main"
        failures = [check for check in validate_manifest(manifest) if check["status"] == "fail"]
        self.assertTrue(any(check["name"].endswith("revision") for check in failures))

    def test_validation_rejects_unsupported_provider_and_malformed_nested_file(self) -> None:
        manifest = minimal_manifest("baseline.py", "0" * 64, 1)
        manifest["repositories"][0]["provider"] = "unknown"
        manifest["huggingface_assets"][0]["required_files"] = [None]
        failures = [check for check in validate_manifest(manifest) if check["status"] == "fail"]
        self.assertTrue(any(check["name"].endswith("provider") for check in failures))
        self.assertTrue(any("required_files[0]" in check["name"] for check in failures))

    def test_validation_handles_malformed_baseline_entry_without_exception(self) -> None:
        manifest = minimal_manifest("baseline.py", "0" * 64, 1)
        manifest["legacy_baseline_files"] = [None]
        failures = [check for check in validate_manifest(manifest) if check["status"] == "fail"]
        self.assertTrue(any("legacy_baseline_files[0]" in check["name"] for check in failures))

    def test_validation_rejects_unknown_baseline_canonicalization(self) -> None:
        manifest = minimal_manifest("baseline.py", "0" * 64, 1)
        manifest["legacy_baseline_files"][0]["canonicalization"] = "unknown"
        failures = [check for check in validate_manifest(manifest) if check["status"] == "fail"]
        self.assertTrue(any(check["name"].endswith("canonicalization") for check in failures))

    def test_dataset_contract_must_match_a_pinned_dataset_asset(self) -> None:
        manifest = minimal_manifest("baseline.py", "0" * 64, 1)
        asset = manifest["huggingface_assets"][0]
        asset.update(name="example_dataset", kind="dataset", repo_id="owner/dataset")
        manifest["dataset_contracts"] = [
            {
                "source_asset_name": "example_dataset",
                "repo_id": "owner/dataset",
                "revision": "b" * 40,
                "config": "default",
                "splits": {"train": 2, "validation": 1},
                "total_rows": 3,
                "features": {"text": "string"},
                "observation": {"date": "2026-08-24", "method": "recorded fixture"},
            }
        ]
        failures = [check for check in validate_manifest(manifest) if check["status"] == "fail"]
        self.assertEqual(failures, [])
        checks = audit_dataset_contracts(manifest["dataset_contracts"], manifest["huggingface_assets"])
        self.assertEqual([check["status"] for check in checks], ["pass", "warn"])

        manifest["dataset_contracts"][0]["revision"] = "e" * 40
        failures = [check for check in validate_manifest(manifest) if check["status"] == "fail"]
        self.assertTrue(any(check["name"].endswith("revision") for check in failures))

    def test_workspace_manifest_is_structurally_complete(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = load_manifest(root / "artifacts" / "source_manifest.json")
        failures = [check for check in validate_manifest(manifest) if check["status"] == "fail"]
        self.assertEqual(failures, [])
        self.assertTrue(all(asset["required_files"] for asset in manifest["huggingface_assets"]))


class LocalAuditTests(unittest.TestCase):
    def test_local_hash_passes_then_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"legacy reward bytes\n"
            target = root / "baseline.py"
            target.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            manifest = minimal_manifest("baseline.py", digest, len(payload))

            checks = audit_local_baseline(manifest, root)
            self.assertEqual([check["status"] for check in checks], ["pass"])

            target.write_bytes(payload + b"changed")
            checks = audit_local_baseline(manifest, root)
            self.assertEqual([check["status"] for check in checks], ["fail"])

    def test_utf8_lf_fingerprint_accepts_crlf_and_detects_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_payload = b"first line\nsecond line\n"
            target = root / "baseline.py"
            target.write_bytes(canonical_payload.replace(b"\n", b"\r\n"))
            manifest = minimal_manifest(
                "baseline.py",
                hashlib.sha256(canonical_payload).hexdigest(),
                len(canonical_payload),
            )
            manifest["legacy_baseline_files"][0]["canonicalization"] = "utf8_lf"

            checks = audit_local_baseline(manifest, root)
            self.assertEqual([check["status"] for check in checks], ["pass"])

            target.write_bytes(b"first line\r\nchanged line\r\n")
            checks = audit_local_baseline(manifest, root)
            self.assertEqual([check["status"] for check in checks], ["fail"])

    def test_local_path_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = minimal_manifest("../outside.py", "0" * 64, 0)
            checks = audit_local_baseline(manifest, root)
            self.assertEqual(checks[0]["status"], "fail")
            self.assertIn("escapes", checks[0]["detail"])


class RemoteMetadataTests(unittest.TestCase):
    def test_repository_revision_must_resolve_exactly(self) -> None:
        source = minimal_manifest("x", "0" * 64, 0)["repositories"]
        passing = audit_repositories(source, lambda _url: {"sha": "a" * 40})
        failing = audit_repositories(source, lambda _url: {"sha": "f" * 40})
        self.assertEqual(passing[0]["status"], "pass")
        self.assertEqual(failing[0]["status"], "fail")

    def test_huggingface_revision_and_required_file_metadata(self) -> None:
        asset = minimal_manifest("x", "0" * 64, 0)["huggingface_assets"]
        metadata = {
            "sha": "b" * 40,
            "siblings": [
                {
                    "rfilename": "weights.bin",
                    "size": 12,
                    "blobId": "c" * 40,
                    "lfs": {"sha256": "d" * 64},
                }
            ],
        }
        passing = audit_huggingface_assets(asset, lambda _url: metadata)
        self.assertEqual([check["status"] for check in passing], ["pass", "pass"])

        metadata["siblings"][0]["size"] = 13
        failing = audit_huggingface_assets(asset, lambda _url: metadata)
        self.assertEqual(failing[-1]["status"], "fail")

    def test_offline_full_report_never_calls_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"stable"
            (root / "baseline.py").write_bytes(payload)
            manifest = minimal_manifest("baseline.py", hashlib.sha256(payload).hexdigest(), len(payload))

            def forbidden_client(_url: str) -> dict:
                self.fail("offline audit attempted a network request")

            report = audit_manifest(manifest, workspace_root=root, online=False, client=forbidden_client)
            self.assertTrue(report["ok"])
            self.assertEqual(report["mode"], "offline")
            self.assertEqual(report["summary"]["skip"], 1)


if __name__ == "__main__":
    unittest.main()
