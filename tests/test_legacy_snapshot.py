from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "source_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LegacySnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.entries = cls.manifest["legacy_baseline_files"]

    def test_all_snapshot_files_match_pinned_coste_bytes(self) -> None:
        mismatches = []
        for entry in self.entries:
            path = ROOT / entry["path"]
            actual = {
                "exists": path.is_file(),
                "size": path.stat().st_size if path.is_file() else None,
                "sha256": sha256(path) if path.is_file() else None,
            }
            expected = {"exists": True, "size": entry["size"], "sha256": entry["sha256"]}
            if actual != expected:
                mismatches.append({"path": entry["path"], "expected": expected, "actual": actual})
        self.assertEqual(mismatches, [], json.dumps(mismatches, indent=2))

    def test_reward_and_ppo_behavior_files_are_covered(self) -> None:
        protected = {entry["path"] for entry in self.entries}
        required = {
            "src/reward_modeling/scoring/score.py",
            "src/reward_modeling/scoring/ppo_reward_functions.py",
            "src/ppo/custom_helpers.py",
            "src/ppo/custom_trlx_trainers/custom_accelerate_base_trainer.py",
            "src/ppo/custom_trlx_trainers/custom_accelerate_ppo_trainer.py",
            "src/data_utils/oa_custom_datasets/dataset_loader.py",
            "src/data_utils/oa_custom_datasets/rank_datasets.py",
            "src/data_utils/oa_custom_datasets/get_dataset_patch.py",
            "src/data_utils/rm_dataset_formatter.py",
            "src/reward_modeling/training/trainer_rm.py",
            "configs/config_rm.yaml",
            "configs/ppo_config.yaml",
        }
        self.assertEqual(required - protected, set())

    def test_manifest_records_exact_coste_revision(self) -> None:
        repositories = {entry["name"]: entry for entry in self.manifest["repositories"]}
        self.assertEqual(
            repositories["coste_llm_optimization"]["revision"],
            "416b03cc2c3c8125208679acd88891584d9eefd2",
        )


if __name__ == "__main__":
    unittest.main()
