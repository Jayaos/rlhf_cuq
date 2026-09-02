from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from src.data_utils.split_manifest import RECORD_ID_FIELD
from src.data_utils import manifest_dataset_loader
from src.reward_modeling.training.label_noise import (
    FLIPPED_IDS_FILENAME,
    METADATA_FILENAME,
    apply_preference_label_noise,
    load_optional_label_noise_metadata,
    persist_label_noise_provenance,
    validate_label_noise_config,
    validate_persisted_label_noise_provenance,
)


MANIFEST_SHA256 = "ab" * 32


def rows(count: int) -> list[dict]:
    return [
        {
            RECORD_ID_FIELD: f"preference_{index:04d}",
            "instruction": f"instruction {index}",
            "input": "",
            "answers": [f"answer a {index}", f"answer b {index}"],
            "preference": index % 2,
        }
        for index in range(count)
    ]


class RMLabelNoiseTests(unittest.TestCase):
    def test_thirty_percent_flips_exactly_without_mutating_inputs(self) -> None:
        original = rows(10)
        before = [dict(row) for row in original]

        result = apply_preference_label_noise(
            original,
            rate=0.30,
            seed=7,
            manifest_sha256=MANIFEST_SHA256,
        )

        self.assertEqual(original, before)
        self.assertEqual(result.metadata["flip_count"], 3)
        self.assertEqual(result.metadata["realized_rate"], 0.3)
        self.assertEqual(len(result.flipped_record_ids), 3)
        selected = set(result.flipped_record_ids)
        for clean, noisy in zip(original, result.rows):
            self.assertEqual(
                noisy["preference"],
                1 - clean["preference"]
                if clean[RECORD_ID_FIELD] in selected
                else clean["preference"],
            )
            self.assertEqual(
                {key: value for key, value in noisy.items() if key != "preference"},
                {key: value for key, value in clean.items() if key != "preference"},
            )

    def test_selection_is_order_independent_and_seed_dependent(self) -> None:
        forward = apply_preference_label_noise(
            rows(40), rate=0.30, seed=11, manifest_sha256=MANIFEST_SHA256
        )
        reverse = apply_preference_label_noise(
            list(reversed(rows(40))), rate=0.30, seed=11, manifest_sha256=MANIFEST_SHA256
        )
        different_seed = apply_preference_label_noise(
            rows(40), rate=0.30, seed=12, manifest_sha256=MANIFEST_SHA256
        )

        self.assertEqual(forward.flipped_record_ids, reverse.flipped_record_ids)
        self.assertNotEqual(forward.flipped_record_ids, different_seed.flipped_record_ids)

    def test_zero_rate_is_an_identity_copy(self) -> None:
        original = rows(5)
        result = apply_preference_label_noise(
            original, rate=0.0, seed=0, manifest_sha256=MANIFEST_SHA256
        )
        self.assertFalse(result.metadata["enabled"])
        self.assertEqual(result.metadata["flip_count"], 0)
        self.assertEqual(result.rows, original)
        self.assertIsNot(result.rows[0], original[0])

    def test_invalid_configuration_and_rows_are_rejected(self) -> None:
        for rate in (-0.1, 1.1, float("nan"), True, "0.3"):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                validate_label_noise_config(rate, 1)
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_label_noise_config(0.3, seed)

        invalid = rows(2)
        invalid[0]["preference"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            apply_preference_label_noise(
                invalid, rate=0.3, seed=1, manifest_sha256=MANIFEST_SHA256
            )

    def test_persisted_provenance_round_trips_and_rejects_mismatch(self) -> None:
        result = apply_preference_label_noise(
            rows(10), rate=0.3, seed=19, manifest_sha256=MANIFEST_SHA256
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            persist_label_noise_provenance(root, result.metadata, result.flipped_record_ids)
            (root / "config.json").write_text(
                json.dumps({"rm_label_noise": result.metadata}), encoding="utf-8"
            )

            validated = validate_persisted_label_noise_provenance(
                root, expected_rate=0.3, expected_seed=19
            )
            self.assertEqual(validated, result.metadata)
            self.assertEqual(load_optional_label_noise_metadata(root), result.metadata)
            self.assertTrue((root / METADATA_FILENAME).is_file())
            self.assertEqual(
                len((root / FLIPPED_IDS_FILENAME).read_text(encoding="utf-8").splitlines()),
                3,
            )

            with self.assertRaisesRegex(RuntimeError, "rate mismatch"):
                validate_persisted_label_noise_provenance(root, expected_rate=0.2)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                persist_label_noise_provenance(
                    root,
                    {**result.metadata, "seed": 20},
                    result.flipped_record_ids,
                )

    def test_optional_metadata_distinguishes_clean_and_partial_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            self.assertIsNone(load_optional_label_noise_metadata(root))

            (root / METADATA_FILENAME).write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_optional_label_noise_metadata(root)

    def test_downstream_artifacts_record_proxy_noise_provenance(self) -> None:
        for relative_path in (
            "scripts/prepare_cpdpo_artifacts.py",
            "scripts/prepare_advpo_confidence.py",
            "src/ppo/trainer_reward_overoptimization.py",
        ):
            with self.subTest(path=relative_path):
                source = (Path(__file__).parents[1] / relative_path).read_text(
                    encoding="utf-8"
                )
                self.assertIn('"proxy_rm_label_noise"', source)
                self.assertIn("load_optional_label_noise_metadata", source)

    def test_manifest_adapter_corrupts_training_but_not_validation(self) -> None:
        training_rows = rows(10)
        validation_rows = rows(4)

        class FakeConcatDataset:
            def __init__(self, datasets):
                self.datasets = datasets

        class FakePreferenceDataset:
            def __init__(self, dataset, stop, train=True):
                self.data = list(dataset)[:stop]
                self.train = train

            def __len__(self):
                return len(self.data)

        torch_module = ModuleType("torch")
        torch_utils_module = ModuleType("torch.utils")
        torch_data_module = ModuleType("torch.utils.data")
        torch_data_module.ConcatDataset = FakeConcatDataset
        rank_module = ModuleType("src.data_utils.oa_custom_datasets.rank_datasets")
        rank_module.CustomHFPref = FakePreferenceDataset
        fake_modules = {
            "torch": torch_module,
            "torch.utils": torch_utils_module,
            "torch.utils.data": torch_data_module,
            "src.data_utils.oa_custom_datasets.rank_datasets": rank_module,
        }

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")

            def load_role(_path, role, expected_kind):
                self.assertEqual(expected_kind, "preference")
                return training_rows if role == "D_rm_train" else validation_rows

            conf = SimpleNamespace(
                rm_label_noise_rate=0.30,
                rm_label_noise_seed=5,
                eval_size=None,
            )
            with patch.dict("sys.modules", fake_modules), patch.object(
                manifest_dataset_loader, "load_split_records", side_effect=load_role
            ):
                train, evaluations = manifest_dataset_loader._preference_dataset(
                    conf, manifest_path
                )

        noisy = train.datasets[0]
        self.assertEqual(noisy.rm_label_noise_metadata["flip_count"], 3)
        changed = sum(
            clean["preference"] != corrupted["preference"]
            for clean, corrupted in zip(training_rows, noisy.data)
        )
        self.assertEqual(changed, 3)
        self.assertEqual(evaluations["D_rm_val"].data, validation_rows)
        self.assertFalse(hasattr(evaluations["D_rm_val"], "rm_label_noise_metadata"))


if __name__ == "__main__":
    unittest.main()
