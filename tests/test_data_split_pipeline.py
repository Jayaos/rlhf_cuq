from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from scripts.build_data_manifest import (
    DEFAULT_CONFIG,
    SplitBuildError,
    _resolve_split_data_files,
    _validate_expected_source_counts,
    allocate_grouped_records,
    annotate_source_records,
    compute_target_counts,
    write_split_bundle,
)
from src.data_utils.manifest_dataset_loader import get_manifest_dataset
from src.data_utils.split_manifest import (
    DUPLICATE_ORDINAL_FIELD,
    PROMPT_ID_FIELD,
    RECORD_ID_FIELD,
    ROLE_FIELD,
    SplitManifestError,
    load_split_records,
    verify_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _preference_rows(count: int, prefix: str = "rm") -> list[dict]:
    return [
        {
            "instruction": f"{prefix} instruction {index}",
            "input": "",
            "answers": [f"answer A {index}", f"answer B {index}"],
            "preference": index % 2,
        }
        for index in range(count)
    ]


def _prompt_rows(count: int, prefix: str = "rl") -> list[dict]:
    return [
        {
            "instruction": f"{prefix} instruction {index}",
            "input": "",
            "output": f"source output {index}",
        }
        for index in range(count)
    ]


def _annotate(rows: list[dict], kind: str, source_split: str) -> list[dict]:
    return annotate_source_records(
        rows,
        kind=kind,
        asset_name=f"synthetic_{kind}",
        repo_id=f"test/{kind}",
        revision="0123456789abcdef",
        source_split=source_split,
    )


class DataSplitPipelineTests(unittest.TestCase):
    def test_exact_duplicate_rows_keep_stable_distinct_occurrence_ids(self) -> None:
        repeated = _preference_rows(1)[0]
        rows = [dict(repeated), dict(repeated), *_preference_rows(1, prefix="unique")]

        forward = _annotate(rows, "preference", "train")
        reverse = _annotate(reversed(rows), "preference", "train")

        self.assertEqual(len({record[RECORD_ID_FIELD] for record in forward}), 3)
        self.assertEqual(
            {record[RECORD_ID_FIELD] for record in forward},
            {record[RECORD_ID_FIELD] for record in reverse},
        )
        repeated_records = [
            record for record in forward if record["instruction"] == repeated["instruction"]
        ]
        self.assertEqual(
            [record[DUPLICATE_ORDINAL_FIELD] for record in repeated_records],
            [0, 1],
        )
        self.assertEqual(
            len({record[PROMPT_ID_FIELD] for record in repeated_records}),
            1,
        )

    def test_coste_config_freezes_exact_source_files_and_counts(self) -> None:
        config_text = (ROOT / "configs/data_split_coste_v1.yaml").read_text(encoding="utf-8")
        for expected in (
            "train/human_pref.json",
            "train/sft.json",
            "train/synth_pref.json",
            "train/unlabelled.json",
            "validation/val.json",
            "alpaca_instructions/unlabeled.json",
            "alpaca_instructions/val.json",
            "train: 49383",
            "unlabeled: 20001",
            "cross_source_prompt_overlap_policy: allow_coste_native_and_report",
        ):
            self.assertIn(expected, config_text)

        builder_text = (ROOT / "scripts/build_data_manifest.py").read_text(encoding="utf-8")
        self.assertIn('load_dataset(\n        "json",', builder_text)
        self.assertNotIn("load_dataset(str(preference_path)", builder_text)
        self.assertNotIn("load_dataset(str(ppo_path)", builder_text)

    def test_primary_config_reserves_unlabeled_prompts_exclusively_for_ppo(self) -> None:
        config_path = ROOT / "configs/data_split_prompt_disjoint_v1.yaml"
        config_text = config_path.read_text(encoding="utf-8")

        self.assertEqual(DEFAULT_CONFIG, config_path)
        for expected in (
            "name: alpaca_farm_prompt_disjoint_v1",
            "cross_source_prompt_overlap_policy: forbid",
            "train/human_pref.json",
            "train/sft.json",
            "train/synth_pref.json",
            "validation/val.json",
            "rm_pool: 31382",
            "alpaca_instructions/unlabeled.json",
            "unlabeled: 20001",
            "D_rm_train: 90",
            "D_rm_val: 5",
            "D_cal: 5",
            "D_rl_train_prompts: 80",
            "D_rl_val_prompts: 10",
            "D_rl_test_prompts: 10",
        ):
            self.assertIn(expected, config_text)
        self.assertNotIn("- train/unlabelled.json", config_text)
        self.assertNotIn("alpaca_instructions/val.json", config_text)

        data_overlay = (ROOT / "configs/config_data_split.yaml").read_text(
            encoding="utf-8"
        )
        rm_overlay = (ROOT / "configs/config_rm_cluster.yaml").read_text(
            encoding="utf-8"
        )
        rl_config = (ROOT / "configs/config_rl.yaml").read_text(encoding="utf-8")
        strict_manifest = "data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json"
        self.assertIn(
            "prompt_disjoint_data_split_v1:\n"
            f"  data_split_manifest_path: {strict_manifest}",
            data_overlay,
        )
        self.assertIn(
            "rm-pythia-44m-cluster-split:\n"
            "  model_name: assets/proxy_rm_sft_base\n"
            "  output_dir: models/rm-pythia-44m-prompt-disjoint\n"
            f"  data_split_manifest_path: {strict_manifest}",
            rm_overlay,
        )
        self.assertIn(
            "output_dir: models/rm-pythia-44m-prompt-disjoint",
            rm_overlay,
        )
        for seed in range(1, 6):
            self.assertIn(
                f"models/rm-pythia-44m-prompt-disjoint_seed{seed}",
                rl_config,
            )

    def test_source_files_are_explicit_verified_and_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("train/a.json", "train/b.json", "validation/val.json"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[]\n", encoding="utf-8")

            source_asset = {
                "name": "synthetic_preference",
                "required_files": [
                    {"path": "train/a.json"},
                    {"path": "train/b.json"},
                    {"path": "validation/val.json"},
                ],
            }
            resolved, provenance = _resolve_split_data_files(
                source_asset,
                root,
                {
                    "train": ["train/a.json", "train/b.json"],
                    "validation": ["validation/val.json"],
                },
                ["train", "validation"],
            )
            self.assertEqual(provenance["train"], ["train/a.json", "train/b.json"])
            self.assertEqual(resolved["validation"], [str((root / "validation/val.json").resolve())])

            with self.assertRaisesRegex(SplitBuildError, "not in the verified source manifest"):
                _resolve_split_data_files(
                    source_asset,
                    root,
                    {"train": ["train/unverified.json"]},
                    ["train"],
                )
            with self.assertRaisesRegex(SplitBuildError, "assigned more than once"):
                _resolve_split_data_files(
                    source_asset,
                    root,
                    {
                        "train": ["train/a.json"],
                        "validation": ["train/a.json"],
                    },
                    ["train", "validation"],
                )

    def test_expected_source_counts_reject_recursive_discovery_duplicates(self) -> None:
        source_config = {"expected_source_counts": {"train": 2, "validation": 1}}
        _validate_expected_source_counts(
            source_config,
            {"train": [{}, {}], "validation": [{}]},
            asset_name="synthetic_preference",
        )
        with self.assertRaisesRegex(SplitBuildError, "row-count mismatch"):
            _validate_expected_source_counts(
                source_config,
                {"train": [{}, {}, {}, {}], "validation": [{}]},
                asset_name="synthetic_preference",
            )

    def test_frozen_source_pool_quotas(self) -> None:
        self.assertEqual(
            compute_target_counts(
                31_382,
                {"D_rm_train": 90, "D_rm_val": 5, "D_cal": 5},
            ),
            {"D_rm_train": 28_244, "D_rm_val": 1_569, "D_cal": 1_569},
        )
        self.assertEqual(
            compute_target_counts(
                49_383,
                {"D_rm_train": 90, "D_rm_val": 5, "D_cal": 5},
            ),
            {"D_rm_train": 44_445, "D_rm_val": 2_469, "D_cal": 2_469},
        )
        self.assertEqual(
            compute_target_counts(
                20_001,
                {
                    "D_rl_train_prompts": 80,
                    "D_rl_val_prompts": 10,
                    "D_rl_test_prompts": 10,
                },
            ),
            {
                "D_rl_train_prompts": 16_001,
                "D_rl_val_prompts": 2_000,
                "D_rl_test_prompts": 2_000,
            },
        )

    def test_hash_assignment_is_order_independent_and_keeps_prompt_groups_together(self) -> None:
        rows = _preference_rows(8)
        rows.extend(
            [
                {
                    "instruction": "shared prompt",
                    "input": "",
                    "answers": ["first A", "first B"],
                    "preference": 0,
                },
                {
                    "instruction": "shared prompt",
                    "input": "",
                    "answers": ["second A", "second B"],
                    "preference": 1,
                },
            ]
        )
        annotated = _annotate(rows, "preference", "train")
        allocations = {"train": 50, "validation": 50}
        first, targets = allocate_grouped_records(annotated, allocations, seed="test-seed")
        second, _ = allocate_grouped_records(reversed(annotated), allocations, seed="test-seed")

        first_roles = {
            record[RECORD_ID_FIELD]: role for role, records in first.items() for record in records
        }
        second_roles = {
            record[RECORD_ID_FIELD]: role for role, records in second.items() for record in records
        }
        self.assertEqual(targets, {"train": 5, "validation": 5})
        self.assertEqual(first_roles, second_roles)

        shared_prompt_ids = {
            record[PROMPT_ID_FIELD]
            for record in annotated
            if record["instruction"] == "shared prompt"
        }
        self.assertEqual(len(shared_prompt_ids), 1)
        shared_prompt_id = next(iter(shared_prompt_ids))
        containing_roles = {
            role
            for role, records in first.items()
            if any(record[PROMPT_ID_FIELD] == shared_prompt_id for record in records)
        }
        self.assertEqual(len(containing_roles), 1)

    def test_bundle_round_trip_and_tamper_detection(self) -> None:
        rm_records = _annotate(_preference_rows(20), "preference", "train")
        rl_records = _annotate(_prompt_rows(20), "prompt", "unlabeled")
        rm_splits, rm_targets = allocate_grouped_records(
            rm_records,
            {"D_rm_train": 90, "D_rm_val": 5, "D_cal": 5},
            seed="rm-seed",
        )
        rl_splits, rl_targets = allocate_grouped_records(
            rl_records,
            {
                "D_rl_train_prompts": 80,
                "D_rl_val_prompts": 10,
                "D_rl_test_prompts": 10,
            },
            seed="rl-seed",
        )
        splits = {**rm_splits, **rl_splits}
        kinds = {role: "preference" for role in rm_splits}
        kinds.update({role: "prompt" for role in rl_splits})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            manifest_path = write_split_bundle(
                root,
                split_records=splits,
                split_kinds=kinds,
                split_targets={**rm_targets, **rl_targets},
                provenance={"name": "synthetic"},
                overlap_audit={
                    "logical_cross_source_prompt_count": 0,
                    "all_preserved_cross_source_prompt_count": 0,
                },
            )
            counts = verify_split_manifest(manifest_path)
            self.assertEqual(counts["D_rm_train"], 18)
            self.assertEqual(counts["D_rm_val"], 1)
            self.assertEqual(counts["D_cal"], 1)
            self.assertEqual(counts["D_rl_train_prompts"], 16)
            self.assertEqual(counts["D_rl_val_prompts"], 2)
            self.assertEqual(counts["D_rl_test_prompts"], 2)

            calibration = load_split_records(
                manifest_path, "D_cal", expected_kind="preference"
            )
            self.assertEqual(len(calibration), 1)

            train_path = root / "splits" / "D_rm_train.jsonl"
            train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SplitManifestError, "data hash mismatch"):
                load_split_records(manifest_path, "D_rm_train", expected_kind="preference")

    def test_coste_native_cross_source_prompt_overlap_is_allowed_and_audited(self) -> None:
        preference_rows = _preference_rows(20, prefix="shared")
        prompt_rows = [
            {
                "instruction": row["instruction"],
                "input": row["input"],
                "output": f"source output {index}",
            }
            for index, row in enumerate(preference_rows)
        ]
        rm_records = _annotate(preference_rows, "preference", "train")
        rl_records = _annotate(prompt_rows, "prompt", "unlabeled")
        rm_splits, rm_targets = allocate_grouped_records(
            rm_records,
            {"D_rm_train": 90, "D_rm_val": 5, "D_cal": 5},
            seed="rm-overlap-seed",
        )
        rl_splits, rl_targets = allocate_grouped_records(
            rl_records,
            {
                "D_rl_train_prompts": 80,
                "D_rl_val_prompts": 10,
                "D_rl_test_prompts": 10,
            },
            seed="rl-overlap-seed",
        )
        splits = {**rm_splits, **rl_splits}
        kinds = {role: "preference" for role in rm_splits}
        kinds.update({role: "prompt" for role in rl_splits})
        prompt_ids = {
            role: {record[PROMPT_ID_FIELD] for record in records}
            for role, records in splits.items()
        }
        by_role_pair = {
            rm_role: {
                rl_role: len(prompt_ids[rm_role].intersection(prompt_ids[rl_role]))
                for rl_role in sorted(rl_splits)
            }
            for rm_role in sorted(rm_splits)
        }
        overlap_audit = {
            "policy": "allow_coste_native_and_report",
            "logical_cross_source_prompt_count": 20,
            "all_preserved_cross_source_prompt_count": 20,
            "by_role_pair": by_role_pair,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = write_split_bundle(
                root / "allowed",
                split_records=splits,
                split_kinds=kinds,
                split_targets={**rm_targets, **rl_targets},
                provenance={"name": "synthetic-overlap"},
                overlap_audit=overlap_audit,
            )
            self.assertEqual(sum(verify_split_manifest(manifest_path).values()), 40)

            forbidden_audit = dict(overlap_audit, policy="forbid")
            forbidden_manifest = write_split_bundle(
                root / "forbidden",
                split_records=splits,
                split_kinds=kinds,
                split_targets={**rm_targets, **rl_targets},
                provenance={"name": "synthetic-overlap-forbidden"},
                overlap_audit=forbidden_audit,
            )
            with self.assertRaisesRegex(SplitManifestError, "overlap is forbidden"):
                verify_split_manifest(forbidden_manifest)

    def test_trainers_connect_only_training_and_validation_roles(self) -> None:
        loader_text = (ROOT / "src/data_utils/manifest_dataset_loader.py").read_text(
            encoding="utf-8"
        )
        ppo_text = (ROOT / "src/ppo/trainer_rl.py").read_text(encoding="utf-8")
        rm_wrapper = (
            ROOT / "src/reward_modeling/training/trainer_rm_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"D_rm_train"', loader_text)
        self.assertIn('"D_rm_val"', loader_text)
        self.assertIn('"D_rl_train_prompts"', loader_text)
        self.assertIn('"D_rl_val_prompts"', loader_text)
        self.assertNotIn('load_split_records(manifest_path, "D_cal"', loader_text)
        self.assertNotIn('load_split_records(manifest_path, "D_rl_test_prompts"', loader_text)
        self.assertIn("get_manifest_dataset(training_conf, mode=\"rl\")", ppo_text)
        self.assertIn("legacy_trainer.get_dataset = get_manifest_dataset", rm_wrapper)

    def test_preserved_external_roles_are_included_in_leakage_audit(self) -> None:
        rm_records = _annotate(_preference_rows(20), "preference", "train")
        rl_records = _annotate(_prompt_rows(20), "prompt", "unlabeled")
        rm_splits, rm_targets = allocate_grouped_records(
            rm_records,
            {"D_rm_train": 90, "D_rm_val": 5, "D_cal": 5},
            seed="rm-seed",
        )
        rl_splits, rl_targets = allocate_grouped_records(
            rl_records,
            {
                "D_rl_train_prompts": 80,
                "D_rl_val_prompts": 10,
                "D_rl_test_prompts": 10,
            },
            seed="rl-seed",
        )
        leaked = dict(rm_splits["D_rm_train"][0])
        leaked[ROLE_FIELD] = "D_rm_external_val"
        splits = {**rm_splits, **rl_splits, "D_rm_external_val": [leaked]}
        kinds = {role: "preference" for role in rm_splits}
        kinds.update({role: "prompt" for role in rl_splits})
        kinds["D_rm_external_val"] = "preference"

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = write_split_bundle(
                Path(directory) / "bundle",
                split_records=splits,
                split_kinds=kinds,
                split_targets={
                    **rm_targets,
                    **rl_targets,
                    "D_rm_external_val": None,
                },
                provenance={"name": "synthetic-leak"},
                overlap_audit={
                    "logical_cross_source_prompt_count": 0,
                    "all_preserved_cross_source_prompt_count": 0,
                },
            )
            with self.assertRaisesRegex(SplitManifestError, "leakage detected"):
                verify_split_manifest(manifest_path)

    def test_manifest_adapter_satisfies_legacy_rm_and_ppo_dataset_contracts(self) -> None:
        rm_records = _annotate(_preference_rows(20), "preference", "train")
        rl_records = _annotate(_prompt_rows(20), "prompt", "unlabeled")
        rm_splits, rm_targets = allocate_grouped_records(
            rm_records,
            {"D_rm_train": 90, "D_rm_val": 5, "D_cal": 5},
            seed="rm-seed",
        )
        rl_splits, rl_targets = allocate_grouped_records(
            rl_records,
            {
                "D_rl_train_prompts": 80,
                "D_rl_val_prompts": 10,
                "D_rl_test_prompts": 10,
            },
            seed="rl-seed",
        )
        splits = {**rm_splits, **rl_splits}
        kinds = {role: "preference" for role in rm_splits}
        kinds.update({role: "prompt" for role in rl_splits})

        class FakeConcatDataset:
            def __init__(self, datasets):
                self.datasets = datasets

            def __len__(self):
                return sum(len(dataset) for dataset in self.datasets)

            def __getitem__(self, index):
                for dataset in self.datasets:
                    if index < len(dataset):
                        return dataset[index]
                    index -= len(dataset)
                raise IndexError(index)

        class FakeSubset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, index):
                return self.dataset[self.indices[index]]

        class FakePreferenceDataset:
            def __init__(self, dataset, stop, train=True):
                self.data = list(dataset)[:stop]
                self.train = train

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                return self.data[index]

        torch_module = ModuleType("torch")
        torch_utils_module = ModuleType("torch.utils")
        torch_data_module = ModuleType("torch.utils.data")
        torch_data_module.ConcatDataset = FakeConcatDataset
        torch_data_module.Subset = FakeSubset
        rank_module = ModuleType("src.data_utils.oa_custom_datasets.rank_datasets")
        rank_module.CustomHFPref = FakePreferenceDataset
        model_training_module = ModuleType("model_training")
        custom_datasets_module = ModuleType("model_training.custom_datasets")
        oa_utils_module = ModuleType("model_training.custom_datasets.utils")
        oa_utils_module._filter_by_words = lambda value: value
        fake_modules = {
            "torch": torch_module,
            "torch.utils": torch_utils_module,
            "torch.utils.data": torch_data_module,
            "src.data_utils.oa_custom_datasets.rank_datasets": rank_module,
            "model_training": model_training_module,
            "model_training.custom_datasets": custom_datasets_module,
            "model_training.custom_datasets.utils": oa_utils_module,
        }

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = write_split_bundle(
                Path(directory) / "bundle",
                split_records=splits,
                split_kinds=kinds,
                split_targets={**rm_targets, **rl_targets},
                provenance={"name": "synthetic-adapter"},
                overlap_audit={
                    "logical_cross_source_prompt_count": 0,
                    "all_preserved_cross_source_prompt_count": 0,
                },
            )
            conf = SimpleNamespace(
                data_split_manifest_path=str(manifest_path),
                eval_size=1,
                rl_dataset_path_override="",
            )
            with patch.dict("sys.modules", fake_modules):
                rm_train, rm_evals = get_manifest_dataset(conf, mode="rm")
                rl_train, rl_evals = get_manifest_dataset(conf, mode="rl")

            self.assertEqual(len(rm_train), 18)
            self.assertEqual(list(rm_evals), ["D_rm_val"])
            self.assertEqual(len(rm_evals["D_rm_val"]), 1)
            self.assertEqual(len(rl_train), 16)
            self.assertEqual(list(rl_evals), ["D_rl_val_prompts"])
            self.assertEqual(len(rl_evals["D_rl_val_prompts"]), 1)
            self.assertEqual(len(rl_train[0]), 1)


if __name__ == "__main__":
    unittest.main()
