from __future__ import annotations

import ast
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from src.ppo.runtime_config import (
    DEFAULT_PPO_CONFIG_PATH,
    apply_local_asset_overrides,
    resolve_ppo_config_path,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeConfigTests(unittest.TestCase):
    def test_default_selects_unchanged_coste_config(self) -> None:
        selected = resolve_ppo_config_path(None, working_directory=ROOT)
        self.assertEqual(selected, (ROOT / DEFAULT_PPO_CONFIG_PATH).resolve())

    def test_explicit_smoke_config_is_selectable(self) -> None:
        selected = resolve_ppo_config_path(
            "configs/ppo_config_smoke.yaml",
            working_directory=ROOT,
        )
        self.assertEqual(selected, (ROOT / "configs" / "ppo_config_smoke.yaml").resolve())

    def test_missing_config_fails_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "PPO configuration does not exist"):
                resolve_ppo_config_path("missing.yaml", working_directory=Path(directory))

    def test_local_asset_overrides_support_offline_single_rm_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy"
            proxy = root / "proxy"
            dataset = root / "alpaca_farm"
            for path in (policy, proxy, dataset):
                path.mkdir()
            training = SimpleNamespace(
                policy_model_path_override=str(policy),
                proxy_rm_path_override=str(proxy),
                rl_dataset_path_override=str(dataset),
                datasets=["alpaca_farm"],
                datasets_extra=[],
            )
            rank = SimpleNamespace(model_names=["moving/proxy"])
            sft = SimpleNamespace(model_name="moving/policy")
            apply_local_asset_overrides(training, rank, sft)
            self.assertEqual(sft.model_name, str(policy.resolve()))
            self.assertEqual(rank.model_names, [str(proxy.resolve())])
            self.assertEqual(
                training.datasets,
                [{"alpaca_farm": {"dataset_path": str(dataset.resolve())}}],
            )

    def test_local_proxy_override_rejects_ensemble(self) -> None:
        training = SimpleNamespace(
            policy_model_path_override="",
            proxy_rm_path_override="/checkpoints/proxy",
            rl_dataset_path_override="",
            datasets=["alpaca_farm"],
            datasets_extra=[],
        )
        with self.assertRaisesRegex(ValueError, "single-RM"):
            apply_local_asset_overrides(
                training,
                SimpleNamespace(model_names=["one", "two"]),
                SimpleNamespace(model_name="policy"),
            )

    def test_smoke_profile_is_one_update_and_opt_in(self) -> None:
        rl_text = (ROOT / "configs" / "config_rl.yaml").read_text(encoding="utf-8")
        smoke_text = (ROOT / "configs" / "ppo_config_smoke.yaml").read_text(encoding="utf-8")
        smoke_overlay = rl_text.split("baseline_smoke:", 1)[1].split("pythia_rlhf_individual:", 1)[0]
        self.assertIn("ppo_config_path: configs/ppo_config.yaml", rl_text)
        self.assertIn("run_gold_evaluation: true", rl_text)
        self.assertIn("baseline_smoke:", rl_text)
        self.assertIn("ppo_config_path: configs/ppo_config_smoke.yaml", smoke_overlay)
        self.assertIn("run_gold_evaluation: false", smoke_overlay)
        for setting in ("total_steps: 1", "num_rollouts: 2", "chunk_size: 2", "ppo_epochs: 1"):
            self.assertIn(setting, smoke_text)
        self.assertIn("batch_size: 2", smoke_text)
        self.assertIn("debug: false", smoke_overlay)

    def test_legacy_sources_are_editable_and_match_manifest_commits(self) -> None:
        source_requirements = (ROOT / "requirements" / "legacy-sources.txt").read_text(encoding="utf-8")
        requirement_lines = [
            line for line in source_requirements.splitlines() if line and not line.startswith("#")
        ]
        manifest = json.loads((ROOT / "artifacts" / "source_manifest.json").read_text(encoding="utf-8"))
        expected_revisions = {
            entry["revision"]
            for entry in manifest["repositories"]
            if entry["name"] != "coste_llm_optimization"
        }
        self.assertEqual(len(requirement_lines), 4)
        self.assertTrue(all(line.startswith("-e git+https://") for line in requirement_lines))
        for revision in expected_revisions:
            self.assertTrue(any(f"@{revision}#" in line for line in requirement_lines), revision)

        project_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("git+https://", project_text)
        runtime_text = (ROOT / "requirements" / "legacy-runtime.txt").read_text(encoding="utf-8")
        self.assertIn("typer==0.9.0", runtime_text)

    def test_cluster_install_covers_observed_native_build_prerequisites(self) -> None:
        build_text = (ROOT / "requirements" / "legacy-build.txt").read_text(
            encoding="utf-8"
        )
        conda_constraints = (
            ROOT / "requirements" / "legacy-conda.constraints.txt"
        ).read_text(encoding="utf-8")
        wheel_constraints = (
            ROOT / "requirements" / "legacy-cu118.constraints.txt"
        ).read_text(encoding="utf-8")
        environment_text = (ROOT / "environment.cluster.yml").read_text(encoding="utf-8")
        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        storage_helper = (
            ROOT / "scripts" / "configure_cluster_storage.sh"
        ).read_text(encoding="utf-8")

        required_build_pins = (
            "cmake==3.25.0",
            "lit==15.0.7",
            "pybind11==2.11.1",
        )
        for pin in required_build_pins:
            self.assertIn(pin, build_text)
            self.assertIn(pin, conda_constraints)
            self.assertIn(pin, wheel_constraints)
        self.assertIn("pybind11=2.11.1", environment_text)
        self.assertIn("pytest=7.4.0", environment_text)
        runtime_text = (ROOT / "requirements" / "legacy-runtime.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("pathtools==0.1.2", runtime_text)
        self.assertIn("threadpoolctl>=3.1.0", runtime_text)
        for constraints in (conda_constraints, wheel_constraints):
            self.assertIn("pathtools==0.1.2", constraints)
            self.assertIn("pytest==7.4.0", constraints)
        self.assertLess(
            readme_text.index("--requirement requirements/legacy-build.txt"),
            readme_text.index("--requirement requirements/legacy-runtime.txt"),
        )
        self.assertIn("source scripts/configure_cluster_storage.sh", readme_text)
        for variable in (
            "PIP_CACHE_DIR",
            "TMPDIR",
            "XDG_CACHE_HOME",
            "TORCH_EXTENSIONS_DIR",
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "CONDA_PKGS_DIRS",
            "PYTHONNOUSERSITE",
        ):
            self.assertIn(f"export {variable}=", storage_helper)
        self.assertIn("unset PYTHONHOME", storage_helper)
        self.assertIn("unset PYTHONPATH", storage_helper)
        self.assertIn('"$1" == "$HOME"', storage_helper)
        self.assertIn("python -m pytest -q tests", readme_text)

    def test_gold_call_is_guarded_by_runtime_flag(self) -> None:
        tree = ast.parse((ROOT / "src" / "ppo" / "trainer_rl.py").read_text(encoding="utf-8"))
        all_gold_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "gold_score"
        ]
        guarded_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not (
                isinstance(node.test, ast.Attribute)
                and node.test.attr == "run_gold_evaluation"
                and isinstance(node.test.value, ast.Name)
                and node.test.value.id == "training_conf"
            ):
                continue
            guarded_calls.extend(
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "gold_score"
            )
        self.assertEqual(len(all_gold_calls), 1)
        self.assertEqual(guarded_calls, all_gold_calls)


if __name__ == "__main__":
    unittest.main()
