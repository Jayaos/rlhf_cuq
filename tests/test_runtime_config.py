from __future__ import annotations

import ast
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.ppo.runtime_config import (
    DEFAULT_PPO_CONFIG_PATH,
    apply_local_asset_overrides,
    resolve_ppo_config_path,
)
from src.ppo.policy_variants import (
    DEFAULT_POLICY_VARIANT,
    get_policy_variant,
    resolve_policy_num_layers_unfrozen,
    validate_policy_checkpoint,
)
from src.reward_modeling.training.local_model_compat import apply_local_model_family
from src.cpdpo.optimizer import validate_training_precision


ROOT = Path(__file__).resolve().parents[1]


class RuntimeConfigTests(unittest.TestCase):
    @staticmethod
    def _write_policy_config(path: Path, *, hidden_size: int, layers: int, reward: bool = False) -> None:
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["GPTNeoXRewardModel" if reward else "GPTNeoXForCausalLM"],
                    "model_type": "gpt_neox_reward_model" if reward else "gpt_neox",
                    "hidden_size": hidden_size,
                    "num_hidden_layers": layers,
                    "vocab_size": 50304,
                }
            ),
            encoding="utf-8",
        )

    def test_named_policy_variants_validate_full_causal_lms(self) -> None:
        self.assertEqual(DEFAULT_POLICY_VARIANT, "1p4b")
        self.assertEqual(get_policy_variant("1p4b").default_path, "assets/initial_sft_policy")
        self.assertEqual(get_policy_variant("70m").default_path, "assets/proxy_rm_sft_base")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = root / "small"
            large = root / "large"
            self._write_policy_config(small, hidden_size=512, layers=6)
            self._write_policy_config(large, hidden_size=2048, layers=24)
            self.assertEqual(validate_policy_checkpoint(small, "70m")["variant"], "70m")
            self.assertEqual(validate_policy_checkpoint(large, "1p4b")["variant"], "1p4b")
            with self.assertRaisesRegex(ValueError, "expects hidden_size=2048"):
                validate_policy_checkpoint(small, "1p4b")

    def test_named_policy_variants_reject_scalar_reward_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reward = Path(directory) / "reward"
            self._write_policy_config(reward, hidden_size=512, layers=6, reward=True)
            with self.assertRaisesRegex(ValueError, "scalar reward-model checkpoint"):
                validate_policy_checkpoint(reward, "70m")

    def test_evaluator_resolves_recorded_hydra_wrapper_depth(self) -> None:
        architecture = {"num_hidden_layers": 6}
        metadata = {
            "policy_optimization": {"num_layers_unfrozen": 1},
            "trlx_config": {"model": {"num_layers_unfrozen": 1}},
        }
        self.assertEqual(resolve_policy_num_layers_unfrozen(metadata, architecture), 1)
        self.assertEqual(
            resolve_policy_num_layers_unfrozen(
                {"trlx_config": {"model": {"num_layers_unfrozen": 1}}}, architecture
            ),
            1,
        )
        self.assertEqual(resolve_policy_num_layers_unfrozen({}, architecture), 2)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            resolve_policy_num_layers_unfrozen(
                {
                    "policy_optimization": {"num_layers_unfrozen": 1},
                    "trlx_config": {"model": {"num_layers_unfrozen": 2}},
                },
                architecture,
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            resolve_policy_num_layers_unfrozen(
                {"policy_optimization": {"num_layers_unfrozen": 7}}, architecture
            )

    def test_experiment_jobs_thread_named_policy_variant(self) -> None:
        helper = (ROOT / "scripts/configure_policy_variant.sh").read_text(encoding="utf-8")
        self.assertIn('POLICY_VARIANT="${RLHF_POLICY_VARIANT:-1p4b}"', helper)
        self.assertIn('POLICY_DEFAULT_PATH="assets/proxy_rm_sft_base"', helper)
        self.assertIn("RLHF_POLICY_VARIANT must be exactly", helper)
        self.assertIn("RLHF_POLICY_LEARNING_RATE", helper)
        self.assertIn("RLHF_POLICY_NUM_LAYERS_UNFROZEN", helper)
        self.assertIn("RLHF_POLICY_MAX_GRAD_NORM", helper)
        self.assertIn("RLHF_POLICY_PRECISION", helper)
        self.assertIn("POLICY_ACCELERATE_CONFIG", helper)

        jobs = (
            "train_reward_overoptimization.sbatch",
            "evaluate_reward_overoptimization.sbatch",
            "smoke_reward_overoptimization.sbatch",
            "prepare_cpdpo_v2_references.sbatch",
            "train_cpdpo_v2.sbatch",
            "evaluate_cpdpo_v2.sbatch",
            "evaluate_additive_reward_overoptimization.sbatch",
            "smoke_cpdpo_v2.sbatch",
            "prepare_advpo_references.sbatch",
            "train_advpo.sbatch",
            "evaluate_advpo.sbatch",
            "smoke_advpo.sbatch",
        )
        for filename in jobs:
            with self.subTest(filename=filename):
                text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
                self.assertIn("source scripts/configure_policy_variant.sh", text)
                self.assertIn("scripts/validate_policy_variant.py", text)
                self.assertIn('--policy-variant "$POLICY_VARIANT"', text)

        for filename in (
            "train_reward_overoptimization.sbatch",
            "smoke_reward_overoptimization.sbatch",
            "train_cpdpo_v2.sbatch",
            "smoke_cpdpo_v2.sbatch",
            "train_advpo.sbatch",
            "smoke_advpo.sbatch",
        ):
            text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
            self.assertIn('"${POLICY_TRAINING_ARGS[@]}"', text)
            self.assertIn('"$POLICY_ACCELERATE_CONFIG"', text)

        trainer = (ROOT / "src/ppo/trainer_reward_overoptimization.py").read_text(encoding="utf-8")
        self.assertIn("--optimizer-learning-rate", trainer)
        self.assertIn("--num-layers-unfrozen", trainer)
        self.assertIn("--training-precision", trainer)
        self.assertIn('"policy_optimization"', trainer)

        fp32 = (ROOT / "configs/accelerate_config_fp32.yaml").read_text(encoding="utf-8")
        self.assertIn('mixed_precision: "no"', fp32)

    def test_declared_training_precision_must_match_accelerate(self) -> None:
        trainer = SimpleNamespace(accelerator=SimpleNamespace(mixed_precision="no"))
        validate_training_precision(trainer, "fp32")
        trainer.accelerator.mixed_precision = "bf16"
        validate_training_precision(trainer, "bf16")
        with self.assertRaisesRegex(RuntimeError, "precision mismatch"):
            validate_training_precision(trainer, "fp32")

        for filename in (
            "train_reward_overoptimization.sbatch",
            "evaluate_reward_overoptimization.sbatch",
            "train_cpdpo_v2.sbatch",
            "evaluate_cpdpo_v2.sbatch",
            "train_advpo.sbatch",
            "evaluate_advpo.sbatch",
            "evaluate_additive_reward_overoptimization.sbatch",
        ):
            text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
            self.assertIn("reward_overoptimization_policy_70m", text)

        for filename in (
            "prepare_cpdpo_v2_references.sbatch",
            "train_cpdpo_v2.sbatch",
            "prepare_advpo_references.sbatch",
            "train_advpo.sbatch",
        ):
            text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
            self.assertIn("policy_70m", text)
            self.assertIn("${POLICY_PRECISION}", text)

        for filename in (
            "prepare_cpdpo_v2_references.sbatch",
            "prepare_advpo_references.sbatch",
        ):
            text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
            self.assertIn('--dtype "$POLICY_PRECISION"', text)

        for filename in ("smoke_cpdpo_v2.sbatch", "smoke_advpo.sbatch"):
            text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
            reference_stage, training_stage = text.split(
                'srun accelerate launch --config_file "$POLICY_ACCELERATE_CONFIG"', 1
            )
            self.assertIn('--dtype "$POLICY_PRECISION"', reference_stage)
            self.assertNotIn('"${POLICY_TRAINING_ARGS[@]}"', reference_stage)
            self.assertIn('"${POLICY_TRAINING_ARGS[@]}"', training_stage)

    def test_additive_evaluation_array_maps_methods_and_seeds(self) -> None:
        job = (
            ROOT / "scripts/slurm/evaluate_additive_reward_overoptimization.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-5", job)
        self.assertIn("METHODS=(cpdpo_v2 advpo)", job)
        self.assertIn("TASK_ID % 2", job)
        self.assertIn("TASK_ID / 2 + 1", job)
        self.assertIn('ADVPO_B="${ADVPO_B:-}"', job)
        self.assertIn("method_run_name(\"cpdpo_v2\"", job)
        self.assertIn("advpo_run_name", job)
        self.assertIn("scripts/evaluate_policy_checkpoints.py", job)
        self.assertIn("PASS additive offline evaluation", job)

    def test_offline_evaluation_jobs_use_accounting_based_resources(self) -> None:
        for filename in (
            "evaluate_reward_overoptimization.sbatch",
            "evaluate_cpdpo_v2.sbatch",
            "evaluate_advpo.sbatch",
            "evaluate_additive_reward_overoptimization.sbatch",
        ):
            with self.subTest(filename=filename):
                text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
                self.assertIn("#SBATCH --cpus-per-task=4", text)
                self.assertIn("#SBATCH --mem=24G", text)
                self.assertIn("#SBATCH --time=03:00:00", text)
                self.assertIn('OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"', text)
                self.assertNotIn("#SBATCH --mem=128G", text)
                self.assertNotIn("#SBATCH --time=24:00:00", text)

    def test_local_rm_family_hint_preserves_path_and_exposes_pythia(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "proxy_rm_sft_base"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["GPTNeoXForCausalLM"],
                        "model_type": "gpt_neox",
                    }
                ),
                encoding="utf-8",
            )
            conf = SimpleNamespace(
                model_name="proxy_rm_sft_base",
                model_family="pythia",
            )

            result = apply_local_model_family(conf, working_directory=root)

            self.assertIs(result, conf)
            self.assertEqual(str(conf.model_name), "proxy_rm_sft_base")
            self.assertTrue("pythia" in conf.model_name)
            self.assertFalse("gpt-neox" in conf.model_name)
            self.assertEqual(root / Path(conf.model_name), model)
            restored_name = pickle.loads(pickle.dumps(conf.model_name))
            self.assertEqual(str(restored_name), "proxy_rm_sft_base")
            self.assertTrue("pythia" in restored_name)

    def test_local_rm_family_hint_rejects_wrong_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "proxy_rm_sft_base"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "bert"}), encoding="utf-8"
            )
            conf = SimpleNamespace(
                model_name="proxy_rm_sft_base",
                model_family="pythia",
            )
            with self.assertRaisesRegex(ValueError, "declares 'bert'"):
                apply_local_model_family(conf, working_directory=root)

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
        self.assertIn("tracker: null", smoke_text)
        self.assertIn("debug: false", smoke_overlay)

    def test_ppo_smoke_sbatch_is_isolated_one_update_and_no_gold(self) -> None:
        job_text = (ROOT / "scripts/slurm/smoke_ppo.sbatch").read_text(
            encoding="utf-8"
        )
        for directive in (
            "#SBATCH --account=gts-yxie77-paid",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=32G",
            "#SBATCH --time=00:45:00",
            "#SBATCH --gres=gpu:1",
            '#SBATCH --constraint="A100|H100|H200"',
        ):
            self.assertIn(directive, job_text)
        self.assertIn("set +u", job_text)
        self.assertIn("export PYTHONNOUSERSITE=1", job_text)
        self.assertIn("--asset initial_sft_policy", job_text)
        self.assertIn("configs/data_split_prompt_disjoint_v1.yaml", job_text)
        self.assertIn("prompt-disjoint-smoke_seed1}", job_text)
        self.assertIn("prompt_disjoint_data_split_v1 baseline_smoke", job_text)
        self.assertIn("--run_gold_evaluation false", job_text)
        self.assertIn("PASS proxy-RM checkpoint validation", job_text)
        self.assertIn("python scripts/validate_legacy_sources.py", job_text)
        self.assertIn("PASS finite PPO smoke metrics", job_text)
        self.assertIn("runs/ppo_smoke_checkpoints", job_text)
        self.assertIn("PASS one-update PPO pipeline smoke", job_text)
        self.assertNotIn("configs/ppo_config.yaml", job_text)

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

        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('--src "$CONDA_PREFIX/legacy-src"', readme_text)
        self.assertIn('mv "$PROJECT_ROOT/src/trlx"', readme_text)

        project_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("git+https://", project_text)
        runtime_text = (ROOT / "requirements" / "legacy-runtime.txt").read_text(encoding="utf-8")
        self.assertIn("typer==0.9.0", runtime_text)

        validator_text = (
            ROOT / "scripts" / "validate_legacy_sources.py"
        ).read_text(encoding="utf-8")
        for package_name, source_name in (
            ("model_training", "open_assistant"),
            ("oasst_data", "open_assistant"),
            ("trlx", "trlx"),
            ("alpaca_farm", "coste_alpaca_farm_fork"),
        ):
            self.assertIn(f'("{package_name}", "{source_name}")', validator_text)
        self.assertIn('getattr(module, "train", None)', validator_text)
        self.assertIn('importlib.import_module("trlx.trlx")', validator_text)
        self.assertIn("editable checkout placed inside", validator_text)

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
        self.assertIn("matplotlib=3.7.2", environment_text)
        runtime_text = (ROOT / "requirements" / "legacy-runtime.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("pathtools==0.1.2", runtime_text)
        self.assertIn("threadpoolctl>=3.1.0", runtime_text)
        self.assertIn("fsspec[http]==2023.9.2", runtime_text)
        for constraints in (conda_constraints, wheel_constraints):
            self.assertIn("fsspec==2023.9.2", constraints)
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

    def test_proxy_rm_sbatch_requests_gpu_and_runs_manifest_trainer(self) -> None:
        job_text = (ROOT / "scripts/slurm/train_proxy_rm.sbatch").read_text(
            encoding="utf-8"
        )
        for directive in (
            "#SBATCH --account=gts-yxie77-paid",
            "#SBATCH --mem=32G",
            "#SBATCH --time=04:00:00",
            "#SBATCH --gres=gpu:1",
            '#SBATCH --constraint="A100|H100|H200"',
            "#SBATCH -qinferno",
        ):
            self.assertIn(directive, job_text)
        self.assertNotIn("#SBATCH --gres=gpu:1 -C A100", job_text)
        self.assertIn('[[ -z "${SLURM_JOB_ID:-}" ]]', job_text)
        self.assertIn("source scripts/configure_cluster_storage.sh", job_text)
        self.assertIn("assert torch.cuda.is_available()", job_text)
        self.assertIn("assert torch.cuda.is_bf16_supported()", job_text)
        self.assertIn("src/reward_modeling/training/trainer_rm_manifest.py", job_text)
        self.assertIn(
            "--configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-split",
            job_text,
        )
        self.assertIn('--rng_seed "$RM_SEED"', job_text)
        self.assertIn('OUTPUT_BASE="${RM_OUTPUT_BASE:-', job_text)
        self.assertIn('OUTPUT_DIR="${OUTPUT_BASE}_seed${RM_SEED}"', job_text)
        self.assertIn('--output_dir "$OUTPUT_BASE"', job_text)
        self.assertIn('export WANDB_DIR="$JOB_STORAGE_ROOT/wandb"', job_text)
        self.assertIn('CHECKSUM_DIR="${RM_CHECKSUM_DIR:-artifacts/checksums}"', job_text)
        self.assertIn('> "$CHECKSUM_FILE"', job_text)
        self.assertIn('RESUME_CHECKPOINT="${RM_RESUME_CHECKPOINT:-}"', job_text)
        self.assertIn('RESUME_ARGS+=(--resume_from_checkpoint)', job_text)
        self.assertIn('"${RESUME_ARGS[@]}"', job_text)
        self.assertIn("PASS resumable RM checkpoint", job_text)
        self.assertIn("trainer_state.json", job_text)
        self.assertIn("optimizer.pt", job_text)
        self.assertIn("scheduler.pt", job_text)
        self.assertIn("rng_state.pth", job_text)
        self.assertIn("the legacy trainer resumes only the latest checkpoint", job_text)
        for expected in (
            'RM_LABEL_NOISE_RATE="${RM_LABEL_NOISE_RATE:-0.0}"',
            'RM_LABEL_NOISE_SEED="${RM_LABEL_NOISE_SEED:-1}"',
            "rm-pythia-44m-prompt-disjoint-label-noise-${NOISE_TAG}",
            '--rm_label_noise_rate "$RM_LABEL_NOISE_RATE"',
            '--rm_label_noise_seed "$RM_LABEL_NOISE_SEED"',
            "scripts/validate_rm_label_noise.py",
        ):
            self.assertIn(expected, job_text)

    def test_proxy_rm_jobs_make_conda_activation_nounset_safe(self) -> None:
        for filename in (
            "train_proxy_rm.sbatch",
            "train_proxy_rm_1p4b.sbatch",
            "smoke_proxy_rm.sbatch",
            "smoke_proxy_rm_any_gpu.sbatch",
            "smoke_proxy_rm_1p4b.sbatch",
        ):
            with self.subTest(filename=filename):
                job_text = (ROOT / "scripts/slurm" / filename).read_text(
                    encoding="utf-8"
                )
                self.assertNotIn("conda activate mfg_flow", job_text)
                nounset_off = job_text.index("set +u")
                hook = job_text.index('eval "$(conda shell.bash hook)"')
                activate = job_text.index('conda activate "$CONDA_ENV_NAME"')
                nounset_on = job_text.index("set -u", activate)
                self.assertLess(nounset_off, hook)
                self.assertLess(hook, activate)
                self.assertLess(activate, nounset_on)

    def test_matched_capacity_rm_track_reuses_sft_asset_and_effective_batch(self) -> None:
        config_text = (ROOT / "configs/config_rm_cluster.yaml").read_text(
            encoding="utf-8"
        )
        full_overlay = config_text.split("rm-pythia-1p4b-cluster-split:", 1)[1].split(
            "rm-pythia-1p4b-cluster-smoke:", 1
        )[0]
        for setting in (
            "model_name: assets/initial_sft_policy",
            "model_family: pythia",
            "data_split_manifest_path: "
            "data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json",
            "gradient_checkpointing: true",
            "gradient_accumulation_steps: 32",
            "per_device_train_batch_size: 1",
            "per_device_eval_batch_size: 1",
            "reward_normalization_batch_size: 1",
        ):
            self.assertIn(setting, full_overlay)

        # The overlay is merged after the Coste entry, whose microbatch 8 and
        # accumulation 4 also produce effective batch 32.
        self.assertEqual(1 * 32, 8 * 4)

        wrapper = (
            ROOT / "src/reward_modeling/training/trainer_rm_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertIn("reward_normalization_batch_size", wrapper)
        self.assertIn("batch_size=min(batch_size, normalization_batch_size)", wrapper)

    def test_matched_capacity_rm_jobs_are_isolated_scratch_backed_and_resumable(self) -> None:
        full_job = (ROOT / "scripts/slurm/train_proxy_rm_1p4b.sbatch").read_text(
            encoding="utf-8"
        )
        for expected in (
            "#SBATCH --mem=64G",
            "#SBATCH --time=48:00:00",
            '#SBATCH --constraint="A100|H100|H200"',
            "$JOB_STORAGE_ROOT/models/rm-pythia-1p4b-prompt-disjoint",
            "--asset initial_sft_policy",
            "--configs defaults_rm rm-pythia-44m rm-pythia-1p4b-cluster-split",
            'RESUME_CHECKPOINT="${RM_RESUME_CHECKPOINT:-}"',
            "the legacy trainer resumes only the latest checkpoint",
            "final_eval_results.json",
            "PASS matched-capacity 1.4B proxy RM",
            "rm-pythia-1p4b-prompt-disjoint-label-noise-${NOISE_TAG}",
            '--rm_label_noise_rate "$RM_LABEL_NOISE_RATE"',
            '--rm_label_noise_seed "$RM_LABEL_NOISE_SEED"',
            "scripts/validate_rm_label_noise.py",
        ):
            self.assertIn(expected, full_job)

        smoke_job = (ROOT / "scripts/slurm/smoke_proxy_rm_1p4b.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("--asset initial_sft_policy", smoke_job)
        self.assertIn(
            "--configs defaults_rm rm-pythia-44m rm-pythia-1p4b-cluster-smoke",
            smoke_job,
        )
        self.assertIn("PASS 1.4B RM smoke validation", smoke_job)
        self.assertIn("Smoke-only checkpoint", smoke_job)

    def test_proxy_scoring_batches_are_runtime_configurable_for_large_rm(self) -> None:
        for filename in (
            "train_reward_overoptimization.sbatch",
            "evaluate_reward_overoptimization.sbatch",
            "train_cpdpo_v2.sbatch",
            "evaluate_cpdpo_v2.sbatch",
            "evaluate_additive_reward_overoptimization.sbatch",
            "train_advpo.sbatch",
            "evaluate_advpo.sbatch",
            "smoke_reward_overoptimization.sbatch",
            "smoke_cpdpo_v2.sbatch",
            "smoke_advpo.sbatch",
        ):
            with self.subTest(filename=filename):
                text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
                self.assertIn("CPDPO_PROXY_BATCH_SIZE", text)
                self.assertIn('--proxy-batch-size "$PROXY_BATCH_SIZE"', text)

        for filename in (
            "prepare_cpdpo_artifacts.sbatch",
            "prepare_cpdpo_smoke_artifacts.sbatch",
        ):
            text = (ROOT / "scripts/slurm" / filename).read_text(encoding="utf-8")
            self.assertIn("CPDPO_ARTIFACT_BATCH_SIZE", text)
            self.assertIn('--batch-size "$ARTIFACT_BATCH_SIZE"', text)

        confidence = (ROOT / "scripts/slurm/prepare_advpo_confidence.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("ADVPO_ARTIFACT_BATCH_SIZE", confidence)
        self.assertIn('--batch-size "$ARTIFACT_BATCH_SIZE"', confidence)

    def test_proxy_rm_smoke_job_is_small_isolated_and_manifest_backed(self) -> None:
        config_text = (ROOT / "configs/config_rm_cluster.yaml").read_text(
            encoding="utf-8"
        )
        smoke_overlay = config_text.split("rm-pythia-44m-cluster-smoke:", 1)[1].split(
            "rm-pythia-44m-cluster-smoke-fp16:", 1
        )[0]
        for setting in (
            "size: 512",
            "eval_size: 128",
            "num_train_epochs: 1",
            "warmup_steps: 2",
            "logging_steps: 1",
            "eval_steps: 8",
            'save_strategy: "no"',
        ):
            self.assertIn(setting, smoke_overlay)
        self.assertIn(
            "output_dir: models/rm-pythia-44m-prompt-disjoint-smoke",
            smoke_overlay,
        )
        self.assertIn(
            "data_split_manifest_path: "
            "data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json",
            smoke_overlay,
        )

        job_text = (ROOT / "scripts/slurm/smoke_proxy_rm.sbatch").read_text(
            encoding="utf-8"
        )
        for directive in (
            "#SBATCH --account=gts-yxie77-paid",
            "#SBATCH --mem=16G",
            "#SBATCH --gres=gpu:1 -C A100",
            "#SBATCH -t30",
        ):
            self.assertIn(directive, job_text)
        self.assertIn("--asset proxy_rm_sft_base", job_text)
        self.assertIn("configs/data_split_prompt_disjoint_v1.yaml", job_text)
        self.assertIn(
            "--configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-smoke",
            job_text,
        )
        self.assertIn("prompt-disjoint-smoke_seed${RM_SEED}", job_text)
        self.assertIn("PASS finite reward normalization", job_text)
        self.assertIn('RM_LABEL_NOISE_RATE="${RM_LABEL_NOISE_RATE:-0.0}"', job_text)
        self.assertIn('--rm_label_noise_rate "$RM_LABEL_NOISE_RATE"', job_text)
        self.assertIn("scripts/validate_rm_label_noise.py", job_text)
        self.assertIn(
            "Smoke checkpoint (use only for baseline_smoke PPO integration)",
            job_text,
        )
        rl_config = (ROOT / "configs/config_rl.yaml").read_text(encoding="utf-8")
        self.assertNotIn("prompt-disjoint-smoke", rl_config)
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/models/", gitignore)

    def test_generic_gpu_rm_smoke_uses_fp16_without_flash_attention(self) -> None:
        config_text = (ROOT / "configs/config_rm_cluster.yaml").read_text(
            encoding="utf-8"
        )
        fp16_overlay = config_text.split(
            "rm-pythia-44m-cluster-smoke-fp16:", 1
        )[1].split("rm-pythia-44m-cluster-coste-native-split:", 1)[0]
        self.assertIn(
            "output_dir: models/rm-pythia-44m-prompt-disjoint-smoke-fp16",
            fp16_overlay,
        )
        self.assertIn("dtype: fp16", fp16_overlay)
        self.assertIn("use_flash_attention: false", fp16_overlay)

        accelerate_text = (ROOT / "configs/accelerate_config_fp16.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("mixed_precision: fp16", accelerate_text)
        self.assertNotIn("mixed_precision: bf16", accelerate_text)

        job_text = (ROOT / "scripts/slurm/smoke_proxy_rm_any_gpu.sbatch").read_text(
            encoding="utf-8"
        )
        for directive in (
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=16G",
            "#SBATCH -t30",
            "#SBATCH --gres=gpu:1\n",
        ):
            self.assertIn(directive, job_text)
        self.assertNotIn("-C A100", job_text)
        self.assertNotIn("is_bf16_supported", job_text)
        self.assertIn("configs/accelerate_config_fp16.yaml", job_text)
        self.assertIn(
            "--configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-smoke "
            "rm-pythia-44m-cluster-smoke-fp16",
            job_text,
        )
        self.assertIn("prompt-disjoint-smoke-fp16_seed${RM_SEED}", job_text)
        self.assertIn("PASS generic-GPU FP16 proxy-RM pipeline smoke", job_text)

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
