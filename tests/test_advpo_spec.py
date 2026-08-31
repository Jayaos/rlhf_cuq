from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.advpo.spec import ADVPO_PAPER_B_GRID, AdvPOConfig, advpo_run_name, number_tag
from src.cpdpo.experiment import resolve_training_budget


ROOT = Path(__file__).resolve().parents[1]


class AdvPOContractTests(unittest.TestCase):
    def test_B_has_named_run_identity_and_paper_grid(self) -> None:
        self.assertEqual(ADVPO_PAPER_B_GRID, (1.0, 5.0, 10.0, 15.0))
        self.assertEqual(number_tag(0.25), "0p25")
        self.assertEqual(advpo_run_name(5), "advpo_B_5")
        self.assertEqual(AdvPOConfig(confidence_radius_squared=9).confidence_radius, 3.0)
        with self.assertRaises(TypeError):
            AdvPOConfig()
        with self.assertRaisesRegex(ValueError, "positive"):
            AdvPOConfig(confidence_radius_squared=0)
        with self.assertRaisesRegex(ValueError, "prompt pairs"):
            AdvPOConfig(confidence_radius_squared=1, adversarial_batch_responses=63)

    def test_advpo_uses_equal_scalar_ppo_budget(self) -> None:
        budget = resolve_training_budget(
            "advpo", prompts_per_rollout=256, pair_batch_size=32, ppo_epochs=4
        )
        self.assertEqual(budget.response_count_per_rollout, 512)
        self.assertEqual(budget.proxy_rm_calls_per_rollout, 512)
        self.assertEqual(budget.trainer_rollout_units, 512)
        self.assertEqual(budget.trainer_batch_units, 64)
        self.assertEqual(budget.optimizer_updates_per_rollout, 32)

    def test_advpo_entry_points_never_accept_gold(self) -> None:
        for relative in (
            "scripts/prepare_advpo_confidence.py",
            "scripts/prepare_advpo_references.py",
            "src/ppo/trainer_reward_overoptimization.py",
        ):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            options = {
                argument.value
                for call in ast.walk(tree)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
                for argument in call.args
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            }
            self.assertFalse(any("gold" in option.lower() for option in options), relative)

    def test_advpo_is_additive_and_does_not_reuse_cpdpo_geometry(self) -> None:
        trainer = (ROOT / "src/ppo/trainer_reward_overoptimization.py").read_text(encoding="utf-8")
        confidence = (ROOT / "scripts/prepare_advpo_confidence.py").read_text(encoding="utf-8")
        train_job = (ROOT / "scripts/slurm/train_advpo.sbatch").read_text(encoding="utf-8")
        v1_job = (ROOT / "scripts/slurm/train_reward_overoptimization.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn('args.method == "advpo"', trainer)
        self.assertIn('trlx_config.train.trainer = "ExperimentAcceleratePPOTrainer"', trainer)
        self.assertIn("build_advpo_confidence_geometry", confidence)
        self.assertNotIn("build_pair_geometry", confidence)
        self.assertNotIn("conformal", confidence.lower())
        self.assertIn("--method advpo", train_job)
        self.assertIn("--advpo-B", train_job)
        self.assertNotIn("advpo", v1_job.lower())
        self.assertIn(
            "scale_reward: false",
            (ROOT / "configs/ppo_config_reward_overoptimization.yaml").read_text(
                encoding="utf-8"
            ),
        )

    def test_plot_loader_selects_named_advpo_B(self) -> None:
        module_path = ROOT / "scripts/aggregate_and_plot_reward_overoptimization.py"
        spec = importlib.util.spec_from_file_location("advpo_plot_script", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for method, run_variant in (
                ("ppo", "ppo"),
                ("pairppo", "pairppo"),
                ("cpdpo", "cpdpo"),
                ("advpo", "advpo_B_5"),
            ):
                run = root / "seed_1" / run_variant
                metrics = run / "evaluation" / "D_rl_val_prompts" / "checkpoint_metrics.jsonl"
                metrics.parent.mkdir(parents=True)
                metadata = {
                    "method": method,
                    "run_variant": run_variant,
                    "cpdpo_alpha": 0.10 if method == "cpdpo" else None,
                    "advpo_B": 5.0 if method == "advpo" else None,
                    "prompt_schedule_sha256": "schedule",
                    "prompt_id_sequence_sha256": "prompt-ids",
                }
                (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                metrics.write_text(
                    json.dumps(
                        {
                            "method": method,
                            "seed": 1,
                            "rollout_step": 0,
                            "proxy_reward_mean": 1.0,
                            "gold_reward_mean": 2.0,
                            "eval_kl_mean": 0.0,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            selected = module.load_records(
                root,
                "D_rl_val_prompts",
                include_advpo=True,
                advpo_B=5.0,
            )
        self.assertEqual(
            {record["method"] for record in selected},
            {"ppo", "pairppo", "cpdpo", "advpo"},
        )
        self.assertEqual(next(row for row in selected if row["method"] == "advpo")["_advpo_B"], 5.0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is validated in the cluster environment")
class AdvPOTorchTests(unittest.TestCase):
    def test_confidence_matrix_is_unnormalized_individual_feature_sum(self) -> None:
        import torch

        from src.advpo.geometry import build_advpo_confidence_geometry

        geometry, gram_sum = build_advpo_confidence_geometry(
            [torch.tensor([[1.0, 0.0], [0.0, 2.0]])], ridge_lambda=1.0
        )
        self.assertEqual(geometry.n_responses, 2)
        self.assertTrue(
            torch.equal(gram_sum, torch.tensor([[1.0, 0.0], [0.0, 4.0]], dtype=torch.float64))
        )
        matrix = geometry.cholesky @ geometry.cholesky.T
        self.assertTrue(
            torch.allclose(matrix, torch.tensor([[2.0, 0.0], [0.0, 5.0]], dtype=torch.float64))
        )

    def test_closed_form_advpo_uses_one_shared_direction(self) -> None:
        import torch

        from src.advpo.geometry import AdvPOConfidenceGeometry, advpo_batch_signal

        geometry = AdvPOConfidenceGeometry(torch.eye(2, dtype=torch.float64), 1.0, 1)
        current_features = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        reference_features = torch.zeros_like(current_features)
        head = torch.tensor([3.0, 4.0])
        bias = 2.0
        current_rewards = current_features @ head + bias
        reference_rewards = reference_features @ head + bias
        signal = advpo_batch_signal(
            current_rewards=current_rewards,
            reference_rewards=reference_rewards,
            current_features=current_features,
            reference_features=reference_features,
            geometry=geometry,
            confidence_radius_squared=4.0,
        )
        expected_direction = torch.tensor([2.0**0.5, 2.0**0.5], dtype=torch.float64)
        self.assertTrue(torch.allclose(signal["adversarial_direction"], expected_direction))
        self.assertTrue(
            torch.allclose(
                signal["current_penalty"],
                current_features.double() @ expected_direction,
            )
        )
        self.assertAlmostEqual(
            signal["robust_objective"].item(), signal["closed_form_objective"].item()
        )
        self.assertFalse(signal["current_adversarial_reward"].requires_grad)

    def test_zero_mean_feature_difference_has_finite_zero_displacement(self) -> None:
        import torch

        from src.advpo.geometry import AdvPOConfidenceGeometry, advpo_batch_signal

        geometry = AdvPOConfidenceGeometry(torch.eye(2, dtype=torch.float64), 1.0, 2)
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rewards = torch.tensor([1.0, 2.0])
        signal = advpo_batch_signal(
            current_rewards=rewards,
            reference_rewards=rewards,
            current_features=features,
            reference_features=features,
            geometry=geometry,
            confidence_radius_squared=1.0,
        )
        self.assertTrue(torch.equal(signal["adversarial_direction"], torch.zeros(2, dtype=torch.float64)))
        self.assertEqual(signal["lambda_star"].item(), 0.0)
        self.assertTrue(signal["degenerate_direction"].item())
        self.assertTrue(torch.isfinite(signal["current_adversarial_reward"]).all())


if __name__ == "__main__":
    unittest.main()
