from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from src.cpdpo.experiment import (
    assert_equal_method_budgets,
    resolve_training_budget,
    validate_policy_quality_record,
)
from src.cpdpo.evaluation import checkpoint_summary, format_alpaca_gold_sample
from src.cpdpo.math_spec import (
    calibration_score,
    finite_sample_quantile,
    finite_sample_quantile_rank,
    pair_signal,
)
from src.cpdpo.prompt_schedule import (
    ExperimentSeeds,
    build_prompt_schedule,
    load_schedule_as_duplicated_prompts,
)
from src.cpdpo.spec import CPDPOConfig
from src.cpdpo.run_logging import (
    append_rollout_record,
    archive_pair_rollouts_after_checkpoint,
    load_rollout_records,
    rewind_rollout_records,
)


ROOT = Path(__file__).resolve().parents[1]


class CPDPOMathSpecTests(unittest.TestCase):
    def test_finite_sample_higher_quantile_has_no_interpolation(self) -> None:
        self.assertEqual(finite_sample_quantile_rank(10, 0.10), 10)
        self.assertEqual(finite_sample_quantile([0, 4, 1, 3, 2], 0.20), (4.0, 5))

    def test_calibration_score_uses_only_misranking_hinge(self) -> None:
        self.assertEqual(calibration_score(1, 2.0, 4.0, 1e-8), 0.0)
        self.assertAlmostEqual(calibration_score(1, -2.0, 4.0, 1e-8), 2.0 / (4.0 + 1e-8))
        self.assertAlmostEqual(calibration_score(-1, 2.0, 4.0, 1e-8), 2.0 / (4.0 + 1e-8))

    def test_pairppo_identity_and_cpdpo_strict_boundary(self) -> None:
        pairppo = pair_signal("pairppo", -3.0, 99.0, 42.0, 1e-8)
        self.assertEqual(pairppo["reward"], -3.0)
        boundary = pair_signal("cpdpo", 2.0, 1.0, 2.0 / (1.0 + 1e-8), 1e-8)
        self.assertFalse(boundary["certified"])
        self.assertEqual(boundary["reward"], 0.0)
        positive = pair_signal("cpdpo", 3.0, 1.0, 2.0, 1e-8)
        negative = pair_signal("cpdpo", -3.0, 1.0, 2.0, 1e-8)
        self.assertAlmostEqual(positive["reward"], 1.0)
        self.assertAlmostEqual(negative["reward"], -1.0)

    def test_required_ablation_flags_are_explicit(self) -> None:
        config = CPDPOConfig(method="cpdpo", reward_variant="sign_only", geometry_mode="unit")
        self.assertEqual(config.reward_variant, "sign_only")
        self.assertEqual(config.geometry_mode, "unit")
        self.assertTrue(config.log_pair_records)
        with self.assertRaisesRegex(ValueError, "PairPPO"):
            CPDPOConfig(method="pairppo", reward_variant="sign_only")
        with self.assertRaisesRegex(ValueError, "persistence"):
            CPDPOConfig(method="cpdpo", log_pair_records=False)


class CPDPOExperimentContractTests(unittest.TestCase):
    def test_evaluator_normalizes_path_for_pinned_trlx_loader(self) -> None:
        evaluator = ROOT / "scripts/evaluate_policy_checkpoints.py"
        tree = ast.parse(evaluator.read_text(encoding="utf-8"))
        load_policy = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_policy"
        )
        loader_call = next(
            node
            for node in ast.walk(load_policy)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        )
        self.assertTrue(loader_call.args)
        self.assertIsInstance(loader_call.args[0], ast.Call)
        self.assertIsInstance(loader_call.args[0].func, ast.Name)
        self.assertEqual(loader_call.args[0].func.id, "str")

    def test_pinned_alpaca_gold_prompt_has_no_literal_placeholders(self) -> None:
        value = format_alpaca_gold_sample("Do it", "context", "answer")
        self.assertIn("### Instruction:\nDo it", value)
        self.assertIn("### Input:\ncontext", value)
        self.assertTrue(value.endswith("### Response:answer"))
        self.assertNotIn("{instruction}", value)

    def test_proxy_evaluation_format_adds_one_terminal_eos(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch is validated in the pinned cluster environment")
        from src.cpdpo.reward_features import format_proxy_samples

        training = format_proxy_samples(["prompt"], ["answer"])[0]
        evaluation = format_proxy_samples(["prompt"], ["answer"], evaluation=True)[0]
        self.assertFalse(training.endswith("<|endoftext|>"))
        self.assertEqual(evaluation, training + "<|endoftext|>")

    def test_checkpoint_summary_uses_same_response_rows(self) -> None:
        rows = [
            {"response_id": "a", "proxy_reward": 1.0, "gold_reward": 2.0, "sampled_kl": 4.0},
            {"response_id": "b", "proxy_reward": 3.0, "gold_reward": 4.0, "sampled_kl": 0.0},
        ]
        summary = checkpoint_summary(rows)
        self.assertEqual(summary["proxy_reward_mean"], 2.0)
        self.assertEqual(summary["gold_reward_mean"], 3.0)
        self.assertEqual(summary["eval_kl_mean"], 2.0)
        self.assertAlmostEqual(summary["sqrt_eval_kl"], 2.0 ** 0.5)

    def test_equal_response_proxy_and_update_budget(self) -> None:
        budgets = [
            resolve_training_budget(method, prompts_per_rollout=256, pair_batch_size=32, ppo_epochs=4)
            for method in ("ppo", "pairppo", "cpdpo")
        ]
        assert_equal_method_budgets(budgets)
        for budget in budgets:
            self.assertEqual(budget.response_count_per_rollout, 512)
            self.assertEqual(budget.proxy_rm_calls_per_rollout, 512)
            self.assertEqual(budget.optimizer_updates_per_rollout, 32)

    def test_seed_namespaces_and_prompt_schedule_are_deterministic(self) -> None:
        seeds = ExperimentSeeds.from_base(7)
        self.assertEqual((seeds.training, seeds.rollout_generation, seeds.evaluation_generation, seeds.prompt_schedule),
                         (7, 10007, 20007, 30007))
        records = [
            {"_split_prompt_id": f"p{i}", "instruction": f"i{i}", "input": ""}
            for i in range(5)
        ]
        first = build_prompt_schedule(records, rollout_steps=2, prompts_per_rollout=3, seed=seeds.prompt_schedule)
        second = build_prompt_schedule(records, rollout_steps=2, prompts_per_rollout=3, seed=seeds.prompt_schedule)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)

    def test_schedule_expands_each_prompt_to_adjacent_a_b(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.jsonl"
            path.write_text(
                '{"rollout_step": 0, "within_rollout": 0, "prompt_id": "p", '
                '"instruction": "hello", "input": "world"}\n',
                encoding="utf-8",
            )
            values = load_schedule_as_duplicated_prompts(path)
        self.assertEqual([value["orientation"] for value in values], ["a", "b"])
        self.assertEqual(values[0]["prompt_id"], values[1]["prompt_id"])
        self.assertEqual(values[0]["prompt"], values[1]["prompt"])

    def test_training_cli_does_not_accept_gold_checkpoint(self) -> None:
        trainer_path = ROOT / "src/ppo/trainer_reward_overoptimization.py"
        tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
        option_strings = {
            argument.value
            for call in ast.walk(tree)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"
            for argument in call.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        }
        self.assertFalse(any("gold" in option.lower() for option in option_strings))

    def test_smoke_artifacts_require_an_explicit_smoke_only_training_opt_in(self) -> None:
        smoke_job = (ROOT / "scripts/slurm/smoke_reward_overoptimization.sbatch").read_text(encoding="utf-8")
        full_job = (ROOT / "scripts/slurm/train_reward_overoptimization.sbatch").read_text(encoding="utf-8")
        artifact_job = (ROOT / "scripts/slurm/prepare_cpdpo_smoke_artifacts.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("--allow-smoke-artifacts", smoke_job)
        self.assertNotIn("--allow-smoke-artifacts", full_job)
        self.assertIn("--max-rm-pairs", artifact_job)
        self.assertIn("--max-cal-pairs", artifact_job)

    def test_plot_records_reject_internal_pair_reward(self) -> None:
        record = {
            "method": "cpdpo",
            "seed": 1,
            "rollout_step": 10,
            "proxy_reward_mean": 1.0,
            "gold_reward_mean": 2.0,
            "eval_kl_mean": 3.0,
        }
        validate_policy_quality_record(record)
        with self.assertRaisesRegex(ValueError, "Internal pair rewards"):
            validate_policy_quality_record({**record, "pair_reward_mean": 0.5})

    def test_rollout_accounting_is_cumulative_and_monotone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for step in (1, 2):
                append_rollout_record(
                    directory,
                    {
                        "rollout_step": step,
                        "response_count": 4,
                        "generated_token_count": 10,
                        "proxy_call_count": 4,
                    },
                )
            rows = load_rollout_records(directory)
        self.assertEqual(rows[2]["generated_responses"], 8)
        self.assertEqual(rows[2]["generated_tokens"], 20)
        self.assertEqual(rows[2]["proxy_rm_calls"], 8)

    def test_resume_rewinds_only_uncheckpointed_rollout_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for step in (1, 2, 3):
                append_rollout_record(
                    directory,
                    {
                        "rollout_step": step,
                        "response_count": 4,
                        "generated_token_count": 10,
                        "proxy_call_count": 4,
                    },
                )
            backup = rewind_rollout_records(directory, 2)
            pair_dir = Path(directory) / "pair_rollouts"
            pair_dir.mkdir()
            for step in (1, 2, 3):
                (pair_dir / f"rollout_{step:06d}.pt").write_bytes(str(step).encode("ascii"))
            pair_backup = archive_pair_rollouts_after_checkpoint(directory, 2)
            rows = load_rollout_records(directory)
            self.assertIsNotNone(backup)
            self.assertTrue(backup.is_file())
            self.assertEqual(sorted(rows), [1, 2])
            self.assertEqual(len(backup.read_text(encoding="utf-8").splitlines()), 3)
            self.assertTrue(pair_backup.is_dir())
            self.assertTrue((pair_backup / "rollout_000003.pt").is_file())
            self.assertFalse((pair_dir / "rollout_000003.pt").exists())


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("trlx"),
    "torch/trlx are validated in the pinned cluster environment",
)
class CPDPOTorchTests(unittest.TestCase):
    def test_smoke_artifact_scope_is_rejected_unless_explicitly_allowed(self) -> None:
        from src.cpdpo.pair_reward import PairRewardCallback

        callback = PairRewardCallback.__new__(PairRewardCallback)
        callback.artifact_scope = None
        callback.allow_smoke_artifacts = False
        with self.assertRaisesRegex(ValueError, "Smoke-only"):
            callback._validate_artifact_scope({"artifact_scope": "smoke"}, "test")

        callback.artifact_scope = None
        callback.allow_smoke_artifacts = True
        callback._validate_artifact_scope({"artifact_scope": "smoke"}, "test")
        self.assertEqual(callback.artifact_scope, "smoke")
        with self.assertRaisesRegex(ValueError, "scopes do not match"):
            callback._validate_artifact_scope({"artifact_scope": "scientific"}, "test")

    def test_low_certification_warning_after_three_rollouts(self) -> None:
        from src.ppo.custom_trlx_trainers import custom_accelerate_pair_ppo_trainer as module

        trainer = module.CustomAcceleratePairPPOTrainer.__new__(module.CustomAcceleratePairPPOTrainer)
        trainer.pair_config = CPDPOConfig(method="cpdpo")
        trainer.low_certification_rollouts = 0
        with mock.patch.object(module.logger, "warning") as warning:
            trainer._update_certification_monitor(0.05)
            trainer._update_certification_monitor(0.05)
            warning.assert_not_called()
            trainer._update_certification_monitor(0.05)
            warning.assert_called_once()
        self.assertEqual(trainer.low_certification_rollouts, 3)

    def test_geometry_calibration_and_pair_loss(self) -> None:
        import torch

        from src.cpdpo.geometry import build_pair_geometry, calibration_scores, conformal_quantile, pair_signals
        from src.cpdpo.pair_loss import pairwise_clipped_loss

        geometry, gram = build_pair_geometry([torch.tensor([[1.0, 0.0], [0.0, 2.0]])])
        self.assertEqual(geometry.n_rm, 2)
        self.assertTrue(torch.allclose(gram, torch.tensor([[0.5, 0.0], [0.0, 2.0]], dtype=torch.float64)))
        self.assertAlmostEqual(geometry.ridge, 1.0e-3 * 2.5 / 2)
        matrix = geometry.cholesky @ geometry.cholesky.T
        self.assertTrue(torch.all(torch.linalg.eigvalsh(matrix) > 0))
        uncertainty = geometry.uncertainty(torch.tensor([[1.0, 0.0]], dtype=torch.float64))
        self.assertTrue(torch.isfinite(uncertainty).all())
        zero_geometry, zero_gram = build_pair_geometry([torch.zeros((3, 2))])
        self.assertEqual(torch.trace(zero_gram).item(), 0.0)
        self.assertEqual(zero_geometry.ridge, 1.0e-6)
        scores = calibration_scores(torch.tensor([1.0, -1.0]), torch.tensor([-2.0, 3.0]), torch.ones(2))
        q, rank = conformal_quantile(scores, alpha=0.10)
        self.assertEqual(rank, 2)
        self.assertEqual(q, scores.max())
        self.assertEqual(pair_signals(torch.tensor([3.0]), torch.tensor([1.0]), method="cpdpo", q_alpha=2.0)["pair_reward"].item(), 1.0)
        pairppo = pair_signals(torch.tensor([3.0]), None, method="pairppo")
        self.assertTrue(torch.isfinite(pairppo["normalized_margin"]).all())
        self.assertEqual(pairppo["normalized_margin"].item(), 0.0)
        sign_only = pair_signals(
            torch.tensor([3.0, -3.0]),
            torch.tensor([1.0, 1.0]),
            method="cpdpo",
            q_alpha=2.0,
            reward_variant="sign_only",
        )["pair_reward"]
        self.assertTrue(torch.equal(sign_only, torch.tensor([1.0, -1.0])))

        new_a = torch.zeros((1, 2), requires_grad=True)
        new_b = torch.zeros((1, 3), requires_grad=True)
        loss, _ = pairwise_clipped_loss(
            logprobs_a=new_a,
            logprobs_b=new_b,
            old_logprobs_a=torch.zeros_like(new_a),
            old_logprobs_b=torch.zeros_like(new_b),
            ref_logprobs_a=torch.zeros_like(new_a),
            ref_logprobs_b=torch.zeros_like(new_b),
            mask_a=torch.ones_like(new_a),
            mask_b=torch.ones_like(new_b),
            pair_rewards=torch.tensor([2.0]),
            clip_epsilon=0.2,
            kl_beta=0.0,
        )
        # 0.5 * (2 tokens * 2 + 3 tokens * -2) = -1, so loss = 1.
        self.assertAlmostEqual(loss.item(), 1.0)

    def test_geometry_distinguishes_covered_and_uncovered_directions(self) -> None:
        import torch

        from src.cpdpo.geometry import build_pair_geometry

        geometry, _gram = build_pair_geometry([torch.tensor([[2.0, 0.0]]).repeat(32, 1)])
        covered = geometry.uncertainty(torch.tensor([[1.0, 0.0]], requires_grad=True))
        uncovered = geometry.uncertainty(torch.tensor([[0.0, 1.0]]))
        self.assertLess(covered.item(), uncovered.item())
        self.assertFalse(covered.requires_grad)

    def test_pair_swap_preserves_uncertainty_and_flips_reward(self) -> None:
        import torch

        from src.cpdpo.geometry import build_pair_geometry, pair_signals

        geometry, _gram = build_pair_geometry([torch.tensor([[1.0, 0.0], [0.0, 1.0]])])
        difference = torch.tensor([[2.0, -1.0]])
        uncertainty_ab = geometry.uncertainty(difference)
        uncertainty_ba = geometry.uncertainty(-difference)
        signal_ab = pair_signals(torch.tensor([4.0]), uncertainty_ab, method="cpdpo", q_alpha=1.0)
        signal_ba = pair_signals(torch.tensor([-4.0]), uncertainty_ba, method="cpdpo", q_alpha=1.0)
        self.assertTrue(torch.allclose(uncertainty_ab, uncertainty_ba))
        self.assertTrue(torch.allclose(signal_ab["gamma"], signal_ba["gamma"]))
        self.assertTrue(torch.equal(signal_ab["certified"], signal_ba["certified"]))
        self.assertTrue(torch.allclose(signal_ab["pair_reward"], -signal_ba["pair_reward"]))

    def test_pair_signal_is_frozen_and_has_no_proxy_or_geometry_gradient(self) -> None:
        import torch

        from src.cpdpo.geometry import pair_signals

        margin = torch.tensor([3.0], requires_grad=True)
        uncertainty = torch.tensor([1.0], requires_grad=True)
        signals = pair_signals(margin, uncertainty, method="cpdpo", q_alpha=2.0)
        frozen = {key: value.clone() for key, value in signals.items()}
        self.assertTrue(all(not value.requires_grad for value in signals.values()))
        for _ in range(4):
            objective = signals["pair_reward"] * torch.tensor([1.0], requires_grad=True)
            objective.sum().backward()
        for key in signals:
            self.assertTrue(torch.equal(signals[key], frozen[key]))
        self.assertIsNone(margin.grad)
        self.assertIsNone(uncertainty.grad)

    def test_pair_loss_gradient_has_exact_orientation_and_pair_reduction(self) -> None:
        import torch

        from src.cpdpo.pair_loss import pairwise_clipped_loss

        new_a = torch.zeros((2, 2), requires_grad=True)
        new_b = torch.zeros((2, 2), requires_grad=True)
        rewards = torch.tensor([2.0, -4.0])
        loss, _stats = pairwise_clipped_loss(
            logprobs_a=new_a,
            logprobs_b=new_b,
            old_logprobs_a=torch.zeros_like(new_a),
            old_logprobs_b=torch.zeros_like(new_b),
            ref_logprobs_a=torch.zeros_like(new_a),
            ref_logprobs_b=torch.zeros_like(new_b),
            mask_a=torch.ones_like(new_a),
            mask_b=torch.ones_like(new_b),
            pair_rewards=rewards,
            clip_epsilon=0.2,
            kl_beta=0.0,
        )
        loss.backward()
        # loss = -mean_pairs[0.5 * (R sum_a - R sum_b)]
        expected_a = -rewards[:, None].expand_as(new_a) / (2 * rewards.numel())
        expected_b = rewards[:, None].expand_as(new_b) / (2 * rewards.numel())
        self.assertTrue(torch.allclose(new_a.grad, expected_a))
        self.assertTrue(torch.allclose(new_b.grad, expected_b))

    def test_q_zero_symmetrized_gradient_recovers_scalar_policy_gradient(self) -> None:
        import torch

        behavior_logits = torch.tensor([0.3, -0.2, 0.7])
        rewards = torch.tensor([-1.0, 0.5, 2.0])
        old_probabilities = torch.softmax(behavior_logits, dim=0).detach()

        scalar_logits = behavior_logits.clone().requires_grad_(True)
        scalar_probabilities = torch.softmax(scalar_logits, dim=0)
        scalar_objective = (scalar_probabilities * rewards).sum()
        scalar_gradient = torch.autograd.grad(scalar_objective, scalar_logits)[0]

        pair_logits = behavior_logits.clone().requires_grad_(True)
        pair_probabilities = torch.softmax(pair_logits, dim=0)
        pair_objective = torch.zeros((), dtype=pair_logits.dtype)
        for a in range(rewards.numel()):
            for b in range(rewards.numel()):
                pair_reward = rewards[a] - rewards[b]
                ratio_a = pair_probabilities[a] / old_probabilities[a]
                ratio_b = pair_probabilities[b] / old_probabilities[b]
                pair_objective = pair_objective + (
                    old_probabilities[a]
                    * old_probabilities[b]
                    * 0.5
                    * pair_reward
                    * (ratio_a - ratio_b)
                )
        pair_gradient = torch.autograd.grad(pair_objective, pair_logits)[0]
        self.assertTrue(torch.allclose(pair_gradient, scalar_gradient, atol=1e-6, rtol=1e-6))

    def test_zero_certified_batch_has_finite_zero_pair_gradient(self) -> None:
        import torch

        from src.cpdpo.pair_loss import pairwise_clipped_loss

        new_a = torch.randn((2, 3), requires_grad=True)
        new_b = torch.randn((2, 4), requires_grad=True)
        loss, _stats = pairwise_clipped_loss(
            logprobs_a=new_a,
            logprobs_b=new_b,
            old_logprobs_a=new_a.detach().clone(),
            old_logprobs_b=new_b.detach().clone(),
            ref_logprobs_a=new_a.detach().clone(),
            ref_logprobs_b=new_b.detach().clone(),
            mask_a=torch.ones_like(new_a),
            mask_b=torch.ones_like(new_b),
            pair_rewards=torch.zeros(2),
            clip_epsilon=0.2,
            kl_beta=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.equal(new_a.grad, torch.zeros_like(new_a)))
        self.assertTrue(torch.equal(new_b.grad, torch.zeros_like(new_b)))

    def test_all_uncertain_batch_retains_only_configured_kl(self) -> None:
        import torch

        from src.cpdpo.pair_loss import pairwise_clipped_loss

        new_a = torch.tensor([[0.2, -0.1]], requires_grad=True)
        new_b = torch.tensor([[-0.3, 0.4]], requires_grad=True)
        loss, stats = pairwise_clipped_loss(
            logprobs_a=new_a,
            logprobs_b=new_b,
            old_logprobs_a=new_a.detach().clone(),
            old_logprobs_b=new_b.detach().clone(),
            ref_logprobs_a=torch.zeros_like(new_a),
            ref_logprobs_b=torch.zeros_like(new_b),
            mask_a=torch.ones_like(new_a),
            mask_b=torch.ones_like(new_b),
            pair_rewards=torch.zeros(1),
            clip_epsilon=0.2,
            kl_beta=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["loss/pair"].item(), 0.0)
        self.assertGreater(stats["loss/kl"].item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(new_a.grad).all())
        self.assertTrue(torch.isfinite(new_b.grad).all())

    def test_feature_hook_reconstructs_exact_scalar_head(self) -> None:
        import torch

        from src.cpdpo.reward_features import RewardHeadFeatureExtractor

        class Output:
            def __init__(self, logits):
                self.logits = logits

        class FakeRewardModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 3)
                self.out_proj = torch.nn.Linear(3, 1, bias=True)

            def forward(self, input_ids, attention_mask):
                hidden = self.embedding(input_ids)
                index = attention_mask.cumsum(1).argmax(1)
                pooled = hidden.gather(1, index[:, None, None].expand(-1, 1, 3)).squeeze(1)
                return Output(self.out_proj(pooled))

        model = FakeRewardModel()
        extractor = RewardHeadFeatureExtractor(model)
        rewards, features = extractor.forward(
            torch.tensor([[1, 2, 0], [3, 4, 5]]), torch.tensor([[1, 1, 0], [1, 1, 1]])
        )
        reconstructed = torch.nn.functional.linear(features, model.out_proj.weight, model.out_proj.bias)[:, 0]
        self.assertTrue(torch.allclose(rewards, reconstructed))
        reward_difference = rewards[0] - rewards[1]
        feature_difference = features[0] - features[1]
        projected_difference = torch.dot(model.out_proj.weight[0], feature_difference)
        self.assertTrue(torch.allclose(reward_difference, projected_difference))

    def test_pair_store_collator_keeps_orientations_atomic(self) -> None:
        import torch

        from src.cpdpo.rollout_store import PairRolloutElement, pair_collate_fn

        def element(prompt_id, a, b):
            scalar = torch.tensor(1.0)
            return PairRolloutElement(
                prompt_id=prompt_id,
                behavior_policy_step=0,
                proxy_rm_fingerprint="proxy",
                geometry_fingerprint="geometry",
                calibration_fingerprint="calibration",
                query_tensor=torch.tensor([1, 2]),
                response_tensor_a=torch.tensor(a),
                response_tensor_b=torch.tensor(b),
                old_logprobs_a=torch.zeros(len(a)),
                old_logprobs_b=torch.zeros(len(b)),
                ref_logprobs_a=torch.zeros(len(a)),
                ref_logprobs_b=torch.zeros(len(b)),
                pair_reward=scalar,
                reward_a=scalar,
                reward_b=-scalar,
                margin=scalar,
                uncertainty=scalar,
                normalized_margin=scalar,
                certified=torch.tensor(True),
                gamma=scalar,
            )

        batch = pair_collate_fn("left", 0, [element("p1", [3], [4, 5]), element("p2", [6, 7], [8])])
        self.assertEqual(batch.prompt_ids, ["p1", "p2"])
        self.assertEqual(batch.behavior_policy_steps, [0, 0])
        self.assertEqual(batch.proxy_rm_fingerprints, ["proxy", "proxy"])
        self.assertEqual(batch.response_tensors_a.tolist(), [[3, 0], [6, 7]])
        self.assertEqual(batch.response_tensors_b.tolist(), [[4, 5], [8, 0]])
        with tempfile.TemporaryDirectory() as directory:
            from src.cpdpo.rollout_store import PairRolloutStorage

            store = PairRolloutStorage(0, "left")
            store.push([element("p1", [3], [4, 5]), element("p2", [6, 7], [8])])
            snapshot = store.save_rollout_snapshot(directory, 1)
            payload = torch.load(snapshot, map_location="cpu")
            self.assertEqual(payload["pair_count"], 2)
            self.assertEqual(payload["records"][0]["response_mask_a"].tolist(), [True])
            self.assertEqual(payload["records"][0]["proxy_rm_fingerprint"], "proxy")


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is installed in the cluster environment")
class CPDPOPlotTests(unittest.TestCase):
    def test_three_method_plot_smoke(self) -> None:
        module_path = ROOT / "scripts/aggregate_and_plot_reward_overoptimization.py"
        spec = importlib.util.spec_from_file_location("cpdpo_plot_script", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = []
        for method_index, method in enumerate(("ppo", "pairppo", "cpdpo")):
            for seed in (1, 2, 3):
                for step in (0, 10):
                    records.append(
                        {
                            "method": method,
                            "seed": seed,
                            "rollout_step": step,
                            "proxy_reward_mean": 1.0 + method_index + step / 10,
                            "gold_reward_mean": 2.0 + method_index + step / 20,
                            "eval_kl_mean": float(step),
                            "sqrt_eval_kl": float(step) ** 0.5,
                            "generated_responses": step * 512,
                            "proxy_rm_calls": step * 512,
                            "initial_policy_fingerprint": "initial",
                            "policy_checkpoint_fingerprint": "initial" if step == 0 else f"{method}-{seed}-{step}",
                            "reference_policy_fingerprint": "reference",
                            "proxy_rm_fingerprint": "proxy",
                            "gold_rm_fingerprint": "gold",
                            "evaluation_manifest_sha256": "manifest",
                            "evaluation_prompt_ids_sha256": "eval-prompts",
                            "_prompt_schedule_sha256": f"schedule-{seed}",
                            "_prompt_id_sequence_sha256": f"prompt-ids-{seed}",
                        }
                    )
        aggregated = module.aggregate(records)
        with tempfile.TemporaryDirectory() as directory:
            figures = Path(directory)
            module.plot(aggregated, figures)
            for name in ("figure_2a_reward_vs_training", "figure_2b_reward_vs_kl"):
                self.assertTrue((figures / f"{name}.pdf").is_file())
                self.assertTrue((figures / f"{name}.png").is_file())


if __name__ == "__main__":
    unittest.main()
