from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.reward_modeling.training.final_evaluation import (
    FINAL_EVAL_PREFIX,
    evaluate_and_save_final_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeTrainer:
    def __init__(self, metrics: object, eval_dataset: object = None) -> None:
        self.metrics = metrics
        self.eval_dataset = (
            {"D_rm_val": object()} if eval_dataset is None else eval_dataset
        )
        self.evaluate_calls: list[tuple[object, str]] = []
        self.logged: list[tuple[str, object]] = []
        self.saved: list[tuple[str, object]] = []

    def evaluate(self, eval_dataset: object = None, metric_key_prefix: str = "eval") -> object:
        self.evaluate_calls.append((eval_dataset, metric_key_prefix))
        return self.metrics

    def log_metrics(self, split: str, metrics: object) -> None:
        self.logged.append((split, metrics))

    def save_metrics(self, split: str, metrics: object) -> None:
        self.saved.append((split, metrics))


class FinalEvaluationTests(unittest.TestCase):
    def test_manifest_trainer_runs_final_evaluation_after_training(self) -> None:
        wrapper = (
            ROOT / "src/reward_modeling/training/trainer_rm_manifest.py"
        ).read_text(encoding="utf-8")
        train_call = "train_output = super().train(*args, **kwargs)"
        final_eval_call = "evaluate_and_save_final_metrics(self)"
        self.assertIn("class ManifestRMTrainer(legacy_trainer.RMTrainer):", wrapper)
        self.assertLess(wrapper.index(train_call), wrapper.index(final_eval_call))
        self.assertIn("legacy_trainer.RMTrainer = ManifestRMTrainer", wrapper)

    def test_final_accuracy_is_evaluated_logged_and_saved(self) -> None:
        metrics = {
            "final_eval_D_rm_val_loss": 0.5,
            "final_eval_D_rm_val_accuracy": 0.625,
        }
        trainer = FakeTrainer(metrics)

        result = evaluate_and_save_final_metrics(trainer)

        self.assertEqual(result, metrics)
        self.assertEqual(
            trainer.evaluate_calls,
            [(trainer.eval_dataset["D_rm_val"], f"{FINAL_EVAL_PREFIX}_D_rm_val")],
        )
        self.assertEqual(trainer.logged, [(FINAL_EVAL_PREFIX, metrics)])
        self.assertEqual(trainer.saved, [(FINAL_EVAL_PREFIX, metrics)])

    def test_unnamed_validation_dataset_uses_plain_final_prefix(self) -> None:
        metrics = {"final_eval_accuracy": 0.625}
        dataset = object()
        trainer = FakeTrainer(metrics, eval_dataset=dataset)

        evaluate_and_save_final_metrics(trainer)

        self.assertEqual(
            trainer.evaluate_calls,
            [(None, FINAL_EVAL_PREFIX)],
        )

    def test_missing_or_nonfinite_accuracy_fails_before_persisting(self) -> None:
        for metrics in (
            {"final_eval_D_rm_val_loss": 0.5},
            {"final_eval_D_rm_val_accuracy": math.nan},
        ):
            with self.subTest(metrics=metrics):
                trainer = FakeTrainer(metrics)
                with self.assertRaises(RuntimeError):
                    evaluate_and_save_final_metrics(trainer)
                self.assertEqual(trainer.logged, [])
                self.assertEqual(trainer.saved, [])


if __name__ == "__main__":
    unittest.main()
