"""Final validation reporting for manifest-backed reward-model training."""

from __future__ import annotations

import json
import math
from typing import Any


FINAL_EVAL_PREFIX = "final_eval"


def evaluate_and_save_final_metrics(trainer: Any) -> dict[str, float]:
    """Evaluate the exact final in-memory model and persist its metrics.

    The legacy Coste trainer evaluates periodically, but does not explicitly
    evaluate after its last optimizer update.  This additive helper does not
    update model or optimizer state.  With the manifest adapter's evaluation
    mapping, the accuracy key is ``final_eval_D_rm_val_accuracy``.
    """

    eval_dataset = trainer.eval_dataset
    if isinstance(eval_dataset, dict):
        if not eval_dataset:
            raise RuntimeError("Final RM evaluation has no validation datasets")
        metrics: dict[str, float] = {}
        for dataset_name, dataset in eval_dataset.items():
            dataset_metrics = trainer.evaluate(
                eval_dataset=dataset,
                metric_key_prefix=f"{FINAL_EVAL_PREFIX}_{dataset_name}",
            )
            if not isinstance(dataset_metrics, dict):
                raise RuntimeError(
                    f"Final RM evaluation for {dataset_name!r} did not return "
                    "a metric mapping"
                )
            duplicate_keys = metrics.keys() & dataset_metrics.keys()
            if duplicate_keys:
                raise RuntimeError(
                    "Final RM evaluation returned duplicate metric keys: "
                    f"{sorted(duplicate_keys)}"
                )
            metrics.update(dataset_metrics)
    else:
        metrics = trainer.evaluate(metric_key_prefix=FINAL_EVAL_PREFIX)
        if not isinstance(metrics, dict):
            raise RuntimeError("Final RM evaluation did not return a metric mapping")

    accuracy_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if key.endswith("_accuracy")
    }
    if not accuracy_metrics:
        raise RuntimeError("Final RM evaluation did not report an accuracy metric")
    if not all(math.isfinite(value) for value in accuracy_metrics.values()):
        raise RuntimeError(
            f"Final RM evaluation reported non-finite accuracy: {accuracy_metrics}"
        )

    trainer.log_metrics(FINAL_EVAL_PREFIX, metrics)
    trainer.save_metrics(FINAL_EVAL_PREFIX, metrics)
    print(f"PASS final RM validation {json.dumps(accuracy_metrics, sort_keys=True)}")
    return metrics
