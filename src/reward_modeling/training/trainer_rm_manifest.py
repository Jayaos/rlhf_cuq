"""Run the legacy reward-model trainer with the explicit split manifest adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


# ``accelerate launch path/to/script.py`` exposes the script directory, not the
# repository root, on sys.path.  Add the root before importing the ``src``
# namespace so the documented path-based cluster command works from any cwd.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils.manifest_dataset_loader import get_manifest_dataset  # noqa: E402
from src.reward_modeling.training.final_evaluation import (  # noqa: E402
    evaluate_and_save_final_metrics,
)
from src.reward_modeling.training.local_model_compat import (  # noqa: E402
    apply_local_model_family,
)
from src.reward_modeling.training import trainer_rm as legacy_trainer  # noqa: E402


def _with_local_model_family(parser: Callable[[], Any]) -> Callable[[], Any]:
    """Apply validated local-checkpoint and memory-safe scoring overrides."""

    def parse() -> Any:
        conf = apply_local_model_family(parser())
        normalization_batch_size = getattr(
            conf, "reward_normalization_batch_size", None
        )
        if normalization_batch_size is not None:
            if (
                not isinstance(normalization_batch_size, int)
                or isinstance(normalization_batch_size, bool)
                or normalization_batch_size < 1
            ):
                raise ValueError(
                    "reward_normalization_batch_size must be a positive integer"
                )
            uncapped_get_reward = legacy_trainer.get_reward

            def memory_safe_get_reward(
                samples: Any,
                model: Any,
                tokenizer: Any,
                device: Any,
                batch_size: int = 128,
            ) -> Any:
                return uncapped_get_reward(
                    samples,
                    model,
                    tokenizer,
                    device,
                    batch_size=min(batch_size, normalization_batch_size),
                )

            legacy_trainer.get_reward = memory_safe_get_reward
            print(
                "RM reward-normalization batch-size cap:",
                normalization_batch_size,
            )
        return conf

    return parse


class ManifestRMTrainer(legacy_trainer.RMTrainer):
    """Add exact post-update validation without changing Coste training."""

    def train(self, *args: Any, **kwargs: Any) -> Any:
        train_output = super().train(*args, **kwargs)
        evaluate_and_save_final_metrics(self)
        return train_output


def main() -> None:
    """Delegate all model/training behavior to Coste while replacing data loading."""

    legacy_trainer.get_dataset = get_manifest_dataset
    legacy_trainer.RMTrainer = ManifestRMTrainer
    legacy_trainer.argument_parsing = _with_local_model_family(
        legacy_trainer.argument_parsing
    )
    legacy_trainer.main()


if __name__ == "__main__":
    main()
