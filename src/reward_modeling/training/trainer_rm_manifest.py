"""Run the legacy reward-model trainer with the explicit split manifest adapter."""

from src.data_utils.manifest_dataset_loader import get_manifest_dataset
from src.reward_modeling.training import trainer_rm as legacy_trainer


def main() -> None:
    """Delegate all model/training behavior to Coste while replacing data loading."""

    legacy_trainer.get_dataset = get_manifest_dataset
    legacy_trainer.main()


if __name__ == "__main__":
    main()
