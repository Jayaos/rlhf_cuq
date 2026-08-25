"""Run the legacy reward-model trainer with the explicit split manifest adapter."""

from __future__ import annotations

import sys
from pathlib import Path


# ``accelerate launch path/to/script.py`` exposes the script directory, not the
# repository root, on sys.path.  Add the root before importing the ``src``
# namespace so the documented path-based cluster command works from any cwd.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils.manifest_dataset_loader import get_manifest_dataset  # noqa: E402
from src.reward_modeling.training import trainer_rm as legacy_trainer  # noqa: E402


def main() -> None:
    """Delegate all model/training behavior to Coste while replacing data loading."""

    legacy_trainer.get_dataset = get_manifest_dataset
    legacy_trainer.main()


if __name__ == "__main__":
    main()
