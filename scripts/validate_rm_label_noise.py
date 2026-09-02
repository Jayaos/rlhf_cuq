#!/usr/bin/env python3
"""Validate an opt-in noisy proxy-RM checkpoint and its exact flip audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reward_modeling.training.label_noise import (  # noqa: E402
    validate_persisted_label_noise_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    metadata = validate_persisted_label_noise_provenance(
        args.model_dir,
        expected_rate=args.rate,
        expected_seed=args.seed,
    )
    print(
        "PASS noisy proxy RM",
        json.dumps(
            {
                "model_dir": str(Path(args.model_dir).resolve()),
                "rate": metadata["requested_rate"],
                "realized_rate": metadata["realized_rate"],
                "seed": metadata["seed"],
                "flips": metadata["flip_count"],
                "train_records": metadata["train_record_count"],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
