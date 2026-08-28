#!/usr/bin/env python
"""Materialize a shared prompt schedule for all three training methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cpdpo.prompt_schedule import materialize_prompt_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--rollout-steps", required=True, type=int)
    parser.add_argument("--prompts-per-rollout", type=int, default=256)
    args = parser.parse_args()
    metadata = materialize_prompt_schedule(
        manifest_path=args.manifest,
        output_path=args.output,
        base_seed=args.base_seed,
        rollout_steps=args.rollout_steps,
        prompts_per_rollout=args.prompts_per_rollout,
    )
    print(f"PASS prompt schedule: {args.output}")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
