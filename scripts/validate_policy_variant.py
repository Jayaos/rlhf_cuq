#!/usr/bin/env python
"""Fail fast unless a local policy matches its declared capacity variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ppo.policy_variants import POLICY_VARIANTS, validate_policy_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(POLICY_VARIANTS), required=True)
    parser.add_argument("--policy-model", required=True)
    args = parser.parse_args()
    metadata = validate_policy_checkpoint(args.policy_model, args.variant)
    print(f"PASS policy variant: {json.dumps(metadata, sort_keys=True)}")


if __name__ == "__main__":
    main()
