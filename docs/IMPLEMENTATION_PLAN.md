# CPDPO implementation plan

Status labels: `done`, `in progress`, `pending`, `blocked`.

## 1. Preserve and reuse the validated baseline — done

- Keep hash-pinned Coste scoring and PPO files unchanged.
- Reuse the working manifest loader, local model overrides, Accelerate launch,
  policy/reference model construction, optimizer, scheduler, and checkpoint
  writer.
- Retain the existing proxy-RM and PPO smoke jobs as regression tests.

## 2. Freeze the revised scientific contract — done

- Record both PDF hashes and distinguish method from experiment requirements.
- Remove AdvPO and adaptive-threshold work from the active plan.
- Freeze full pair-difference geometry, alpha 0.10 calibration, strict
  certification, PairPPO margin reward, CPDPO robust-margin reward, and the
  pairwise clipped loss.
- Leave only the total rollout horizon as an explicit full-run parameter.

## 3. Implement transparent CPDPO numerical primitives — done

- Reward-head feature hook and identity check.
- Pair formatting/orientation helpers.
- Float64 Gram accumulation, specified ridge, Cholesky factor, and uncertainty
  solve.
- Finite-sample conformal score/quantile.
- PairPPO and CPDPO signal computation.
- Pairwise clipped surrogate with the exact response-sum/pair-average
  reduction and separate KL term.

## 4. Build immutable offline artifacts — done (cluster integration pending)

- Add a command that reads `D_rm_train` and `D_cal` from the verified split
  manifest and the trained proxy RM.
- Save `pair_geometry.pt`, `pair_geometry_metadata.json`,
  `calibration_scores.pt`, and `conformal_calibration.json` atomically.
- Store and validate model, tokenizer, data-manifest, source-role, extraction,
  dtype, code-revision, and artifact fingerprints.
- Refuse overwrite unless explicitly requested; verify existing artifacts on
  resume.

## 5. Add fair experiment trainers — done (cluster smoke pending)

- Add pair-aware dataclasses, collator, and rollout store.
- Add a CPDPO/PairPPO trainer that samples exactly two independent responses
  for each scheduled prompt and stores frozen old/reference log-probabilities
  and pair signals.
- Persist one atomic, fingerprinted pair-record snapshot per rollout.
- Disable GAE/value loss in pair methods.
- Add a fair-budget PPO trainer that consumes the same prompt schedule and
  2B responses while preserving standard scalar PPO.
- Register both as additive trainers; do not alter the existing smoke-tested
  `CustomAcceleratePPOTrainer`.

## 6. Add schedule, launch, and checkpoint metadata — done

- Materialize deterministic per-seed prompt schedules with the specified seed
  namespaces.
- Add separate training entry points/configs for PPO, PairPPO, and CPDPO. None
  accepts a gold checkpoint.
- Record method, resolved seeds, rollout/optimizer counters, generated
  responses/tokens, proxy calls, and all required fingerprints.
- Save rollout-boundary optimizer/scheduler/RNG state and validate immutable
  context on resume; archive and rewind only uncheckpointed rollout metrics.
- Provide one-rollout Slurm smoke jobs and a full three-method array/job
  template that requires an explicit rollout horizon.

## 7. Add isolated checkpoint evaluation and figures — done (gold asset external)

- Evaluate checkpoint zero and each common scheduled checkpoint on fixed
  held-out prompts.
- Generate each evaluation response once and persist it before scoring.
- Proxy-score, gold-score, and compute sampled reference KL on those exact IDs
  in a separate command.
- Aggregate per-seed records and produce reward-vs-rollout and reward-vs-sqrt-KL
  PDF/PNG figures with a consistent uncertainty convention.

## 8. Verification — in progress

- Unit tests cover frozen equations, feature identity, paired reduction,
  zero-certified gradients, atomic pair collation, schedule/budget invariants,
  no-gold CLI exposure, pinned gold formatting, and result validation.
- Fingerprint mismatch checks are implemented in artifact consumers.
- The complete standard-library suite passes locally; torch tests are skipped
  locally because Torch is intentionally absent from the Windows control
  environment and must run in the pinned cluster environment.
- Provide cluster commands for the torch/model integration tests and the new
  one-rollout PPO/PairPPO/CPDPO smoke matrix.

## Completion boundary

Implementation is complete when all three training variants can use the same
materialized schedule, pair artifacts validate against the proxy checkpoint,
the smoke matrix finishes without gold access, the separate evaluator emits
common checkpoint records, and both plots can be rebuilt without rerunning
training.
