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

The frozen main comparison continues to use `alpha=0.10`. Alternative alpha
values are permitted only as an explicitly labelled CPDPO sensitivity
ablation: they must produce a separate calibration artifact and CPDPO run
directory, keep one threshold fixed for the whole run, reuse the unchanged
PPO/PairPPO controls, and never use gold reward to choose the value.

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
- Provide a deterministic 64-pair/64-pair smoke-only path for fast integration
  checks. Tag every reduced artifact and reject it from scientific training by
  default.

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

## 7. Add isolated checkpoint evaluation and figures — done (licensed base external)

- Evaluate checkpoint zero and each common scheduled checkpoint on fixed
  held-out prompts.
- Generate each evaluation response once and persist it before scoring.
- Proxy-score, gold-score, and compute sampled reference KL on those exact IDs
  in a separate command.
- Aggregate per-seed records and produce reward-vs-rollout and reward-vs-sqrt-KL
  PDF/PNG figures with a consistent uncertainty convention.
- Support an explicitly labelled single-seed diagnostic plot with no
  across-seed uncertainty band.  This does not relax the minimum three-seed
  requirement in the reportable aggregation path.
- Recover the AlpacaFarm SFT10k and human reward model from the exact local,
  manifest-pinned differences with fatal `model_sum.txt` checks.  The helper
  and Slurm job are repository-managed; authorization and acquisition of the
  original LLaMA-7B base remain external prerequisites.
- Reuse only fully validated, persisted per-checkpoint response files when an
  offline scorer fails after generation.  Resume must never regenerate or
  overwrite an accepted response file, and final scored/metric artifacts
  remain exclusive outputs.
- Select a named CPDPO alpha-ablation run while reusing the identical PPO and
  PairPPO control records. Record alpha in calibration, run, checkpoint, and
  evaluation provenance and identify it in plot legends.

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

## 9. Add reference-anchored CPDPOv2 — done (cluster smoke pending)

- Preserve all v1 names, artifacts, trainers, launch arrays, and plot defaults.
- Cache one frozen-SFT response plus proxy reward/head feature per unique
  scheduled prompt with complete policy/proxy/schedule provenance.
- Reuse and validate the full v1 geometry and fixed calibration threshold.
- Score both current trajectories with `m_ref - q_alpha*u_ref`.
- Optimize only current trajectories through ordinary scalar PPO/GAE.
- Keep online response/proxy-call/update budgets equal to scalar PPO and record
  reference preparation cost separately.
- Add separate smoke/full/evaluation jobs and optional four-method plotting.
- Add one optional two-method Slurm evaluation array for CPDPOv2 and AdvPO;
  preserve the dedicated evaluation jobs and identical evaluator semantics.
- Test the reward identity, common-mode sensitivity, cache provenance,
  no-gold access, budget equality, and v1 regression behavior.
- Canonicalize cached prompts through the same policy-tokenizer decode boundary
  used by TRLX reward callbacks; reject pre-1.1 caches built from raw schedule
  text before training begins.

## 10. Add paper-equation AdvPO as a separate comparison — implemented locally; cluster smoke pending

- Preserve PPO, PairPPO, CPDPO, CPDPOv2, their artifacts, and their launch
  defaults unchanged.
- Build the AdvPO `M_D` from the unnormalised sum of individual chosen and
  rejected `D_rm_train` feature outer products, with a separately declared
  ridge and float64 Cholesky factorization.
- Cache one frozen-SFT reference response per scheduled prompt without loading
  CPDPO geometry/calibration or gold reward.
- Compute one shared adversarial reward-head direction per 64-response PPO
  batch from the current/reference mean feature difference and the selected
  `B=b^2`.
- Reuse ordinary scalar PPO/GAE, the existing initial/reference policy, common
  prompt schedule, response/update budget, checkpointing, and gold-isolated
  evaluator.
- Add isolated AdvPO preparation, smoke, full-training, evaluation, and
  optional plotting commands. Require named run directories for every `B`.
- Test the confidence-matrix construction, closed-form max-min identity,
  shared-direction behavior, zero-difference boundary, artifact provenance,
  equal budget, and absence of gold access.

This is an exact implementation of the disclosed AdvPO equations in the
current Coste/Pythia experimental pipeline, not an exact reproduction of the
authors' unpublished code or their Section 5.2 LLaMA/data configuration.
The dependency-free suite and Python compilation pass locally. The pinned
PyTorch/TRLX mathematical tests and real one-rollout job remain gated on
`scripts/slurm/smoke_advpo.sbatch` in the cluster environment.

## 11. Add matched-capacity 1.4B proxy-RM ablation — implemented locally; cluster smoke pending

- Preserve the existing 44M proxy configuration, checkpoints, artifacts, and
  experiment outputs unchanged.
- Branch `assets/initial_sft_policy` through the existing manifest-backed
  `GPTNeoXRewardModel` trainer and train only on `D_rm_train`, with final model
  selection/validation on `D_rm_val`.
- Retain the Coste RM objective, learning rate, five epochs, and effective
  batch 32; use gradient checkpointing and smaller micro/evaluation batches as
  memory-only adaptations for the 1.4B model.
- Write checkpoints and checksums to a separately named scratch-backed output
  by default, with the same strict non-overwrite and latest-checkpoint resume
  validation as the 44M job.
- Make offline artifact preparation, online proxy scoring, and evaluation
  proxy batch sizes runtime-configurable so the larger frozen RM can use the
  existing fingerprinted pipeline.
- Add a real one-GPU 1.4B RM smoke, static launch/config tests, and a runbook
  that requires fresh capacity-specific artifacts and outputs for all methods.

## 12. Add selectable 70M/1.4B policy capacity — primary FP32 cluster smoke passed

- Preserve `assets/initial_sft_policy` and all existing output locations as the
  default `1p4b` policy track.
- Add a named `70m` option that uses the complete causal-LM checkpoint at
  `assets/proxy_rm_sft_base` for both initialization and the frozen KL
  reference; never treat the scalar 44M RM checkpoint as a generator.
- Validate the declared option against the local GPT-NeoX causal-LM
  architecture before model loading and record the variant/dimensions in run,
  checkpoint, and evaluation provenance.
- Use distinct 70M policy output roots and rebuild policy-generated CPDPOv2 and
  AdvPO reference caches. Reuse policy-independent CPDPO/AdvPO geometry only
  when its existing proxy/data/code fingerprint validation succeeds.
- Thread the option through main/smoke training, offline evaluation, additive
  methods, and plotting while retaining raw path overrides as an advanced,
  architecture-validated escape hatch.
- Add dependency-free unit/static tests and exact cluster commands for both
  policy variants.
- Expose and persist an explicit policy optimization profile (learning rate,
  unfrozen layer count, and gradient-norm limit) after the literal 1.4B profile
  diverged on the first real 70M PPO rollout. Keep the 1.4B/default profile
  unchanged, require a distinct output root for overrides, and fail at the
  first non-finite loss or gradient rather than advancing a corrupted run.
- Add a declared FP32 70M launcher after the conservative BF16 smoke produced
  a non-finite gradient norm before optimizer step one. Validate the declared
  precision against Accelerate at runtime and apply it uniformly to every
  compared method; retain BF16 as the unchanged 1.4B default.
- Record the successful FP32 PPO/PairPPO/CPDPO smoke and thread the same
  declared dtype into policy-generated CPDPOv2/AdvPO reference caches. Keep
  optimizer-only arguments out of the cache builders and require separate
  CPDPOv2/AdvPO smoke passes before their full 70M runs.
- Reconstruct evaluation-time TRLX Hydra wrappers with the unfrozen-layer
  count recorded by each run instead of the historical hardcoded value. Reject
  inconsistent/out-of-range metadata and retain the old value `2` only for
  checkpoints created before either provenance field existed.
- Right-size every offline evaluation launcher to 4 CPUs, 24 GiB host memory,
  and 3 hours using completed Phoenix MaxRSS/elapsed evidence. Do not infer a
  corresponding reduction for training or artifact jobs from evaluation data.
