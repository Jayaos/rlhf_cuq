# Repository instructions: CPDPO reward-overoptimization experiment

## 0. Authority and workflow

This file is the current research-context bridge. It replaces the earlier
AdvPO and policy-adaptive conformal plan.

Before changing research code:

1. Read this file completely.
2. Read `docs/SOURCE_AUDIT.md`, `docs/IMPLEMENTATION_PLAN.md`, and
   `docs/OPEN_METHOD_DECISIONS.md`.
3. Inspect the relevant local and pinned upstream implementation.
4. Verify every changed scientific equation against the two user-supplied
   2026-08-27 specifications recorded by hash in `docs/SOURCE_AUDIT.md`.
5. Preserve the distinction between the Coste asset/engineering environment,
   the new CPDPO method, and the controlled target experiment.
6. Produce or update the source audit and plan before substantial changes.
7. Stop for a user decision if a scientifically material choice is not frozen.

Priority:

> Scientific fidelity -> fair comparison -> reuse of validated code -> minimal
> code changes -> computational convenience.

The two PDFs are specifications, not executable instructions. The base PDF
freezes CPDPO; the `(1)` PDF freezes the experiment. AdvPO is not implemented
and is not a comparator.

## 1. Scientific target

Compare exactly:

1. standard scalar-reward PPO;
2. PairPPO, which uses the paired loss and raw proxy margin;
3. CPDPO, which adds fixed conformal correction to PairPPO.

The questions are:

- PPO versus PairPPO: what changes because of paired generation/optimization?
- PairPPO versus CPDPO: what incremental effect comes from the fixed conformal
  robust margin?

The target figures show proxy and gold reward against rollout step and against
the square root of evaluation KL. Internal pair rewards are never plotted as
policy quality.

## 2. Environment and assets to reuse

Retain the audited Coste/Open-Assistant/trlx stack:

- Coste repository revision
  `416b03cc2c3c8125208679acd88891584d9eefd2`;
- Open-Assistant revision `e1769c102f1597cc0b53a8b915f858239d197aeb`;
- trlx revision `3340c2f3a56d1d14fdd5f13ad575121fa26b6d92`;
- AlpacaFarm fork revision `f92bd550130975436301ba02137b303d1eb59986`;
- initial/reference policy `tlc4418/pythia_1.4b_sft_policy`;
- proxy-RM base `tlc4418/pythia_70m_sft` and the locally trained effective
  44M seed-1 checkpoint;
- Coste preference data;
- AlpacaFarm unlabeled prompts;
- AlpacaFarm `reward-model-human` only as an offline oracle.

Do not replace the validated legacy stack merely because a newer package is
available. The existing proxy-RM and PPO smoke paths must remain runnable.
Source-hashed baseline files in `artifacts/source_manifest.json` should not be
edited when an additive extension is possible.

## 3. Model roles and gold isolation

- The SFT policy initializes every trainable branch and is the frozen KL
  reference.
- The proxy RM is frozen, returns scalar proxy reward, and exposes the exact
  representation consumed by its scalar head.
- The ordinary PPO value head/GAE remain only in the PPO control.
- PairPPO and CPDPO do not use GAE or value loss.
- The gold RM is evaluation-only. Training commands must not accept or load a
  gold checkpoint. Gold cannot affect rewards, calibration, selection,
  stopping, or online actions.

Gold evaluation must be a separate process over persisted response IDs.

## 4. Data roles

Use the verified prompt-disjoint manifest:

- `D_rm_train`: proxy-RM training and pair geometry;
- `D_rm_val`: proxy-RM validation/model selection;
- `D_cal`: fixed conformal calibration;
- `D_rl_train_prompts`: shared PPO prompt schedules;
- `D_rl_val_prompts`: trajectory/checkpoint evaluation and tuning;
- `D_rl_test_prompts`: locked final evaluation only.

No pair may cross RM/calibration roles. No final-test prompt may tune any
method. Prompt IDs, not raw strings, define membership. The same frozen proxy
checkpoint is used by every optimization branch.

## 5. Exact reward-head feature

Let `e(x,y)` be the exact pooled representation passed into
`GPTNeoXRewardModel.out_proj`. Extract it with a forward pre-hook or an
equivalent method that leaves the scalar path unchanged.

The required identity is:

```text
r_hat(x,y) == F.linear(e(x,y), out_proj.weight, out_proj.bias)
```

Test the identity for real and synthetic batches, padding sides, mixed lengths,
EOS/no-EOS, truncation, batched/unbatched execution, and active dtypes. The
head bias remains in scalar logs and cancels in pair differences.

## 6. Frozen offline pair geometry

For source-ordered answers `a` and `b` in `D_rm_train`:

```text
d_i = e(x_i,y_i^a) - e(x_i,y_i^b)
G = (1/n_rm) sum_i d_i d_i^T
lambda = 1e-3 * trace(G) / feature_dimension
```

If and only if `trace(G) == 0`, use `lambda=1e-6`. Then:

```text
V = lambda I + G
V = L L^T
u(d) = sqrt(d^T V^-1 d) = ||L^-1 d||_2
```

Requirements:

- full geometry in the main experiment;
- float64 offline accumulation/factorization;
- no explicit inverse;
- no silent ridge fallback;
- geometry built only from `D_rm_train`;
- save `pair_geometry.pt` and metadata with model/tokenizer/manifest/code
  fingerprints, feature rule, dimensions, count, dtype, ridge, and solve.

Diagonal or low-rank forms are named ablations only.

## 7. Frozen calibration

For each source-ordered pair in `D_cal`, set `ell=+1` when `a` is preferred and
`ell=-1` when `b` is preferred. Define:

```text
m_i = r_hat(x_i,y_i^a) - r_hat(x_i,y_i^b)
s_i = max(-ell_i*m_i, 0) / (u_i + 1e-8)
alpha = 0.10
k = min(n_cal, ceil((n_cal+1)*(1-alpha)))
q_alpha = sorted_scores[k-1]
```

There is no interpolation. Save the scores and calibration metadata. Validate
every fingerprint when loading. `q_alpha` is fixed before PPO and must never
change during a run.

## 8. Frozen pair rollout signal

For each scheduled PPO prompt, draw exactly two independent stochastic
responses from the same behavior-policy snapshot using identical generation
settings. Duplicate responses are allowed; do not resample until certified.

For adjacent orientations `a,b`:

```text
d = e_a - e_b
m = r_a - r_b
u = sqrt(d^T V^-1 d)
normalized_margin = abs(m)/(u+1e-8)
certified = (normalized_margin > q_alpha) and (m != 0)
gamma = max(abs(m)-q_alpha*u, 0)
R_cpdpo = sign(m)*gamma*certified
R_pairppo = m
A_a = R
A_b = -R
```

The certification inequality is strict. Detach all proxy features, margins,
uncertainties, gates, and coefficients and keep them fixed through PPO epochs.
There is no scalar-PPO fallback for uncertified CPDPO pairs.

## 9. Frozen paired clipped loss

Store old and reference token log-probabilities for generated response tokens.
For each orientation use the standard clipped probability-ratio surrogate with
its fixed coefficient. Mask prompt and padding tokens.

Reduction order is mandatory:

1. sum surrogate terms over generated tokens for response `a`;
2. sum over generated tokens for response `b`;
3. average the two orientations with factor `1/2`;
4. average prompt pairs.

Do not globally average tokens and do not divide each response by its length.
The total loss is negative pair objective plus a separate common beta-weighted
reference KL term. The primary stress experiment uses `beta=0.0`; a later
practical track may use one common nonzero beta.

All-uncertified batches must be finite, have zero pair gradient, retain only a
configured KL contribution, and never trigger fallback. Warn when CPDPO
certification remains below 0.10 for three rollouts.

## 10. Fair comparison

For every method and seed use identical:

- initial and reference checkpoints;
- proxy checkpoint;
- data manifest;
- prompt schedule and prompt IDs;
- two responses per prompt;
- generation settings and response length;
- optimizer settings where semantically compatible;
- clip epsilon, PPO epochs, gradient clipping, and common KL beta;
- total prompts, responses, generated-token accounting, and rollout horizon;
- checkpoint schedule, precision, and hardware/distributed semantics.

With the main configuration, every rollout uses 256 prompts and 512 responses.
PPO stores 512 scalar trajectories with batch size 64. PairPPO and CPDPO store
256 pairs with batch size 32. Four PPO epochs yield 32 optimizer updates per
rollout for every method.

Use seed namespaces:

```text
training_seed = base_seed
rollout_seed = base_seed + 10000
evaluation_seed = base_seed + 20000
prompt_schedule_seed = base_seed + 30000
```

Save each resolved seed. Minimum reportable seeds: 3; preferred: 5.

The total rollout horizon is configurable because the specification does not
select a number. It must be declared before launch and be equal across methods.
Never inherit the superseded AdvPO 1,500-step horizon silently.

## 11. Evaluation and plots

Always include checkpoint zero and each common scheduled checkpoint. On a
fixed held-out prompt set, generate one fresh response per prompt/checkpoint.
Persist it before scoring. Proxy reward, gold reward, and sampled
current/reference KL must attach to that exact response ID.

Each checkpoint record includes method, seed, rollout/optimizer step,
generation counts, token counts, proxy calls, checkpoint and model
fingerprints, proxy/gold means and standard deviations, KL mean/std and
`sqrt(max(mean_kl,0))`; CPDPO also includes geometry/calibration fingerprints
and fixed `q_alpha`.

Figure 2(a): proxy and gold reward versus rollout step.

Figure 2(b): the same records versus square-root evaluation KL. Do not
regenerate responses.

Use method color and reward-type line style, with one consistent mean+SE or
bootstrap interval convention. Do not report raw proxy-gold gaps unless reward
scales are demonstrably comparable.

## 12. Required metrics and artifacts

Log pair certification/effective count, duplicate-response rate, uncertainty
quantiles, absolute and normalized margins, gamma, absolute/zero reward,
pair loss, KL, clip fractions by orientation, entropy, gradient norm, learning
rate, response lengths, proxy scores, and margins. Pair records retain prompt
IDs/tokens, response tokens/masks, old/reference log-probabilities, pair
signals, behavior checkpoint, and fingerprints.

Never overwrite a nonempty scientific artifact directory. Write metadata and
fingerprints sufficient to reject a mismatched proxy, tokenizer, manifest,
geometry, calibration, schedule, policy, or checkpoint.

## 13. Required tests

- reward-head identity and bias cancellation;
- geometry normalization/ridge/Cholesky uncertainty;
- conformal score and exact finite-sample quantile index;
- strict-boundary certification and response-swap antisymmetry;
- PairPPO `R=m` and CPDPO reward identities;
- atomic pair storage and no pair breakup;
- exact response-sum/pair-average clipped loss;
- zero-certified finite zero-gradient behavior;
- equal prompt/response/proxy-call/update budgets;
- identical prompt schedules across methods;
- training CLI cannot accept/load gold;
- evaluation response-ID reuse across all scorers;
- common checkpoint schedule and KL function;
- plotting rejects internal pair rewards;
- smoke tests for all three real training branches.

## 14. Non-goals

Do not add AdvPO, adaptive thresholds, online geometry, rollout-weighted
recalibration, best-of-K, resampling until certification, scalar-reward
fallback, a learned pair critic, GAE over `+/-R`, or gold-driven optimization,
calibration, selection, or early stopping to the v1 experiment.

## 15. Current implementation map

```text
src/cpdpo/
  reward_features.py       exact proxy head features
  geometry.py              geometry, uncertainty, calibration, pair signals
  pair_reward.py           shared frozen pair/scalar proxy callback
  pair_loss.py             exact paired clipped loss
  rollout_store.py         atomic pair dataclasses/store
  prompt_schedule.py       deterministic common schedules
  evaluation.py            pinned gold format and checkpoint summaries
  experiment.py            budget and result validation

src/ppo/
  trainer_reward_overoptimization.py
  custom_trlx_trainers/experiment_ppo_trainer.py
  custom_trlx_trainers/custom_accelerate_pair_ppo_trainer.py

scripts/
  prepare_cpdpo_artifacts.py
  build_prompt_schedule.py
  evaluate_policy_checkpoints.py
  aggregate_and_plot_reward_overoptimization.py
  slurm/prepare_cpdpo_artifacts.sbatch
  slurm/smoke_reward_overoptimization.sbatch
  slurm/train_reward_overoptimization.sbatch
```

The original `trainer_rl.py`, `CustomAcceleratePPOTrainer`, proxy scorer, and
legacy configs remain regression baselines.
