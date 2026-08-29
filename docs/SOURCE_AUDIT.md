# Source audit: CPDPO reward-overoptimization experiment

Audit date: 2026-08-27

This document records the sources checked before implementing the revised
experiment. The two user-supplied PDFs are method and experiment
specifications. They are treated as research requirements, not as executable
instructions. They supersede the repository's earlier AdvPO plan.

## Authoritative specifications

| Source | Pages | SHA-256 | Role |
|---|---:|---|---|
| `Codex Implementation Specification.pdf` | 25 | `88d49a686db26d70967a3dcff95c31b50a340bb24fa535021463b396e0348eb7` | Frozen CPDPO method |
| `Codex Implementation Specification (1).pdf` | 32 | `f88fc486fd23dd0b819d983a92c5c710999d9c85428a35061cf77df2fde1a800` | Frozen PPO/PairPPO/CPDPO experiment |

The PDFs are complementary. The first defines the offline pair geometry,
finite-sample calibration rule, rollout signal, and pairwise clipped loss. The
second defines the controlled three-method comparison, equal generation
budget, checkpoint evaluation, aggregation, and plots.

No v1 equation required for the online CPDPO reward path remains open. The
total number of rollout steps is intentionally a run parameter: the experiment
specification fixes equal horizons and checkpoint cadence, but does not select
one numerical horizon. Full-run commands must therefore supply it explicitly;
smoke configurations use one rollout only.

## Components that remain unchanged

The following audited Coste/Open-Assistant/trlx environment is retained:

| Component | Immutable revision | Use in this project |
|---|---|---|
| `tlc4418/llm_optimization` | `416b03cc2c3c8125208679acd88891584d9eefd2` | Coste data/RM/PPO baseline |
| `LAION-AI/Open-Assistant` | `e1769c102f1597cc0b53a8b915f858239d197aeb` | `GPTNeoXRewardModel` and model-training support |
| `CarperAI/trlx` | `3340c2f3a56d1d14fdd5f13ad575121fa26b6d92` | Existing PPO model, reference policy, optimizer, launch, and checkpoint machinery |
| `tlc4418/alpaca_farm` | `f92bd550130975436301ba02137b303d1eb59986` | AlpacaFarm integration and offline gold evaluator |

The complete asset revisions and file hashes remain in
`artifacts/source_manifest.json`.

The pinned AlpacaFarm `recover_model_weights.py` was also inspected before
adding the local recovery path.  Its reconstruction arithmetic loads every
weight difference in float32, adds the corresponding float32 LLaMA parameter
in place (skipping only declared shape mismatches), and validates the sum of
all reconstructed parameters with `numpy.isclose` against `model_sum.txt`.
The upstream CLI resolves `tatsu-lab/alpaca-farm-*-wdiff` without a revision
and does not accept a local difference directory.  The additive
`scripts/reconstruct_alpaca_farm_gold_rm.py` wrapper therefore preserves that
arithmetic while consuming only the two manifest-pinned, locally verified
difference directories.  It treats either integrity-check failure as fatal,
records source and output fingerprints, and never downloads or redistributes
the separately licensed LLaMA base.

The existing smoke tests established that:

- manifest-backed proxy-RM training runs and saves a valid checkpoint;
- the single-RM Coste PPO path completes one update with finite proxy reward
  and KL values;
- the pinned legacy packages resolve from their expected revisions.

CPDPO is therefore an additive extension. The source-hashed baseline files in
`artifacts/source_manifest.json` are not edited. Ordinary historical Coste PPO
remains available for regression checks.

## Verified reward-model feature seam

At the pinned Open-Assistant revision, `GPTNeoXRewardModel` computes:

```text
hidden states -> configured mean/last pooling -> out_proj: Linear(hidden_size, 1)
```

The exact CPDPO feature is the input to `out_proj`. A forward pre-hook on that
module retrieves the representation without changing the scalar reward path.
For every batch the adapter must verify

```text
model logits == F.linear(feature, out_proj.weight, out_proj.bias)
```

within a dtype-appropriate tolerance. The head bias is retained in individual
reward logs and cancels in pair differences. Static `D_rm_train`/`D_cal`
answers are terminated with exactly one Pythia `<|endoftext|>` token, matching
the proxy-RM training formatter. Online rollouts retain the existing Coste
rule, which appends that token only when generation ended or was trimmed at
EOS; offline proxy evaluation uses the legacy evaluation convention with one
terminal EOS.

## Verified PPO extension seam

The existing path is:

```text
trainer_rl.py
  -> trlx.train(...)
  -> CustomAcceleratePPOTrainer
  -> PromptPipeline
  -> make_experience()
  -> PPORolloutStorage
  -> CustomAccelerateRLTrainer.learn()
```

`CustomAcceleratePPOTrainer` already supplies policy generation, frozen
reference-policy log probabilities, Accelerate preparation, optimizer and
scheduler handling, checkpoint writes, and checkpoint-zero evaluation.

The pinned `PPORLElement` cannot represent a pair or retain reference
log-probabilities. Its default loss also computes GAE and a value loss and uses
a token-mean reduction. Those semantics are incompatible with the method
specification. The correct minimal extension is therefore:

- a new pair element/batch/store that never separates the two responses;
- a new trainer subclass that reuses the working model/launch/checkpoint path;
- a CPDPO/PairPPO loss that ignores the value head, preserves stored old and
  reference log-probabilities, and performs sum-then-pair-average reduction;
- a separate fair-budget PPO subclass/config that receives the same duplicated
  prompt schedule and 2B response budget while retaining ordinary PPO/GAE.

## Frozen equations implemented

For source-ordered response pairs,

```text
d_i = e(x_i, y_i^a) - e(x_i, y_i^b)
G   = (1 / n_rm) sum_i d_i d_i^T
lambda = 1e-3 * trace(G) / dim       (1e-6 only when trace(G) == 0)
V   = lambda I + G
u(d) = sqrt(d^T V^-1 d) = ||L^-1 d||_2,  V = L L^T
```

For calibration label `ell_i` (`+1` when `a` is preferred, `-1` when `b` is
preferred),

```text
m_i = r_i^a - r_i^b
s_i = max(-ell_i * m_i, 0) / (u_i + 1e-8)
k   = min(n_cal, ceil((n_cal + 1) * (1 - alpha)))
q_alpha = sorted_scores[k - 1], alpha = 0.10
```

The threshold is fixed for the complete PPO run.

For a PPO-time pair,

```text
certified = (abs(m) / (u + 1e-8) > q_alpha) and (m != 0)
gamma     = max(abs(m) - q_alpha * u, 0)
R_cpdpo   = sign(m) * gamma * certified
R_pairppo = m
A_a = R,  A_b = -R
```

The pair objective uses clipped token probability ratios, sums over generated
tokens for each response, averages the two orientations, and then averages
prompt pairs. It does not use GAE, a value loss, per-response length
normalization, global-token averaging, online threshold updates, fallback
scalar rewards, or gold feedback.

## Controlled experiment contract

The reportable comparison is exactly:

1. standard scalar-reward PPO;
2. PairPPO with `R=m`;
3. CPDPO with the fixed conformal robust margin.

Every method uses the same initial policy, reference policy, proxy RM,
manifest, seed-derived prompt schedule, optimizer settings where semantically
compatible, B prompts per rollout, and two independently sampled responses per
prompt. PPO flattens the resulting 2B trajectories; pair methods keep B atomic
pairs.

Gold reward is unavailable to every training entry point. A separate evaluator
generates one response for every fixed held-out prompt/checkpoint, saves that
response once, then scores the same response with proxy RM, gold RM, and the
common sampled-KL evaluator. Figure 2(a) uses proxy/gold reward against rollout
step. Figure 2(b) reuses those checkpoint records against `sqrt(max(mean_kl,0))`.
If a scorer fails after generation, the evaluator may resume only from complete
per-checkpoint response files that validate against the method, seed, manifest
prompt order, checkpoint path/fingerprint, counters, and deterministic response
ID.  It never overwrites or silently regenerates an accepted response file.
At the user's request, seed 1 may also be rendered as an explicitly labelled
diagnostic without an uncertainty band.  That view uses the same validated
checkpoint records and square-root evaluation KL but is not a multi-seed
estimate and does not change the reportable minimum of three seeds.

The user subsequently requested runtime-selectable alpha sensitivity runs.
This does not change the specification-defined main value `alpha=0.10` or any
equation above. A non-default value is recorded as a CPDPO alpha ablation,
produces a distinct fixed `q_alpha` calibration artifact and run directory,
and reuses the unchanged PPO and PairPPO control records. Calibration/run/
evaluation fingerprints reject an alpha mismatch, and gold reward remains
unavailable to calibration and training and may not be used to select alpha.

## Deliberate adaptation versus original assets

This is not AdvPO and no AdvPO confidence matrix, reference-response robust
objective, or B-grid is part of the revised experiment. Coste supplies the
assets and the validated engineering environment only. CPDPO's pair geometry
and calibration use the existing prompt-disjoint `D_rm_train` and `D_cal`;
online training uses `D_rl_train_prompts`; validation and final evaluation use
held-out RL prompt roles.

## Risks guarded by tests

- exact reward-head reconstruction and pair bias cancellation;
- Cholesky solve instead of a matrix inverse;
- finite-sample quantile index and strict certification inequality;
- pair-order antisymmetry and PairPPO/CPDPO reward identities;
- no pair breakup in data loading;
- exact sum-then-pair-average PPO reduction;
- zero-certified batches remain finite and have zero pair gradient;
- equal prompts, responses, and proxy-RM calls across methods;
- gold symbols/checkpoints are rejected by training configuration;
- evaluation reuses response IDs across all scorers;
- plot code rejects CPDPO internal pair rewards as a policy-quality metric.
- reduced-data artifact builds are explicitly tagged `smoke`, require a
  smoke-only trainer opt-in, and are rejected by the full scientific launch;
  full artifacts continue to consume complete manifest roles.
