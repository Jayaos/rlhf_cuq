# Implementation plan

## Boundary

This plan preserves the Coste/legacy-trlx track and implements the experiment program in acceptance-gated stages. Stage 0 and the Stage 1 explicit data-split seam are implemented. No AdvPO reward, conformal equation, pairwise geometry, or adaptive state has been coded. Any row marked **BLOCKED** depends on an explicit owner decision in `docs/OPEN_METHOD_DECISIONS.md`.

“Baseline behavior” below means the default `defaults_rlhf + pythia_rlhf_individual` online reward/PPO computation at the pinned Coste revision. Additive diagnostics may leave that computation unchanged while still changing artifacts or runtime cost.

## Implemented Stage 0 changes

| Path | Existing component reused | Responsibility | Inputs → outputs | Equation | Test | Expected cost | Baseline behavior |
|---|---|---|---|---|---|---|---|
| `pyproject.toml` | Coste packaging | Constrain Python 3.10 and avoid invoking OA’s broken/conflicting transitive VCS resolution | Project install → local package only | None | Parse project metadata; import check after bootstrap | Install-time only | No algorithm change |
| `environment.cluster.yml`, `requirements/legacy-sources.txt`, `legacy-runtime.txt`, `legacy-build.txt`, constraints, and `scripts/configure_cluster_storage.sh` | Pinned OA/trlx/AlpacaFarm sources; trlx CUDA 11.8 environment; observed Phoenix build contracts | Install exact source commits editably/`--no-deps`; preinstall pinned Torch/build tools plus CMake/lit metadata and pybind11; isolate resolution from user/stale Python paths; explicitly supply observed W&B/threadpool dependencies and scoped pytest; keep build/model caches off quota-limited home storage | isolated shell + scratch-backed caches + pip/network → explicit legacy source/runtime environment | None | Exact-ref/build-pin/isolation tests, scoped repository tests, required imports, `pip check`, GPU smoke (target-host pending) | Install-time; source clones and large CUDA wheels | No runtime algorithm change; unresolved transitives remain documented |
| `artifacts/source_manifest.json` | GitHub/Hugging Face source metadata | Record source/model/data revisions, file hashes, schemas, licenses, missing assets, and legacy code hashes | Audited metadata → immutable JSON | None | `test_asset_audit.py`, `test_legacy_snapshot.py` | Kilobytes, no GPU | No |
| `scripts/audit_assets.py` | GitHub/Hugging Face metadata endpoints; local SHA-256 | Read-only local/remote provenance verification; never downloads payloads | Manifest + local tree (+ metadata network) → pass/fail text or JSON | SHA-256 only | Fake-client unit tests; offline self-audit; online audit | O(number of manifest files); negligible memory/network | No |
| `src/ppo/runtime_config.py` | Existing `TRLConfig.load_yaml` call site and loader kwargs | Resolve an explicit PPO YAML and apply opt-in local policy/proxy/AlpacaFarm snapshot paths while preserving all legacy identifiers by default | Optional paths + config namespaces → verified config/overrides | None | `test_runtime_config.py`, ensemble rejection | O(1) filesystem/config work | No for default; enables offline, revision-pinned smoke assets |
| `configs/config_rl.yaml` | Existing OA-style merged configuration | Add default PPO/local-asset controls, default-on offline gold phase, and opt-in `baseline_smoke` overlay | Config names/CLI overrides → runtime fields | None | YAML/config assertions | None | Defaults unchanged; smoke overlay deliberately changes scale and skips gold |
| `configs/ppo_config_smoke.yaml` | Original Coste PPO YAML | Two-rollout/chunk, one-epoch/update integration profile with short rollout generation and isolated checkpoints; batch two avoids legacy singleton-variance NaNs | Real SFT/proxy assets → tiny PPO artifacts | Same legacy PPO objective | Config assertions; target-host end-to-end smoke (pending) | One 1.4B-policy update plus RM/eval; still GPU work | Only when explicitly selected; never a reported experiment |
| `src/ppo/trainer_rl.py` | Existing trainer entry point and post-training `gold_score` | Select PPO YAML; make post-training gold phase opt-out for no-gold online smoke | Runtime config → same `trlx.train`; optional offline gold call | None | Source/default assertions; future no-gold-leakage smoke | One path check | Default remains old behavior; opt-out avoids gold load |
| `tests/test_asset_audit.py` | Stage 0 auditor | Exercise schema, hashes, revisions, file metadata, dataset schema, and failures without network | Synthetic manifests/metadata → assertions | None | It is the test | Milliseconds | No |
| `tests/test_legacy_snapshot.py` | Manifest baseline hashes | Fail if reward callback/scoring/PPO/data/config snapshot changes without explicit audit | Local bytes + expected hashes → assertions | SHA-256 | It is the test | Milliseconds | No; guards behavior-bearing files |
| `tests/test_runtime_config.py` | Runtime helper and YAML text | Verify old default, explicit smoke selection, missing-file failure, and no-gold defaults | Temp paths/repository files → assertions | None | It is the test | Milliseconds | No |

The smoke run itself remains pending because this host lacks the CUDA/Python 3.10 stack, a trained authenticated 44M proxy RM, and validated gold scorer. Stage 0 tests must not be presented as Gate 1.

## Stage 1: validate and instrument the Coste baseline

The data-manifest seam is implemented; the remaining changes are planned but not yet implemented. Each behavior correction must be isolated, compared to the hash-protected snapshot, and called out in an experiment manifest.

The first Stage 1 unit is now implemented: the primary
`configs/data_split_prompt_disjoint_v1.yaml`, retained Coste-native
`configs/data_split_coste_v1.yaml`, `scripts/build_data_manifest.py`,
`src/data_utils/split_manifest.py`, and
`src/data_utils/manifest_dataset_loader.py` create/verify exact logical roles
and adapt only train/validation roles to the legacy trainers. The RM wrapper
`trainer_rm_manifest.py` avoids altering the hash-protected Coste trainer and
bootstraps the repository root for direct path-based `accelerate launch` use;
PPO selects the adapter only when `data_split_manifest_path` is nonempty.
Standard-library tests cover exact quota rounding, order independence, prompt
grouping, disjointness, data/ID hashes, tamper failure, launcher import-path
ordering, and trainer role isolation. A real-asset build and GPU trainer smoke
remain target-host acceptance work.

The first manifest-backed RM launch also reached the pinned Open-Assistant
tokenizer/model factory and exposed its architecture-name substring contract.
`local_model_compat.py` now consumes the explicit cluster-only
`model_family: pythia` hint, verifies the local checkpoint's `config.json`
declares `model_type: gpt_neox`, and preserves the original path while invoking
the unchanged legacy tokenizer and `GPTNeoXRewardModel` branches. This is an
offline path-compatibility adapter, not a change to RM architecture or training
semantics.

The target-host RM path now has separate Slurm entry points. The non-scientific
`scripts/slurm/smoke_proxy_rm.sbatch` selects 512 deterministic training pairs
through the existing custom sampler, 128 validation pairs through the manifest
adapter, one epoch/16 optimizer steps, two evaluations, final normalization,
and an isolated checkpoint. It verifies the pinned base and split bundle before
training and validates finite normalization/model/tokenizer artifacts after
training. `scripts/slurm/smoke_proxy_rm_any_gpu.sbatch` reuses the same small
data/step profile but adds an explicit FP16/no-FlashAttention overlay and FP16
Accelerate config, allowing a queue-availability check on V100/RTX6000/A100
without altering the primary BF16 path. `scripts/slurm/train_proxy_rm.sbatch`
remains the full five-epoch entry point. All jobs keep caches on scratch and
leave the hash-protected trainer unchanged. Full seeds 1--5 are independent
Slurm array tasks, not one distributed RM job, and smoke checkpoints are never
valid PPO inputs or experimental results.

The first Phoenix real-asset attempt exposed two loader compatibility defects
before any bundle was written. The legacy environment now pins
`fsspec==2023.9.2`, and the builder no longer passes whole snapshot directories
to generic dataset discovery. Exact verified JSON paths and expected raw counts
are frozen in the two named split configs; regression tests reject an
unverified/reused file or duplicated row pool. The strict primary config
excludes the Coste preference `unlabelled` pair file, includes the other three
training files plus preference `val` in the RM pool, reserves AlpacaFarm
`unlabeled` exclusively for PPO, and forbids every RM/PPO prompt overlap. A
successful real-asset strict build and manifest verification remain the
target-host acceptance evidence.

| Planned path | Existing component reused | Responsibility | Inputs → outputs | Equation | Required test | Expected cost | Baseline behavior |
|---|---|---|---|---|---|---|---|
| `scripts/build_data_manifest.py` | Coste preference loader; AlpacaFarm prompt loader | **IMPLEMENTED:** materialize revision-pinned, stable, disjoint IDs for RM train/val/calibration and RL train/val/test | Pinned dataset snapshots + seed/split policy → JSONL manifests + hashes | None | CPU synthetic tests pass; real pinned-payload build pending on cluster | O(dataset rows), materialized in host memory | Changes data selection from implicit first-N; required controlled adaptation |
| `src/reward_modeling/scoring/feature_extraction.py` and optional path in `score.py` | OA `GPTNeoXRewardModel.out_proj` | Capture the exact scalar-head input via pre-hook without duplicating pooling; return features only when requested | Tokenized prompt/answer + frozen RM → unchanged logits and optional `[batch,d]` features | `r = w^T e + c` identity only | Reconstruction tolerance, padding/EOS, split-batch equivalence, legacy output identity | Adds O(batch·d) memory/output when enabled; one existing forward | Default no; optional feature mode adds cost/artifact |
| `src/reward_modeling/scoring/ppo_reward_functions.py` | Existing callback factory | Thread optional feature/metadata return through a backward-compatible API and define single-RM variance as absent rather than garbage | Samples/prompts/outputs → proxy scalar (+ optional feature/diagnostics) | None | Callback contract and byte-for-byte default score comparison | Negligible beyond feature capture | Correcting serialized variance changes invalid diagnostics, not online reward |
| `src/ppo/uncertainty_logging.py` plus evaluation seam in `custom_accelerate_base_trainer.py` | Existing eval JSON serialization | Store prompt/sample IDs, optimizer/rollout step, seed, checkpoint, text, tokens, raw proxy, explicit uncertainty fields, and exact KL definitions | Gathered eval/rollout records → append-only schema-versioned JSONL | Named metrics only | Schema, rank aggregation, resume/no-duplicate records, same-sample joins | O(samples·text/features if enabled); bounded buffers | Online reward no; artifacts/runtime yes |
| KL helper in `src/ppo/uncertainty_logging.py` | Coste Eq. (10) calculation | Add response-only, sequence-summed/sample-mean, globally reduced `0.5*sum_t(log pi-log pi_ref)^2`; retain historical field under a legacy name | Response masks + logprobs + world aggregation → explicit KL fields | Coste Appendix C Eq. (10) | Prompt-mask exclusion, hand calculation, one-rank/simulated multi-rank | O(response tokens), one scalar all-reduce | Adds corrected metric; never relabels old `policy/kl` |
| `src/ppo/reference_cache.py` | trlx prompt metadata propagation; frozen SFT policy | Generate/cache one declared reference response and proxy feature per prompt with IDs/revision/decoding provenance | Prompt manifest + frozen SFT + decoding config → immutable cache | Reference expectation input only | Cache determinism, key/provenance mismatch, no trainable-policy use | One offline generation/RM pass per prompt; disk O(N·text+d) | No PPO change until a reward adapter consumes it |
| `src/ppo/custom_helpers.py` and `src/ppo/run_ppo_gold_eval.py` | Existing offline gold seam | Validate AlpacaFarm formatting/loading, load once, sort/filter eval files, run single-process, and never import/load gold online | Saved rollout JSON + reconstructed gold RM → added gold scores | None | Known formatter fixtures, batching, one load, online scorer forced to raise | Large 7B offline inference; no training-process memory | Corrects offline outputs; online reward unchanged |
| isolated checkpoint configuration in `process_configs` | trlx `TrainConfig.checkpoint_dir` | Put checkpoints under the run directory and record resume provenance | Run config → isolated checkpoint path | None | Two runs do not collide; one-update save/resume equivalence | Disk equal to model/optimizer state | Artifact location changes; optimizer math does not |
| terminal reward/mask validation around `make_experience` | Existing Coste PPO trainer | Prove or minimally correct terminal placement for mixed response lengths | Variable-length rollouts → masked token rewards | Legacy PPO reward placement | Mixed lengths/EOS/padding hand fixtures | O(tokens), no extra model pass | Unknown until test; any correction is an explicit baseline patch |

Gate 1 requires a deterministic real-asset smoke, saved generations, proxy/gold correctness, exact feature identity, understood KL, checkpoint resume, and a no-gold-online test. Multi-rank remains out of scope until the same tests pass with distributed aggregation.

## Stage 2: exact AdvPO baseline and Section 5.1 diagnostics

| Planned path | Existing component reused | Responsibility | Inputs → outputs | Equation | Required test | Expected cost | Baseline behavior |
|---|---|---|---|---|---|---|---|
| `src/uncertainty/linear_algebra.py` | PyTorch Cholesky/triangular solves | Stable PSD accumulation, factorization, solves, dtype/device/provenance; never explicit inverse | Feature matrices + declared regularizer → factor/solve API | Linear solves for `M_D^-1 v` | Symmetry/PD, solve vs tiny inverse, dtype/device, failure diagnostics | Build O(Nd²), factor O(d³), state O(d²) | Additive offline module |
| `src/uncertainty/advpo_geometry.py` and `scripts/build_advpo_geometry.py` | Pinned RM feature API + explicit preference manifest | Build exact Eq. (4) from one chosen and one rejected individual feature per example; cache factorization metadata | `D_rm_train`, frozen RM, regularizer → `M_D` factor artifact | AdvPO Eq. (4) | Orientation/count, PSD, repeated-direction uncertainty, cached/on-the-fly agreement | Two RM forwards/example (batched); O(Nd²); O(d²) state | No baseline change |
| `src/uncertainty/advpo_objective.py` | Geometry solve; proxy scalar/features; reference cache | Compute global-batch `g`, `lambda_star`, adjusted policy and reference rewards, and diagnostics | Global rollout/reference features/rewards + `B` + factor → adjusted rewards | AdvPO Eqs. (8)–(9), exactly as frozen | Aggregate identity, `B→0`, identical features, near-zero `g`, scaling, rank aggregation | O(global batch·d + solve); O(d + batch) working memory | Only selected AdvPO method; standard PPO remains identical |
| rollout-pool seam in `custom_accelerate_ppo_trainer.py` | Existing generate/gather/scatter path | Buffer a complete declared rollout pool or perform mathematically equivalent sufficient-statistic aggregation before reward finalization | All chunks/ranks + method adapter → globally consistent rewards | Finite estimator chosen for AdvPO `g` | Chunk-size invariance; one-rank vs simulated/distributed ranks; no deadlock | May retain O(R·tokens/features); collectives O(d) or gathered samples | No for standard PPO branch; changes AdvPO scheduling |
| `src/ppo/reward_adapters.py` | `custom_helpers.get_reward_fn` factory seam | Common baseline/AdvPO adapter lifecycle, diagnostics, state serialization | Method config + batch inputs → scalar rewards + diagnostics/state | Baseline identity or Eq. (9) | Baseline callback identity, no-gold leakage, adapter save/load | Dispatch overhead; method-specific costs above | Baseline adapter must be exactly identical |
| `scripts/run_section51_diagnostics.py` | Fixed sample logs + offline gold + feature API | Score proxy/gold/CI/ensemble on the same stored samples every declared interval; never retrain per estimator | Immutable sample IDs/artifacts → raw metric table | `U_CI=b*sqrt(e^T M_D^-1 e)` | Same-ID joins, signed/absolute/aligned definitions, Pearson fixture | Offline RM/solve passes; O(samples·d) | No training change |
| `configs/experiments/...` and `scripts/run_section52_methods.py` | Coste trainer/config merge | Separate Coste-native and AdvPO-stress protocols; pair seeds/manifests/checkpoints | Frozen experiment config → immutable run manifest | Selected method only | Config snapshot, seed pairing, forbidden-gold scan | Declared full GPU budget only after gates | Explicit per track; never silently edits baseline YAML |

Before these rows can be implemented, the owner must freeze the Eq. (4) regularizer, global finite estimator for `g`, zero-`g` behavior, reference generation, reward rescaling, and `B` protocol. They are listed in the open-decisions document.

## Stages 3–4: proposed static and policy-adaptive method

All rows in this section are **BLOCKED**. The names describe interfaces, not chosen mathematics.

| Planned path | Existing component reused | Responsibility | Inputs → outputs | Equation | Required test | Expected cost | Baseline behavior |
|---|---|---|---|---|---|---|---|
| `src/uncertainty/pairwise_geometry.py` and `scripts/build_pairwise_geometry.py` | Feature API + explicit preference/calibration manifests | Build exactly one chosen-minus-rejected vector per preference example and a frozen geometry artifact | Paired features/labels + frozen geometry definition → factor/provenance | **OPEN A** (`V=lambda I+sum d_i d_i^T` vs Fisher weighting) | Orientation/swap invariance, no all-pairs O(N²), PSD, logit-difference identity | O(Nd²), factor O(d³), state O(d²) | Additive until selected |
| `src/uncertainty/conformal_base.py` | Explicit `D_cal` and pairwise geometry | Define score API and exact finite-sample/weighted quantile primitives | Calibration records/weights + alpha → `q_alpha` + diagnostics | **OPEN B/D** | Quantile index, ties, normalization, zero weights, exchangeable coverage | Sort O(N log N), memory O(N) | Additive |
| `src/uncertainty/conformal_static.py` | Frozen conformal primitive | Freeze static calibration state and map it to uncertainty/reward diagnostics | Calibration artifact + rollout feature → declared output | **OPEN B/D/F** | Fixed `q` invariant; synthetic coverage; objective identity | Usually O(batch·d + solve) | Only proposed-static selection |
| `src/uncertainty/conformal_policy_adaptive.py` | Static state + rollout/reference cache | Estimate declared shift and update weighted calibration at declared cadence/window | Past allowed rollout metadata + calibration data → `q_alpha,t` + weights | **OPEN C/D/E** | Shift/no-shift behavior, clipping/zero weights, no current-gold use, rank consistency | Depends on estimator; must be budgeted before freeze | Only proposed-adaptive selection |
| `src/uncertainty/state.py` | PyTorch/module checkpoint conventions | Serialize quantile, window/history, estimator, cache version, geometry metadata, and log step | Method state ↔ checkpoint dictionary | Frozen equations only | Interrupted vs uninterrupted deterministic smoke | State O(d² + window) | Only selected methods |
| proposed branches in `src/ppo/reward_adapters.py` | Common adapter boundary | Apply exactly the frozen robust objective without gold access | Proxy/reference/features + method state → training reward | **OPEN F** | Synthetic hand calculation, baseline isolation, no-gold leakage | Method-dependent | Only explicitly selected method |

Implementation may start only after `docs/PROPOSED_METHOD_FROZEN.md` records every required equation, convention, guarantee, and config value and the owner approves it. The code and run manifests must point to that frozen document’s hash.

## Verification and experiment order

1. Run standard-library Stage 0 tests and offline/online metadata audit.
2. Resolve/install the legacy GPU environment; freeze its final lock/container.
3. Train or authenticate the proxy RM and validate the offline gold reconstruction.
4. Pass Gate 1 on the opt-in smoke and then on the declared Coste-native config.
5. Add feature identity and sample-level provenance before uncertainty code.
6. Freeze AdvPO implementation choices, build Eq. (4), reproduce Section 5.1 diagnostics, then run PPO/AdvPO paired smoke.
7. Confirm overoptimization in a predeclared track and pass exact AdvPO Gate 3.
8. Freeze all proposed equations and only then implement static/adaptive rows.
9. Run identical seeds/manifests/checkpoints across methods; gold remains offline-only.

No large GPU run is authorized by this plan until its preceding acceptance gates pass.
