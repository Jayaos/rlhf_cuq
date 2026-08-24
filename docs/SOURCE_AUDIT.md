# Source audit

Audit date: 2026-08-24. This is the Stage 0 record required by `AGENTS.md`; it is not evidence that a GPU baseline has run.

## Outcome and scope

The workspace initially contained only `AGENTS.md` and was not a Git repository. The Coste implementation was therefore vendored from the audited commit below. Its legacy reward/PPO path is protected by the hashes in `artifacts/source_manifest.json`. Stage 0 adds only immutable dependency pins, a read-only metadata auditor, an opt-in one-update smoke configuration, and tests. It does not implement AdvPO or any proposed-method equation.

No clearly official public AdvPO implementation was found. The final NeurIPS paper is therefore the authoritative algorithm source. Its reproducibility checklist answers public code/data “No” while stating an intent to release later; the proceedings page, arXiv/OpenReview records, author accounts, ByteDance organization, and indexed repository searches exposed no official code as of the audit date. This is strong negative search evidence, not proof that private or unindexed code does not exist.

## Immutable source selection

| Component | Audited revision | License | Role and caveat |
|---|---|---|---|
| [Coste `llm_optimization`](https://github.com/tlc4418/llm_optimization/tree/416b03cc2c3c8125208679acd88891584d9eefd2) | `416b03cc2c3c8125208679acd88891584d9eefd2` | MIT | Vendored baseline. Current `main` matched the pin during the audit. |
| [Open Assistant](https://github.com/LAION-AI/Open-Assistant/tree/e1769c102f1597cc0b53a8b915f858239d197aeb) | `e1769c102f1597cc0b53a8b915f858239d197aeb` (`v0.0.4-alpha2`) | Apache-2.0 | Supplies `model_training`, `oasst_data`, formatting, and `GPTNeoXRewardModel`. |
| [CarperAI/trlx](https://github.com/CarperAI/trlx/tree/3340c2f3a56d1d14fdd5f13ad575121fa26b6d92) | `3340c2f3a56d1d14fdd5f13ad575121fa26b6d92` | MIT | Selected compatible legacy PPO dependency. Coste/OA previously followed moving `main`; this revision is an audit selection, not recoverable from an upstream lock. Package metadata says 0.7.0, but this commit is not the `v0.7.0` tag. |
| [Coste AlpacaFarm fork](https://github.com/tlc4418/alpaca_farm/tree/f92bd550130975436301ba02137b303d1eb59986) | `f92bd550130975436301ba02137b303d1eb59986` | Apache-2.0 | Gold-RM class and reconstruction tooling. Coste previously followed moving `main`. |

`requirements/legacy-sources.txt` now uses the commit SHAs above. It installs all four source packages editably with dependency resolution disabled. This is necessary because the pinned OA model package both declares a moving direct `trlx` URL (which conflicts with a different root direct URL under pip) and builds a wheel that omits subpackages imported by this trainer. Runtime dependencies are installed separately. Modified Open Assistant and trlx-derived files remain subject to their upstream notices; Coste’s root MIT license does not replace those notices.

## Papers and method provenance

- Original AdvPO: [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/94bbcb744bbada8808fda05b9d9290d6-Paper-Conference.pdf). Equations (4), (8), and (9), Sections 5.1/5.2, and appendices are original-paper claims.
- Original Coste setup: [ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/dda7f9378a210c25e470e19304cce85d-Paper-Conference.pdf) and the pinned repository.
- Controlled adaptations planned for this repository include applying AdvPO to Coste assets, fixed prompt/reference manifests, exact response-only/global KL logging, and denser sample logs. They must be labeled adaptations.
- Pairwise conformal geometry, nonconformity, policy-shift weighting, adaptive quantiles, and their policy objective remain proposed and unresolved. They are not implemented in Stage 0.

AdvPO Eq. (4) uses individual chosen and rejected feature outer products,

`M_D = lambda I + sum_i sum_{y in {chosen,rejected}} e(x_i,y)e(x_i,y)^T`.

For Eq. (9), `g = mean_policy(e) - mean_reference(e)`, `B=b^2`, and `lambda_star=sqrt((g^T M_D^-1 g)/B)`. Subtracting `(1/lambda_star)e^T M_D^-1 g` from both policy and reference rewards gives aggregate proxy displacement minus `b*sqrt(g^T M_D^-1 g)`. It is not independent per-sample CI subtraction. The paper does not provide a numerical geometry regularizer, finite/distributed estimator for `g`, the zero-`g` convention, or an unambiguous reward-rescaling formula. Those are open implementation decisions.

## Released assets

Exact revisions, required filenames, Git blob IDs, LFS SHA-256 values, dataset row/schema expectations, and unresolved inputs are machine-readable in `artifacts/source_manifest.json`.

| Asset | Revision | License/provenance status |
|---|---|---|
| `tlc4418/pythia_1.4b_sft_policy` | `7b927d2ca0da03b81e1532a9fec1288fd4ac4d39` | No explicit Hub license metadata; Pythia/AlpacaFarm provenance does not cure the omission. |
| `tlc4418/pythia_70m_sft` | `4f0328e68e44399767f77faaa39837e148bc6971` | No explicit Hub license metadata. This is only the base for locally training the effective 44M RM. |
| `tlc4418/1.4b-policy_preference_data_gold_labelled` | `426ed801a0322fb15e52631cff85b17f12b4f275` | No explicit Hub license metadata; 51,383 viewer rows at audit time (49,383 train, 2,000 validation). Code must use a revision and explicit ID manifest rather than trusting counts/order. |
| `tatsu-lab/alpaca_farm` | `e576524ca841af3c36fd6912e68e5920430928c1` | CC BY-NC 4.0; research/non-commercial restrictions apply. |
| `tatsu-lab/alpaca-farm-sft10k-wdiff` | `91b6ac09e67d4f65e87ecf0585f8790c2de7edbb` | Weight diff; AlpacaFarm terms plus the user-supplied LLaMA license apply. |
| `tatsu-lab/alpaca-farm-reward-model-human-wdiff` | `363f1a5745895431849cd1c1f451bb837646c14f` | Weight diff; same compound licensing and provenance caveat. |

No author-released Coste 44M proxy-RM checkpoint was located. The faithful baseline must train `models/rm-pythia-44m_seed1` from the pinned 70M SFT base, then record its full checkpoint/head hashes. A third-party model must not be silently substituted.

The gold model is not a standalone download. Use the pinned AlpacaFarm recovery code with a legally obtained, checksummed Hugging Face LLaMA-7B base; reconstruct pinned `sft10k` first, then pinned `reward-model-human`, require its `model_sum.txt` integrity check to pass, and record the base provenance plus the complete reconstructed checkpoint checksum. AlpacaFarm’s recovery script itself omits `revision=` in Hub calls, so a reproducible wrapper is still required before gold scoring.

## Coste PPO execution trace

1. `trainer_rl.py` asks the patched OA loader for RL data. OA has no `alpaca_farm` RL registration, so the local fallback loads AlpacaFarm `alpaca_instructions`: `unlabeled` for PPO and `val` for evaluation. Instruction and nonempty input are joined with a newline. No stable prompt IDs survive.
2. OA `format_pairs(..., add_initial_reply_token=True)` emits `<|prompter|>{text}{eos}<|assistant|>`.
3. `trlx.train` builds `PromptPipeline`. `CustomAcceleratePPOTrainer.make_experience` generates one local chunk, gathers it across ranks, calls the reward callback once on rank zero, scatters scalar rewards, and stores local rollouts.
4. `create_reward_fn` rebuilds OA-formatted prompt/answer strings and calls one or more frozen proxy RMs. Scoring tokenizes to 776 tokens and returns raw scalar logits; saved RM mean/std are unused and the internal scoring batch is hard-coded to 32.
5. PPO adds token KL penalties and places the clipped scalar proxy score at the terminal stored reward position. The rollout store is shuffled and reused for four PPO epochs before regeneration.
6. Evaluation runs before the first update and every configured optimizer-step interval. It writes instructions, answers, proxy scores, and ensemble variance to JSON. Offline `gold_score` later reopens each JSON file and adds gold scores.

Gold is not supplied to `trlx.train` or the online callback. In the original entry point it runs only after training and final save. The new smoke profile explicitly skips that separate phase; the baseline default remains enabled.

## Effective batch and step semantics

For the checked config, process count `W=1`, `num_rollouts R=4`, `chunk_size C=2`, requested train batch `T=32`, and `ppo_epochs=4`.

| Quantity | General checked-code meaning | With `W=1` |
|---|---|---:|
| Reward callback batch | Gathered generation chunk, normally `W*C` | 2 |
| Local/global rollout pool | Normally `R` per rank / `W*R` global | 4 / 4 |
| Actual local update batch | `min(T, local store size)` because the final partial batch is kept | 4 |
| Effective DDP update batch | Normally `W*4` | 4 |
| Passes and optimizer updates per pool | Four passes; one partial loader batch per pass | 4 |
| Pools / generations over 3,000 optimizer updates | `3000/4`; each pool has four local generations | 750 / 3,000 |

`total_steps`, evaluation intervals, and checkpoint intervals count optimizer updates, not rollout collections. `minibatch_size` defaults to requested batch 32, yet the four-element partial batch is accepted. The YAML’s `gradient_accumulation_steps: 32` is not passed to `Accelerator()`, so DeepSpeed’s resolved batch semantics require an actual target-host check. An Eq. (9) adapter attached only to the callback would see a two-sample chunk, not the four-rollout pool, and would be wrong unless the intended estimator explicitly chose that approximation.

## Exact proxy feature tensor

At the pinned OA revision, `GPTNeoXRewardModel` takes `outputs[0]`, pools according to the checkpoint configuration, and sends that pooled vector directly to `out_proj = nn.Linear(hidden_size, 1)` with bias. For the intended `pooling: last`, OA computes `attention_mask.cumsum(1).argmax(1)` and gathers that token’s final hidden state.

The lowest-drift future extractor is an optional forward pre-hook on `reward_model.out_proj` that captures and detaches `inputs[0]` across scoring minibatches. Its mandatory identity test is `output.logits == F.linear(feature, weight, bias)` at tight tolerance, including padding/EOS and split-batch cases. Stage 0 deliberately does not add this hook.

## Paper-versus-code drift and baseline defects

| Area | Paper/protocol | Pinned code | Status |
|---|---|---|---|
| Coste rollout settings | 256 rollouts, chunk 32 | 4 rollouts, chunk 2 | Material debug-scale drift; do not call the checked YAML paper-faithful. |
| Other Coste PPO values | 3,000 updates, batch 32, PPO epochs 4, LR `1e-6`, clips 0.2, max output 256 | Same headline values | Batch 32 is only a loader maximum in the checked four-rollout run. |
| Reward handling | Paper does not report ±10 clipping | `scale_reward=false`, but proxy reward is clipped to ±10 | Must be declared. |
| Plotted Coste KL | Sequence-summed/sample-mean `0.5*(log pi/log pi_init)^2` | `mean_kl_est` is saved as `policy/kl` | Code includes prompt tokens and does not all-reduce this estimator. Only a different k3-like metric is all-reduced. |
| Gold configuration | AlpacaFarm `reward-model-human` specialized path | YAML says `is_alpacafarm_rm: false`; README says true | Blocker. The specialized formatter also has a missing f-string and emits literal placeholders. Do not trust either branch yet. |
| Single-RM variance | Not a meaningful ensemble metric | `torch.empty_like` is serialized | Uninitialized garbage; do not analyze it. |
| Eval comparability | Same samples, dense provenance needed for Section 5.1 | Stochastic eval, no prompt IDs/step/seed/checkpoint/token counts; cumulative KL only on first row | Insufficient for the target analysis. |
| Multi-rank | Global scientific statistics required | saved KL not global; RM placement and post-training gold phase are unsafe on multiple ranks | Limit baseline validation to the audited single-process launcher until fixed/tested. |

AdvPO uses 1,500 steps, batch 64, context 2,048, max generation 512, beta 0, evaluation every 100 steps, chosen response references, 30% random mislabeling, and `B` in `{1,5,10,15}`. Those values describe a separate stress/protocol track; they are not silently merged into the Coste-native baseline.

## Required patches and acceptance state

Implemented Stage 0 setup patches are behavior-preserving by default: immutable VCS pins; explicit Python 3.10 compatibility range; selectable PPO config with the old path as default; opt-in one-update config; optional local policy/proxy/dataset snapshot overrides; optional post-training gold phase with the old `true` default; and read-only provenance tests.

Before Gate 1, separately audited patches must validate feature identity, correct/verify gold formatting and loading, replace the single-RM variance garbage with an explicitly defined field, implement response-only globally aggregated KL without relabeling historical data, add stable sample provenance, isolate checkpoints, verify terminal indexing, and test checkpoint/resume and one-rank behavior. These known defects were recorded rather than folded into Stage 0, because changing them now would make a failed baseline impossible to attribute.
