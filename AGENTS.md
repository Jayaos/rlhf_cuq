## 0. How Codex must use this file

This file summarizes the current research decisions made in the ChatGPT Project **“RLHF using Uncertainty Quantification.”** Codex cannot read that project’s conversation history directly, so this file is the context bridge.

Before changing research code, Codex must:

1. Read this entire file.
2. Inspect the repository and all upstream code that this file identifies.
3. Verify every relevant equation against the cited papers.
4. Separate:
   - an exact description of the original AdvPO protocol,
   - the Coste et al. implementation environment,
   - our controlled adaptation of the AdvPO protocol to that environment,
   - and unresolved design choices in our proposed method.
5. Produce a source-audit and implementation plan before making substantial changes.
6. Stop and request a method decision when a required equation or protocol choice is marked **OPEN**. Do not silently invent one.

Priority order:

> **Scientific fidelity → fair comparison → reuse of validated code → minimal code changes → computational convenience.**

This project is not asking Codex to design a new PPO stack, a new reward-model trainer, or an unrelated benchmark. The core task is to reproduce the logic of **Sections 5.1 and 5.2 of the AdvPO paper** in the **Coste et al. AlpacaFarm/Pythia synthetic-RM environment**, then add our proposed policy-adaptive conformal uncertainty method to the same controlled experiment.

---

# 1. Non-negotiable directives

## 1.1 The experimental target is AdvPO Sections 5.1 and 5.2

“Section 5.1” and “Section 5.2” in this document always mean the sections of:

> Xiaoying Zhang, Jean-François Ton, Wei Shen, Hongning Wang, and Yang Liu,  
> **Mitigating Reward Overoptimization via Lightweight Uncertainty Estimation**, NeurIPS 2024.

They do **not** mean Sections 5.1/5.2 of the Coste et al. paper.

The intended experiments are:

- **AdvPO Section 5.1 analogue:** Does an uncertainty estimate increase when proxy reward becomes unreliable under PPO-induced policy distribution shift?
- **AdvPO Section 5.2 analogue:** Does using that uncertainty during optimization mitigate proxy-reward overoptimization and preserve gold reward?

Our proposed method must be evaluated in both roles:

1. As an uncertainty estimator/diagnostic on a common PPO trajectory.
2. As an uncertainty-aware policy-optimization method.

## 1.2 Use the Coste et al. environment and released assets

The primary environment is based on:

> Thomas Coste, Usman Anwar, Robert Kirk, and David Krueger,  
> **Reward Model Ensembles Help Mitigate Overoptimization**, ICLR 2024.

Use, adapt, and pin the authors’ existing pipeline wherever possible:

- Repository: `tlc4418/llm_optimization`
- Initial policy: `tlc4418/pythia_1.4b_sft_policy`
- Proxy-RM base: `tlc4418/pythia_70m_sft`
- Preference dataset: `tlc4418/1.4b-policy_preference_data_gold_labelled`
- Gold evaluator: AlpacaFarm `reward-model-human`
- PPO prompts: AlpacaFarm `unlabeled` split through the Coste data loader
- Existing PPO backend: Open-Assistant + `trlx` through the Coste repository
- Existing post-hoc gold-scoring path
- Existing individual-RM and ensemble reward-function interfaces

Do not replace this stack merely because a newer package exists. First establish a faithful baseline in the original stack. A modern port may be added later as a separate, validated engineering track.

## 1.3 Do not reinvent components that already exist

Before writing code, search the Coste repository and relevant upstream packages for:

- data loading and prompt formatting,
- SFT and proxy-RM model loading,
- reward-model training,
- PPO rollout generation,
- reference-policy/KL handling,
- proxy reward scoring,
- gold reward scoring,
- checkpoint and evaluation logging,
- ensemble WCO/UWO objectives,
- distributed launch and DeepSpeed support.

New code should be limited primarily to:

- extraction of the exact reward-head feature representation,
- AdvPO confidence geometry and batch robust-reward adapter,
- our proposed pairwise geometry and conformal calibrator,
- policy-adaptation logic,
- additional logging/evaluation,
- tests,
- and experiment configuration.

## 1.4 No gold-reward leakage

The gold reward model may be used for:

- producing the already released synthetic preference labels,
- offline evaluation of stored generations,
- validation/test analysis in an explicitly synthetic benchmark,
- and optional oracle diagnostics clearly labeled as such.

The gold reward model must **not** be used as the PPO training reward, to update the proposed calibrator online, to choose individual rollout actions, or to calculate a deployable uncertainty score.

A separate oracle experiment is allowed only when clearly named and never confused with the proposed deployable method.

## 1.5 Do not invent unresolved equations

The project has a settled conceptual method, but several exact equations remain to be frozen. Codex may implement common infrastructure and the AdvPO baseline before those decisions are resolved. Codex must not silently choose:

- a conformal nonconformity score,
- a policy-shift weighting rule,
- a window/update schedule,
- a weighted quantile convention,
- a precise robust reward formula for the proposed method,
- or whether the proposed geometry is updated online.

These are listed explicitly in Section 12.

---

# 2. Executive summary of the scientific plan

The proxy reward model is trained on responses produced by the initial 1.4B SFT policy. PPO then repeatedly updates the policy:

\[
\pi_0 \rightarrow \pi_1 \rightarrow \cdots \rightarrow \pi_t.
\]

Consequently, the distribution of current responses and their reward-model representations moves away from the distribution used to train and calibrate the proxy RM. This is the policy-distribution-shift problem.

AdvPO estimates uncertainty from the proxy RM’s last-layer features using a fixed confidence geometry constructed from the preference dataset. Our proposed method instead emphasizes the preference-relevant **pairwise feature difference**

\[
d_i = h_i^A-h_i^B,
\]

and uses conformal calibration whose threshold can adapt to the policy distribution:

\[
q_{\alpha,0}, q_{\alpha,1}, \ldots, q_{\alpha,t}.
\]

The main claim to test is not merely “coverage is high.” It is:

> As the policy changes, does the uncertainty/calibration mechanism continue to identify unreliable proxy rewards, and does policy-adapted calibration improve the gold-reward trajectory or reduce overoptimization compared with PPO, AdvPO, and a fixed-calibration version of our method?

This is a **controlled protocol replication/adaptation**, not an exact numerical reproduction of AdvPO, because the original AdvPO experiments use different datasets and model families. The experimental logic should match AdvPO Sections 5.1 and 5.2, while models, data, and most of the PPO machinery come from Coste et al.

---

# 3. Canonical terminology and model roles

Do not conflate the following components.

## 3.1 Initial/SFT policy

\[
\pi_{\mathrm{SFT}} = \pi_0
\]

Recommended released checkpoint:

```text
tlc4418/pythia_1.4b_sft_policy
```

Roles:

- initialization of the trainable PPO policy,
- fixed reference policy for KL measurement/penalty,
- optional generator of cached reference responses for AdvPO and the proposed pairwise method.

The initial SFT policy is not a reward model.

## 3.2 Trainable PPO policy

\[
\pi_{\theta_t}
\]

This is initialized from \(\pi_{\mathrm{SFT}}\), generates online rollouts, and is updated by PPO. Its rollout distribution changes over time.

## 3.3 Proxy reward model

\[
\widehat r(x,y)
\]

This is the imperfect RM available to the optimizer.

Roles:

- returns scalar proxy reward,
- exposes the feature representation immediately before the scalar reward projection,
- provides the geometry used by AdvPO and the proposed method,
- remains frozen throughout PPO.

The primary first implementation should use one Coste-style proxy RM, commonly the effective 44M configuration trained from `tlc4418/pythia_70m_sft`, unless source audit reveals a better already-released compatible checkpoint.

## 3.4 Gold reward model

\[
r_{\mathrm{gold}}(x,y)
\]

Primary evaluator:

```text
AlpacaFarm reward-model-human
```

Roles:

- synthetic “ground truth” evaluator,
- source of the already released preference labels,
- offline evaluator of PPO generations.

It must not be in the online training reward path.

## 3.5 PPO value model / critic

This is ordinary PPO infrastructure and should be reused from the existing trainer. It is not the proxy RM and not the gold RM.

## 3.6 Reference response

For AdvPO, each prompt \(x\) needs an acceptable response \(y^{\mathrm{ref}}\). The AdvPO paper permits annotated good responses or responses generated by the SFT policy.

The original AdvPO synthetic setup uses the chosen response in its policy-optimization dataset. Coste PPO uses AlpacaFarm’s unlabeled prompt split, so the same chosen-response field may not exist for those PPO prompts.

**Recommended Coste adaptation, pending explicit confirmation:**

- Generate one response per PPO prompt using the frozen SFT policy before PPO starts.
- Use fixed generation settings and seed.
- Cache prompt ID, response text, tokenization metadata, proxy reward, and proxy-RM feature.
- Never regenerate a different reference response during a run.
- Use the same reference cache for every optimization method and seed where compatible.

Do not join preference examples to PPO prompts using fragile raw-string matching unless a source audit proves that IDs and semantics align.

---

# 4. What the original AdvPO experiments do

## 4.1 AdvPO uncertainty geometry

Let \(e(x,y)\in\mathbb{R}^d\) be the final reward-model feature before the scalar projection. AdvPO constructs

\[
M_D =
\lambda I
+
\sum_{i=1}^{N}
\sum_{y\in\{y_i^c,y_i^r\}}
e(x_i,y)e(x_i,y)^\top.
\]

Its confidence-interval uncertainty is

\[
U^{\mathrm{CI}}(x,y)
=
b\sqrt{e(x,y)^\top M_D^{-1}e(x,y)}.
\]

Important implementation implications:

- \(M_D\) is constructed from both chosen and rejected **individual** response features.
- It is static after construction.
- The RM is frozen.
- Do not explicitly form a numerical inverse when a Cholesky solve can be used.
- \(b\), or \(B=b^2\), is a scale/confidence-region hyperparameter in the AdvPO experiments; it is not empirically calibrated by a conformal coverage procedure.

## 4.2 AdvPO robust objective

With reference responses, the paper optimizes

\[
\max_{\pi_\theta}
\min_{\|\phi-\widehat\phi\|_{M_D}\le b}
\left[
\mathbb{E}_{x,y\sim\pi_\theta}
r_\phi(x,y)
-
\mathbb{E}_{x,y^{\mathrm{ref}}}
r_\phi(x,y^{\mathrm{ref}})
-
\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{SFT}})
\right].
\]

Define the batch/population feature displacement

\[
g_t =
\mathbb{E}_{x,y\sim\pi_{\theta_t}}[e(x,y)]
-
\mathbb{E}_{x,y^{\mathrm{ref}}}[e(x,y^{\mathrm{ref}})].
\]

The inner minimization has a closed form. At the aggregate level, the robust reward-difference term is equivalent to

\[
\widehat\phi^\top g_t
-
b\sqrt{g_t^\top M_D^{-1}g_t}.
\]

The paper also gives a per-sample form suitable for standard policy optimization. Implement the paper’s exact Eq. (9), and unit-test that its batch mean matches the aggregate closed form up to numerical tolerance and any documented normalization.

**Critical:** This is a batch-coupled objective. It is not generally equivalent to independently subtracting \(U^{\mathrm{CI}}(x,y)\) from each sample’s reward. The existing Coste reward callback receives a batch, so a batch-level adapter is feasible.

## 4.3 AdvPO Section 5.1 target

The paper asks whether uncertainty detects divergence between proxy and gold rewards during ordinary PPO.

Protocol characteristics:

- synthetic gold/proxy RM setup,
- run PPO,
- store generated samples every 10 optimization steps,
- compute proxy reward, gold reward, and uncertainty for the same generations,
- compare lightweight CI uncertainty with ensemble uncertainty,
- plot rewards and uncertainty against optimization step,
- plot uncertainty against the proxy–gold reward discrepancy,
- quantify association using Pearson correlation.

Our analogue must run all uncertainty estimators on the **same stored PPO generations**. This isolates uncertainty quality from policy differences.

## 4.4 AdvPO Section 5.2 target

The paper asks whether AdvPO mitigates reward overoptimization.

Protocol characteristics:

- same synthetic gold/proxy setting as Section 5.1,
- compare standard PPO with AdvPO,
- plot proxy and gold reward against:
  - PPO step, and
  - \(\sqrt{D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{SFT}})}\),
- track uncertainty over training,
- use a severe overoptimization setting with KL coefficient \(\beta=0\),
- train for 1,500 PPO steps in the original setup,
- evaluate validation responses every 100 steps,
- tune \(B=b^2\) over \(\{1,5,10,15\}\).

The Coste adaptation must preserve the same conceptual comparison even if the primary Coste-native run uses different batch, rollout, sequence-length, or step settings.

---

# 5. Coste et al. environment to reuse

## 5.1 Released repository

```text
https://github.com/tlc4418/llm_optimization
```

During the 2026-08-23 audit, the repository’s `main` tree resolved to:

```text
416b03cc2c3c8125208679acd88891584d9eefd2
```

Codex must re-check the current state and pin an exact commit in the experiment manifest. Do not depend on a moving branch.

The repository already provides:

- SFT training,
- reward-model training,
- single-RM PPO,
- RM-ensemble PPO,
- WCO and UWO reward aggregation,
- AlpacaFarm loaders,
- proxy scoring,
- post-training gold scoring,
- evaluation outputs,
- Accelerate/DeepSpeed configuration.

## 5.2 Important existing files and extension seams

Inspect at minimum:

```text
README.md
pyproject.toml
configs/config_rm.yaml
configs/config_rl.yaml
configs/ppo_config.yaml

src/data_utils/README.md
src/data_utils/oa_custom_datasets/dataset_loader.py
src/data_utils/oa_custom_datasets/rank_datasets.py

src/reward_modeling/training/trainer_rm.py
src/reward_modeling/scoring/score.py
src/reward_modeling/scoring/ppo_reward_functions.py

src/ppo/trainer_rl.py
src/ppo/custom_helpers.py
src/ppo/custom_trlx_trainers/custom_accelerate_base_trainer.py
src/ppo/custom_trlx_trainers/custom_accelerate_ppo_trainer.py
```

Known seams from the source audit:

- `src/ppo/trainer_rl.py` prepares prompts, obtains `reward_fn`, invokes `trlx.train`, and triggers post-hoc gold scoring.
- `src/ppo/custom_helpers.py:get_reward_fn` selects single or ensemble reward logic.
- `src/reward_modeling/scoring/ppo_reward_functions.py:create_reward_fn` is the main reward callback factory.
- The reward callback receives batches of `samples`, `prompts`, and `outputs`.
- `src/reward_modeling/scoring/score.py:get_reward` handles proxy forward passes.
- The repository already computes ensemble variance and supports mean, WCO, and UWO.
- Gold scoring is intentionally performed after PPO to avoid keeping the 7B gold RM in memory.

These should be extended rather than replaced.

## 5.3 Released models

Use existing checkpoints where possible:

```text
tlc4418/pythia_1.4b_sft_policy
tlc4418/pythia_70m_sft
```

Before training a proxy RM from scratch, search Hugging Face and the repository documentation for a released checkpoint matching the paper’s proxy-RM setup. Record model revision and checksum.

## 5.4 Released preference dataset

Primary static preference dataset:

```text
tlc4418/1.4b-policy_preference_data_gold_labelled
```

Semantics:

- prompts from AlpacaFarm,
- two answers generated by the 1.4B SFT policy,
- preferences labeled by the AlpacaFarm human-preference reward model,
- intended for proxy-RM training.

Expected fields include:

```text
instruction
input
answers        # exactly two responses
preference     # index 0 or 1
```

The current Hugging Face viewer reports roughly 51k total examples and two splits, but code must not hard-code viewer counts. Download using an explicit revision and create a manifest containing:

- Hub dataset ID,
- revision/commit,
- file names,
- split names,
- row counts,
- schema,
- hashes,
- any local split generation,
- and filtering decisions.

## 5.5 PPO prompt data is separate from RM preference data

The Coste pipeline uses AlpacaFarm’s `unlabeled` split for RL prompts.

The static preference dataset does **not** replace PPO rollout data.

Correct data flow:

```text
Static Coste preference pairs
    -> proxy RM training
    -> uncertainty geometry/calibration

AlpacaFarm unlabeled prompts
    -> current policy generates responses online
    -> proxy/robust reward drives PPO
    -> stored generations are gold-scored offline
```

Do not train PPO by repeatedly treating the two static answers in the RM dataset as on-policy rollouts.

---

# 6. Replication/adaptation matrix

| Component | Original AdvPO Sections 5.1/5.2 | Coste source | Project decision |
|---|---|---|---|
| Task/data family | Anthropic HH and TL;DR | AlpacaFarm instruction following | Use AlpacaFarm; this is a protocol adaptation |
| Initial policy | SFT LLaMA-family policy | Pythia 1.4B SFT | Use released Coste policy |
| Trainable policy | PPO policy initialized from SFT | Existing Coste PPO policy | Reuse Coste PPO pipeline |
| Proxy RM | LLaMA 7B synthetic proxy | Coste Pythia proxy RM, preferably effective 44M first | Use same fixed proxy checkpoint for all methods |
| Gold RM | larger 13B synthetic gold RM | AlpacaFarm 7B `reward-model-human` | Use only for labels/offline evaluation |
| RM data | gold-relabelled preferences, with 30% noise | released 1.4B-policy preference pairs | Begin with clean released labels; add named noise tracks later |
| PPO prompts | second half of preference-data prompts | AlpacaFarm unlabeled prompts | Use Coste RL prompt loader |
| Section 5.1 logging | generations every 10 PPO steps | Coste eval defaults differ | Add configurable diagnostic logging every 10 steps |
| Gold scoring | synthetic gold RM | Coste post-hoc gold scorer | Keep offline/post-hoc design |
| CI geometry | individual chosen/rejected features | proxy RM can expose final features | Implement exact AdvPO baseline |
| AdvPO reference | chosen response | unlabeled PPO prompts lack a guaranteed chosen response | Cache one frozen-SFT response per prompt, pending confirmation |
| Policy-shift x-axis | step and \(\sqrt{\mathrm{KL}}\) | Coste logs KL from initial policy | Reuse and verify estimator |
| Proposed geometry | not in AdvPO | pairwise Coste preference responses | Use within-pair differences |
| Proposed calibration | not in AdvPO | held-out static preference data + current rollout features | Exact formula remains OPEN |
| PPO implementation | authors’ tokenwise PPO | Coste Open-Assistant/trlx PPO | Reuse Coste implementation; do not write PPO from scratch |

---

# 7. Reuse-first engineering strategy

## 7.1 Phase A: establish the original Coste pipeline

Before adding uncertainty code:

1. Fork or vendor the Coste repository.
2. Pin its commit.
3. Create a reproducible legacy environment.
4. Run the smallest available debug PPO job.
5. Run proxy scoring.
6. Run post-hoc gold scoring.
7. Verify evaluation files and KL fields.
8. Reproduce a short single-RM PPO trajectory.
9. Document any dependency repairs.

No proposed-method code should be blamed for failures in an unvalidated baseline stack.

## 7.2 Compatibility-first dependency policy

The Coste repository depends on older Open-Assistant and `trlx` components. Therefore:

- Do not run `pip install -U` indiscriminately.
- Pin Python, PyTorch, CUDA, Transformers, Datasets, Accelerate, DeepSpeed, Open-Assistant, AlpacaFarm, and `trlx`.
- Prefer a container or lock file.
- Record all dependency patches.
- Preserve a baseline environment in which the unmodified or minimally patched Coste code runs.

A current Hugging Face `trl` PPO implementation may be useful later, but its API and trainer semantics differ substantially from the legacy `trlx` stack. A modern port is a separate experiment and must first match baseline outputs on a smoke test. Never mix legacy and modernized trainer results in one comparison without a validation study.

## 7.3 Search for official AdvPO code before implementation

No clearly official public AdvPO implementation was located during this context audit. This may have changed.

Codex must search again using:

- paper title,
- author names,
- OpenReview,
- NeurIPS page,
- author GitHub profiles,
- and paper supplemental material.

When official code is found:

1. inspect its license,
2. pin its revision,
3. compare its equations and reward handling to the paper,
4. reuse AdvPO-specific code if it can be integrated safely,
5. do not replace the Coste data/PPO stack unless necessary.

When no official code exists, implement only the small AdvPO-specific delta on top of the Coste pipeline.

## 7.4 Recommended existing packages

Use established packages for standard functionality:

- `torch` for model inference, matrix accumulation, and linear solves,
- `transformers` for tokenizer/model loading,
- `datasets` for Hub data and split management,
- `accelerate` and `deepspeed` through the existing pipeline,
- `numpy`,
- `scipy.stats` for Pearson/Spearman statistics,
- `pandas` and/or `pyarrow` for sample-level logs,
- `pytest` for unit and integration tests,
- `safetensors` and `huggingface_hub` for pinned artifacts,
- optional Weights & Biases only when already supported and configured.

Do not force a generic conformal-prediction package if it cannot represent the exact policy-weighted/adaptive procedure. A small, transparent quantile/calibration module is acceptable when it uses tested numerical primitives and has strong unit tests. “Do not reinvent” means reusing validated infrastructure; it does not mean hiding a novel method inside an unsuitable abstraction.

---

# 8. Dataset and split policy

## 8.1 Required logical splits

Create an explicit split manifest with disjoint IDs for:

### `D_rm_train`

Used to fit the proxy RM.

### `D_rm_val`

Used for proxy-RM early stopping/model selection.

### `D_cal`

Held-out labeled preference pairs used for conformal calibration and calibration diagnostics.

### `D_rl_train_prompts`

AlpacaFarm unlabeled prompts used for online PPO rollouts.

### `D_rl_val_prompts`

Fixed prompts used for checkpoint/trajectory evaluation.

### `D_rl_test_prompts`

Fixed prompts used only for final gold evaluation.

The Coste loader’s historical splits may not map exactly to these logical roles. Build a deterministic manifest rather than relying on implicit slicing inside multiple scripts.

## 8.2 Leakage rules

- No preference pair may be in both `D_rm_train` and `D_cal`.
- No final test prompt may be used to tune \(B\), \(\alpha\), update cadence, or stopping.
- No current/future PPO generation may be inserted into calibration with a gold-derived score unless the run is explicitly labeled oracle.
- Policy-adaptation statistics must be computed causally: step \(t\) may use only the fixed calibration data and rollout information available no later than step \(t\).
- Prompt IDs, not raw text alone, should define disjointness.

## 8.3 Fairness across methods

All optimization methods must use:

- the same proxy RM checkpoint,
- the same initial policy checkpoint,
- the same reference-policy checkpoint,
- the same PPO prompt manifests,
- the same decoding hyperparameters,
- the same base PPO hyperparameters within a track,
- and matched random seeds where possible.

The proposed method uses `D_cal`, while ordinary PPO does not. Report this labeled-data requirement explicitly. Do not compensate by training a different proxy RM for PPO or AdvPO.

Recommended fairness design:

1. Train one proxy RM on `D_rm_train`.
2. Freeze it.
3. Use that exact checkpoint for every method.
4. Build AdvPO \(M_D\) from the examples actually used for RM training, matching its paper definition.
5. Build the proposed geometry from the predefined source split specified in the final method freeze.
6. Use `D_cal` only for the proposed conformal quantile/calibration operation and corresponding diagnostic baselines.
7. Include a calibration-size ablation later.

## 8.4 Noise tracks

Do not introduce label noise in the first engineering milestone.

After the clean pipeline is validated, add explicitly named tracks:

- `coste_noise25`: 25% label noise, matching Coste’s robustness study.
- `advpo_noise30`: 30% label noise, approximating the original AdvPO synthetic stress setup.

Specify whether noise affects:

- only `D_rm_train`,
- both RM training and calibration,
- or a separate noisy-calibration ablation.

Do not allow the proposed method to receive clean calibration labels while describing all methods simply as “25% noisy” without disclosing the asymmetry.

---

# 9. Reward-model feature extraction

This is a high-risk implementation detail and must be tested before any uncertainty experiment.

## 9.1 Required feature

Let

\[
h(x,y)
\]

denote the exact pooled representation consumed by the proxy RM’s scalar reward head.

Do not assume that this is automatically:

- the last token,
- the EOS token,
- the mean of tokens,
- or `hidden_states[-1][:,-1,:]`.

Inspect the custom `GPTNeoXRewardModel` and its pooling logic. Implement one of:

- an explicit model method returning both reward and pooled feature, or
- a forward hook on the scalar projection input.

The feature extractor must preserve the original reward path.

## 9.2 Mandatory identity test

For random and real batches, verify that the model’s scalar output is reconstructed from the extracted feature and reward head:

\[
\widehat r(x,y)
\approx
w^\top h(x,y)+c.
\]

Set a strict tolerance appropriate to dtype.

If the head has a bias:

- document it,
- ensure the AdvPO implementation matches the paper’s assumed linear form,
- note that the bias cancels in response-minus-reference or pairwise differences,
- do not silently discard it in individual reward logging.

## 9.3 Tokenization and pooling tests

Test:

- left/right padding behavior,
- last non-padding token,
- EOS-present and EOS-absent sequences,
- truncation at configured maximum length,
- batched versus unbatched equality,
- proxy reward equality before and after adding feature return,
- mixed response lengths.

## 9.4 Efficiency

Compute scalar reward and feature in the same frozen-RM forward pass. Cache features for:

- static RM/calibration pairs,
- fixed reference responses,
- fixed validation generations where applicable.

Online rollout features must be computed once per generated response.

---

# 10. AdvPO baseline implementation specification

## 10.1 Static geometry construction

For each preference example, obtain features for chosen and rejected responses. Accumulate

\[
M_D =
\lambda I
+
\sum_i h_i^c h_i^{c\top}
+
\sum_i h_i^r h_i^{r\top}.
\]

Implementation requirements:

- accumulate in a numerically stable dtype, preferably float64 for the Gram matrix when feasible,
- save \(\lambda\), dimension, data manifest hash, model revision, pooling definition, and feature dtype,
- use `torch.linalg.cholesky_ex` plus `torch.cholesky_solve`, or another documented stable solve,
- do not repeatedly call a dense matrix inverse,
- test positive definiteness and regularization fallback,
- cache the factorization.

## 10.2 CI diagnostic

For a rollout feature \(h\), calculate

\[
u_{\mathrm{CI}}(h)
=
\sqrt{h^\top M_D^{-1}h}.
\]

Keep the unscaled leverage-like quantity and the scaled value \(b\,u_{\mathrm{CI}}\) as separate logged fields.

This avoids confusing:

- geometry,
- uncertainty ordering,
- and the tuned confidence radius.

## 10.3 AdvPO batch reward

For a rollout batch with fixed reference features:

\[
g_t =
\frac{1}{m}\sum_{j=1}^{m} h(x_j,y_j)
-
\frac{1}{m}\sum_{j=1}^{m} h(x_j,y_j^{\mathrm{ref}}).
\]

Compute the exact Eq. (9) adjusted reward for each sample using the common batch \(g_t\). The implementation must:

- return one scalar reward per PPO sample,
- preserve the aggregate robust-objective identity,
- detach all proxy-RM features and geometry,
- avoid gradients through the RM,
- keep gradients only through PPO’s policy log-probability path as in the existing trainer,
- define behavior when \(g_t^\top M_D^{-1}g_t\) is numerically near zero,
- log the batch penalty and \(g_t\) norm.

## 10.4 AdvPO \(B\) protocol

The paper treats

\[
B=b^2
\]

as a hyperparameter and searches:

```text
B in {1, 5, 10, 15}
```

Do not describe this baseline as conformally calibrated.

Run the same grid or a predeclared scale-adjusted grid if feature scaling in the Coste RM makes the original values degenerate. Any scale adjustment must be justified using only training/validation data and reported.

## 10.5 Reward scaling

The AdvPO appendix reports that running-standard-deviation reward scaling was removed and that the post-subtraction reward was rescaled to the original reward magnitude.

The checked-in Coste PPO config contains `scale_reward: False`, but Codex must inspect the actual trainer semantics. Implement AdvPO reward rescaling only after reproducing the paper’s definition and testing it. Log:

- raw proxy reward,
- unscaled robust reward,
- rescaling factor,
- final PPO reward.

Do not apply an undocumented normalization that changes comparisons.

---

# 11. Proposed method: confirmed design

The working name is **Version C: policy-adaptive conformal uncertainty using preference-pair representation differences**.

## 11.1 Pairwise representation

For a labeled preference pair \(i\), orient responses consistently as preferred/chosen \(A\) and rejected \(B\), then define

\[
d_i = h_i^A-h_i^B.
\]

This is one within-example difference, not every pairwise combination among all examples.

For \(n\) preference examples:

- number of differences: \(n\),
- direct Gram accumulation cost: \(O(nd^2)\),
- not \(O(n^2)\) in dataset size.

## 11.2 Why pairwise differences are central

Reward-model preference training uses the Bradley–Terry likelihood through reward differences:

\[
\widehat r(x_i,y_i^A)-\widehat r(x_i,y_i^B)
=
w^\top(h_i^A-h_i^B)
=
w^\top d_i.
\]

Therefore, \(d_i\) directly represents the directions in feature space that constrain the reward projection under preference learning.

AdvPO’s \(M_D\) instead summarizes individual response features. It can detect whether a new individual feature is unsupported, but it does not directly encode the comparative direction on which a preference label depends.

The proposed geometry \(V\) is intended to align uncertainty with these preference-relevant directions.

## 11.3 Policy-shift adaptation

During PPO, construct the current rollout/reference pair representation:

\[
d_{t,j}
=
h(x_j,y_{t,j})
-
h(x_j,y_j^{\mathrm{ref}}).
\]

The current distribution of \(d_{t,j}\) changes as the policy changes. The fixed labeled calibration set remains the source of calibration scores, while current unlabeled rollout/reference features characterize policy shift.

The intended deployable method must not require current gold scores.

The conformal threshold may change with step or rollout batch:

\[
q_{\alpha,t}.
\]

Conceptually:

```text
fixed labeled calibration information
        +
current policy rollout/reference representation distribution
        ->
policy-adapted calibration threshold q_{alpha,t}
        ->
uncertainty/robust reward used at step t
```

## 11.4 Scientific comparison that isolates adaptation

At minimum implement two versions after the equations are frozen:

### Proposed-static

- same pairwise geometry,
- same nonconformity score,
- one threshold \(q_{\alpha,0}\) fit before PPO,
- threshold remains fixed.

### Proposed-adaptive

- identical except \(q_{\alpha,t}\) adapts using the predeclared policy-shift mechanism.

This comparison isolates the claimed contribution of policy adaptation. Comparing only proposed-adaptive against AdvPO would confound pairwise geometry, conformal calibration, and adaptation.

---

# 12. Proposed method freeze table

Codex may build interfaces and tests around these concepts, but must not finalize the proposed online reward path while an item required by that path remains **OPEN**.

| Item | Status | Current meaning |
|---|---|---|
| Proxy feature \(h(x,y)\) | **FROZEN** | Exact pooled feature consumed by scalar RM head |
| Pair orientation | **FROZEN** | Preferred/chosen minus rejected |
| Pairwise difference \(d_i=h_i^A-h_i^B\) | **FROZEN** | One difference per preference example |
| Use of fixed labeled calibration set | **FROZEN** | Calibration data is disjoint from RM training |
| Policy-adapted threshold \(q_{\alpha,t}\) | **FROZEN concept** | May change with rollout distribution |
| No online gold labels | **FROZEN** | Deployable adaptation uses no gold RM |
| Current rollout/reference differences \(d_{t,j}\) | **WORKING DECISION** | Recommended representation of policy shift |
| Exact geometry \(V\) | **OPEN** | Plain regularized Gram \(\lambda I+\sum d_id_i^\top\) versus Fisher/curvature-weighted form must be frozen |
| Data used to build \(V\) | **OPEN** | RM train pairs, calibration pairs, or both |
| Whether \(V\) is static or updated | **OPEN** | Current plan favors static geometry and adaptive calibration, but this is not final |
| Nonconformity score | **OPEN** | Exact scalar score and label dependence must be specified |
| Finite-sample conformal quantile | **OPEN** | Exact split-conformal or weighted-conformal convention |
| Policy-shift weighting | **OPEN** | Density ratio, kernel/local weighting, nearest-neighbor weighting, or another rule |
| Weight estimation features | **OPEN** | Raw \(d\), whitened \(V^{-1/2}d\), scalar leverage, or another representation |
| Adaptation unit | **OPEN** | Every rollout minibatch, PPO batch, fixed number of steps, or sliding window |
| Adaptation memory | **OPEN** | Current batch only, rolling window, or exponentially weighted history |
| Weight clipping/regularization | **OPEN** | Required for stable weighted conformal behavior |
| Exact calibrated uncertainty interval | **OPEN** | How \(q_{\alpha,t}\) combines with geometry |
| Proposed pessimistic reward | **OPEN** | Sample-wise lower confidence bound versus AdvPO-style batch confidence set or another objective |
| Reference-response role in proposed reward | **WORKING DECISION** | Use same fixed references as AdvPO for controlled comparison |
| Fallback under severe support failure | **OPEN** | Clipping, abstention, larger radius, or fallback to fixed calibration |
| Target \(\alpha\) values | **OPEN** | Predeclare, likely a small grid with one primary value |

Before implementing the proposed optimizer, create:

```text
docs/PROPOSED_METHOD_FROZEN.md
```

It must contain the exact equations for every OPEN row needed by the run. Both the code and this context file should point to that frozen document.

---

# 13. Experiment program

## Stage 0: source and environment audit

Deliverables:

```text
docs/SOURCE_AUDIT.md
docs/ENVIRONMENT_LOCK.md
artifacts/source_manifest.json
```

The audit must record:

- Coste repo revision,
- whether official AdvPO code exists,
- upstream Open-Assistant/trlx revisions,
- package versions,
- model revisions,
- dataset revisions,
- gold RM acquisition instructions,
- PPO config semantics,
- paper-versus-code differences,
- licenses,
- required patches.

No large GPU run is authorized before this audit.

## Stage 1: Coste single-RM PPO baseline

Goal: prove that the reused pipeline works.

Run:

- initial 1.4B SFT policy,
- one frozen proxy RM,
- AlpacaFarm unlabeled PPO prompts,
- standard PPO,
- offline gold scoring,
- proxy/gold/KL trajectory logging.

Success criteria:

- deterministic smoke test completes,
- eval generations are stored,
- proxy scores are present,
- gold scores can be added offline,
- KL to initial policy is present and understood,
- checkpoints resume correctly,
- no gold RM is loaded in the training process.

## Stage 2: AdvPO Section 5.1 analogue

Use one or more **standard PPO** runs. Do not train a different policy for each uncertainty estimator.

Every 10 PPO steps, on a fixed diagnostic prompt subset or the configured rollout sample:

- store prompt ID and response,
- store policy/checkpoint/step,
- proxy-score response,
- extract proxy feature,
- compute AdvPO CI,
- compute proposed-static score when available,
- compute proposed-adaptive score and \(q_{\alpha,t}\) when available,
- optionally compute Coste ensemble variance,
- gold-score the stored response offline.

Primary figures:

1. Gold reward, proxy reward, and uncertainty versus PPO step.
2. Uncertainty versus proxy–gold discrepancy.
3. Pearson and Spearman correlation.
4. Error stratified by uncertainty quantile.
5. \(q_{\alpha,t}\) and policy-shift statistics versus step.
6. Results versus \(\sqrt{\mathrm{KL}}\) as an additional shift axis.

All uncertainty estimators must be evaluated on identical stored samples.

## Stage 3: AdvPO Section 5.2 analogue

Compare optimization methods with the same initial policy, proxy RM, prompt splits, and base PPO settings:

- `ppo_single`
- `advpo_ci`
- `proposed_static`
- `proposed_adaptive`

Optional but strongly useful because code already exists:

- `coste_ensemble_mean`
- `coste_wco`
- `coste_uwo`
- `samplewise_ci_penalty` as an ablation, not as AdvPO

Primary figures:

1. Proxy reward vs PPO step.
2. Gold reward vs PPO step.
3. Proxy and gold reward vs \(\sqrt{\mathrm{KL}}\).
4. Average uncertainty vs PPO step.
5. Proposed \(q_{\alpha,t}\) vs PPO step.
6. Gold–proxy divergence vs PPO step.
7. Final and peak gold reward with confidence intervals.

## Stage 4: adaptation ablations

After the main method works:

- fixed versus adaptive \(q_\alpha\),
- individual-feature geometry versus pairwise geometry,
- calibration-size sensitivity,
- adaptation window/cadence,
- target \(\alpha\),
- clean versus noisy RM labels,
- optional oracle adaptation,
- optional ensemble uncertainty.

Do not run a large ablation grid before the primary comparison is stable.

---

# 14. Configuration tracks

Paper and checked-in code defaults differ. Do not silently combine them.

## 14.1 Track C: Coste-native primary track

Purpose: maximize reuse and establish the main result in the Coste environment.

Start from Coste paper/repository settings, auditing global-versus-device semantics:

- policy: Pythia 1.4B SFT,
- proxy RM: primary Coste single-RM configuration,
- optimizer LR around \(10^{-6}\),
- PPO epochs: 4,
- batch size: 32 in the Coste paper,
- training horizon: 3,000 steps in the Coste paper,
- generation: up to 256 new tokens,
- clip range and value clip: 0.2,
- GAE \(\lambda=0.95\),
- cosine schedule,
- fixed prompt manifests.

For the main overoptimization stress run, set KL coefficient to zero in a named config:

```text
coste_native_beta0
```

Also retain a small-KL control if compute permits:

```text
coste_native_beta_small
```

## 14.2 Track A: AdvPO-protocol stress track

Purpose: approximate the original AdvPO Section 5.2 hyperparameters while retaining Coste data/models.

Candidate settings:

- 1,500 PPO steps,
- LR \(10^{-6}\),
- batch size 64,
- context 2,048,
- max 512 generated tokens,
- value clip 0.2,
- KL coefficient \(\beta=0\),
- validation generation every 100 steps,
- diagnostic generation every 10 steps,
- AdvPO \(B\in\{1,5,10,15\}\).

Run only after Track C is stable. Report it as a separate adaptation track, not pooled with Track C.

## 14.3 Resolve paper-versus-code drift explicitly

The checked-in Coste `configs/ppo_config.yaml` observed during the audit contains values such as:

```text
num_rollouts: 4
chunk_size: 2
batch_size: 32
total_steps: 3000
```

The Coste paper reports larger rollout/chunk values. This could reflect:

- debug-like checked-in defaults,
- per-device versus global values,
- code changes after experiments,
- or documentation drift.

Codex must trace the values through `trlx` and distributed execution before changing them. Create a table:

```text
paper value | repository value | runtime global value | chosen value | justification
```

---

# 15. Metrics and evaluation definitions

## 15.1 Raw sample-level log

Store one row per generated response in Parquet or JSONL with at least:

```text
run_id
method
seed
ppo_step
checkpoint_id
prompt_id
prompt_text_hash
response_id
response_text
reference_response_id
reference_response_hash
proxy_rm_revision
gold_rm_revision
policy_revision
proxy_reward_raw
gold_reward_raw
ppo_reward_final
advpo_ci_unscaled
advpo_ci_scaled
proposed_score_raw
proposed_uncertainty
q_alpha_t
adaptation_weight_summary_id
batch_robust_penalty
kl_to_sft
sqrt_kl_to_sft
response_token_count
prompt_token_count
generation_seed
config_hash
```

Avoid storing only aggregate curves. Every paper figure must be regenerable from raw logs.

## 15.2 Reward-scale issue

The Coste proxy RM and AlpacaFarm gold RM may have different scalar scales. Therefore, a raw difference can be difficult to interpret.

Always retain raw rewards. For Section 5.1 discrepancy analysis, predeclare a scale-alignment rule using only a fixed pre-PPO validation anchor, for example:

- standardize each RM using its initial-policy validation mean and standard deviation, or
- fit a simple affine map from proxy to gold on a separate validation set.

Then report:

```text
raw proxy reward
raw gold reward
scale-aligned signed error
scale-aligned absolute error
```

Do not fit score alignment on final PPO test generations.

Also report rank-based metrics that are invariant to affine scale where appropriate.

## 15.3 Section 5.1 uncertainty quality

Primary:

- Pearson correlation between uncertainty and predeclared reward discrepancy,
- Spearman correlation,
- bootstrap confidence interval,
- discrepancy by uncertainty decile,
- mean gold reward by uncertainty decile,
- high-error detection AUROC/AUPRC using a predeclared error threshold,
- selective risk as high-uncertainty samples are removed.

For a conformal interval with a frozen formal definition, additionally report:

- empirical marginal coverage,
- weighted/current-policy coverage when meaningful,
- interval width or uncertainty magnitude,
- coverage by PPO step/KL bin,
- coverage under increasing policy shift.

Coverage is important but not sufficient. The central evaluation must connect adaptation to policy quality.

## 15.4 Section 5.2 policy quality

Report:

- gold reward at each evaluation step,
- proxy reward at each evaluation step,
- peak gold reward,
- terminal gold reward,
- gold reward at matched KL budgets,
- area under the gold-reward-versus-\(\sqrt{\mathrm{KL}}\) curve,
- maximum proxy–gold divergence,
- step/KL where gold reward begins sustained decline,
- average uncertainty,
- \(q_{\alpha,t}\),
- response length,
- PPO stability metrics.

Do not claim mitigation solely because the policy stays close to the SFT model; compare gold reward at matched KL.

## 15.5 KL definition

Verify the Coste implementation’s KL estimator and logging. The Coste paper reports a sampled approximation based on log-probability ratios, while AdvPO plots \(\sqrt{D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{SFT}})}\).

Log:

- the exact implemented estimator,
- token or sequence aggregation,
- batch averaging,
- whether values are sampled or exact under generated tokens.

Do not relabel an arbitrary trainer statistic as the paper’s KL without verification.

---

# 16. Statistical protocol

For final results:

- use at least 3 PPO seeds per primary method, preferably more when compute allows,
- use paired prompt manifests,
- keep the proxy RM checkpoint fixed for the main method comparison,
- later repeat with multiple proxy-RM seeds as a robustness study,
- bootstrap over prompts and/or seeds as appropriate,
- show individual-seed trajectories in addition to means,
- report uncertainty bands and exact seed count,
- predeclare the primary checkpoint-selection criterion.

Do not select the best method-specific checkpoint using final gold test reward.

Recommended separation:

- validation prompts: tuning \(B\), \(\alpha\), or stopping,
- test prompts: final gold comparison,
- diagnostic prompts: dense every-10-step Section 5.1 logging.

If reproducing a paper protocol that uses synthetic-gold validation for checkpoint selection, label that result as a synthetic-oracle validation protocol and also report a deployable selection rule where feasible.

---

# 17. Suggested code organization

Do not reorganize the whole Coste repository. Prefer additive modules:

```text
src/
  uncertainty/
    __init__.py
    feature_extraction.py
    linear_algebra.py
    advpo_geometry.py
    advpo_objective.py
    pairwise_geometry.py
    conformal_base.py
    conformal_static.py
    conformal_policy_adaptive.py
    state.py

  ppo/
    reward_adapters.py
    reference_cache.py
    uncertainty_logging.py

scripts/
  audit_assets.py
  build_data_manifest.py
  train_or_load_proxy_rm.py
  cache_rm_features.py
  cache_reference_responses.py
  build_advpo_geometry.py
  build_pairwise_geometry.py
  run_section51_diagnostics.py
  run_section52_methods.py
  gold_score_rollouts.py
  make_section51_figures.py
  make_section52_figures.py

configs/
  experiments/
    coste_native_beta0.yaml
    advpo_stress_beta0.yaml
    methods/
      ppo_single.yaml
      advpo_ci.yaml
      proposed_static.yaml
      proposed_adaptive.yaml

tests/
  test_feature_extraction.py
  test_advpo_geometry.py
  test_advpo_objective.py
  test_pairwise_geometry.py
  test_conformal_quantile.py
  test_policy_adaptation.py
  test_no_gold_leakage.py
  test_reward_adapter_smoke.py
```

Exact paths may change after repository inspection. Preserve original public interfaces when possible.

## 17.1 Recommended interface boundaries

A reward-method adapter should conceptually expose:

```python
class RewardMethod:
    def prepare_static_state(...): ...
    def begin_rollout_batch(...): ...
    def score_batch(...): ...
    def diagnostics(...): ...
    def state_dict(...): ...
    def load_state_dict(...): ...
```

Method-independent PPO code should provide:

- prompts,
- generated outputs,
- proxy rewards/features,
- fixed reference rewards/features,
- step/batch metadata.

Method-specific code should return:

- final reward per sample,
- method diagnostics,
- serializable adaptation state.

The gold RM should not be an argument to this interface.

---

# 18. Numerical implementation requirements

## 18.1 Avoid explicit inverse

For any geometry \(A\), compute

\[
x^\top A^{-1}x
\]

through a solve:

```python
L, info = torch.linalg.cholesky_ex(A)
z = torch.cholesky_solve(x[..., None], L).squeeze(-1)
quad = (x * z).sum(-1)
```

Handle factorization failure by increasing a logged regularizer according to a deterministic rule.

## 18.2 Cache and provenance

Every geometry/calibrator file must include:

```text
schema_version
method_name
proxy_model_id
proxy_model_revision
reward_head_hash
tokenizer_id
tokenizer_revision
pooling_definition
dataset_id
dataset_revision
split_manifest_hash
feature_dimension
feature_dtype
accumulation_dtype
regularization
normalization
creation_code_commit
```

Fail fast on provenance mismatch.

## 18.3 Batch/distributed correctness

The batch statistic \(g_t\) and any policy-adaptation summary may need a global batch across distributed workers.

Codex must determine whether to:

- all-reduce sufficient statistics,
- operate per rank,
- or operate per PPO minibatch.

This choice changes the method. Use global rollout-batch statistics when the paper/method defines a global batch expectation, unless memory/communication constraints require a documented approximation.

Unit-test one-process versus simulated multi-rank aggregation.

## 18.4 Checkpoint adaptation state

A resumed run must restore:

- current \(q_{\alpha,t}\),
- adaptation window/history,
- weight-estimator state,
- reference-cache version,
- geometry factorization metadata,
- logging step.

A resumed trajectory should match an uninterrupted deterministic smoke run within expected nondeterminism.

---

# 19. Mandatory tests

## 19.1 Feature tests

- extracted feature reconstructs RM logit,
- padding and EOS handling,
- batch equivalence,
- no change to legacy reward output.

## 19.2 AdvPO geometry tests

- \(M_D\) is symmetric,
- regularized \(M_D\) is positive definite,
- Cholesky solve matches direct inverse on tiny synthetic matrices,
- uncertainty decreases when a direction is repeatedly represented in training features,
- cached and on-the-fly calculations agree.

## 19.3 AdvPO objective tests

- per-sample Eq. (9) mean equals aggregate robust objective,
- \(B\to 0\) approaches reference-centered proxy reward,
- identical rollout and reference features yield near-zero robust displacement,
- no NaN when \(g\) is near zero,
- reward scaling is reproducible.

## 19.4 Pairwise geometry tests

- one \(d_i\) per preference example,
- chosen-minus-rejected orientation is consistent,
- swapping labels and responses preserves the correctly reoriented \(d_i\),
- no accidental all-pairs \(O(n^2)\) computation,
- expected PSD properties,
- feature differences match reward-logit differences through the head.

## 19.5 Conformal tests

After the method is frozen:

- exact finite-sample quantile index,
- deterministic handling of ties,
- weighted quantile normalization,
- clipping and zero-weight behavior,
- expected coverage on exchangeable synthetic data,
- behavior under controlled covariate shift,
- fixed method leaves \(q_\alpha\) unchanged,
- adaptive method changes \(q_{\alpha,t}\) under synthetic shift,
- adaptive method remains stable under no shift,
- no use of current gold labels.

## 19.6 No-gold-leakage test

Make the online gold scorer raise an exception when called. A complete PPO smoke step for every deployable method must pass.

## 19.7 End-to-end smoke test

Use tiny models/data and a handful of PPO steps to test:

- baseline PPO,
- AdvPO,
- proposed-static,
- proposed-adaptive,
- logging,
- checkpoint/resume,
- offline gold scoring.

---

# 20. Experimental acceptance gates

## Gate 1: baseline integrity

Do not interpret proposed-method results until:

- the Coste PPO baseline runs,
- proxy and gold scoring are correct,
- KL logging is verified,
- the proxy feature identity test passes,
- and raw generations are saved.

## Gate 2: overoptimization exists in at least one predeclared stress track

The experiment should exhibit a region where proxy reward continues to improve while gold reward plateaus or declines.

When it does not:

1. do not tune only the proposed method,
2. inspect horizon, KL coefficient, label noise, RM capacity, rollout settings, and reward scaling,
3. use a small predeclared stress grid shared by all methods,
4. document negative results.

## Gate 3: exact AdvPO reproduction within the adapted environment

Before comparing to the proposed method:

- CI geometry matches the paper,
- Section 5.1 diagnostics are reproducible,
- AdvPO batch objective passes tests,
- \(B\) tuning is separated from conformal calibration.

## Gate 4: proposed method is mathematically frozen

All required OPEN rows in Section 12 are resolved in `docs/PROPOSED_METHOD_FROZEN.md`.

## Gate 5: fair final comparison

- identical initial policy/proxy RM/prompt manifests,
- no gold leakage,
- at least 3 seeds,
- raw logs,
- confidence intervals,
- checkpoint selection declared,
- figures reproducible from scripts.

---

# 21. Common failure modes to prevent

1. **Treating the Coste preference dataset as PPO rollouts.**  
   It is static RM/calibration data; PPO must generate current responses.

2. **Using three “policy models.”**  
   There is one trainable PPO policy plus its frozen SFT reference. Proxy and gold RMs are scorers, not policies.

3. **Training with the gold RM.**  
   Gold is offline evaluation only.

4. **Calling sample-wise CI penalty “AdvPO.”**  
   AdvPO is batch/distributionally coupled through the confidence set and reference responses.

5. **Building \(V\) with all pairs of all examples.**  
   Use one within-preference difference per example.

6. **Updating \(q_{\alpha,t}\) with gold rollout errors.**  
   That is an oracle method, not the deployable proposed method.

7. **Letting each method use a different proxy RM.**  
   Main comparison must share the same frozen checkpoint.

8. **Modernizing PPO before reproducing the baseline.**  
   First use the original Coste stack.

9. **Ignoring reward-scale mismatch in Section 5.1.**  
   Preserve raw scores and predeclare scale alignment.

10. **Explicitly inverting large matrices.**  
    Use stable solves and cache factorizations.

11. **Using future rollouts in current adaptation.**  
    Adaptation must be causal.

12. **Selecting checkpoints on final gold test reward.**  
    Keep validation and test roles separate.

13. **Conflating calibration coverage with policy improvement.**  
    Report both, but the central result is adaptation under policy shift and resulting gold policy quality.

14. **Silently changing global batch semantics.**  
    Audit per-device/global rollout and minibatch values.

15. **Failing to store generations.**  
    Sample-level logs are necessary for post-hoc gold scoring and diagnostic fairness.

---

# 22. First task for Codex

Use this as the first Codex prompt after placing this file at the repository root:

```text
Read CODEX_CONTEXT.md completely.

Do not implement the proposed method or start a large training run yet.

1. Inspect the current repository.
2. Inspect the upstream tlc4418/llm_optimization repository and pin the
   exact revision being used.
3. Search for an official AdvPO implementation or supplemental code.
4. Trace the full Coste PPO path:
   data loader -> prompt formatting -> policy rollout -> proxy reward
   callback -> PPO update -> evaluation serialization -> offline gold score.
5. Identify the exact tensor consumed by the proxy reward scalar head and
   explain how to expose it without changing reward values.
6. Audit paper-versus-code PPO hyperparameters, including global batch and
   rollout semantics.
7. Identify the minimum extension points for:
   a. AdvPO geometry and Eq. (9) reward,
   b. pairwise feature geometry,
   c. conformal calibration,
   d. policy-adaptive q_alpha,t,
   e. dense Section 5.1 logging.
8. Do not choose any equation marked OPEN in CODEX_CONTEXT.md.

Create:
- docs/SOURCE_AUDIT.md
- docs/IMPLEMENTATION_PLAN.md
- docs/OPEN_METHOD_DECISIONS.md
- artifacts/source_manifest.json

For each proposed code change, list:
- file path,
- existing upstream component reused,
- responsibility,
- inputs and outputs,
- equation implemented,
- test,
- expected computational/memory cost,
- and whether it changes baseline behavior.

Then implement only:
- environment/setup fixes needed for a tiny baseline smoke run,
- a read-only asset audit,
- and tests proving that the legacy reward path remains unchanged.

Stop before implementing proposed-method equations.
```

---

# 23. Recommended implementation order after the audit

1. Freeze and pin environment.
2. Run unmodified/minimally patched Coste debug PPO.
3. Verify offline gold scoring.
4. Add exact RM feature extraction plus identity tests.
5. Add sample-level rollout serialization.
6. Build/cache AdvPO \(M_D\).
7. Add Section 5.1 CI diagnostics on stored PPO outputs.
8. Implement and test AdvPO Eq. (9).
9. Run PPO vs AdvPO smoke comparison.
10. Freeze `PROPOSED_METHOD_FROZEN.md`.
11. Implement pairwise geometry.
12. Implement static conformal method.
13. Implement policy-adapted calibration.
14. Run all methods on the same tiny end-to-end setup.
15. Run Coste-native primary track.
16. Run AdvPO-stress track if needed.
17. Run final seeds.
18. Generate figures from immutable raw logs.
19. Add noise and adaptation ablations only after the primary result.

---

# 24. Open research/experiment decisions for the project owner

These require explicit confirmation before the corresponding full experiment.

## Decision A: exact proposed geometry

Is the intended geometry exactly

\[
V = \lambda I+\sum_i d_i d_i^\top,
\]

or does it include Bradley–Terry/Fisher weights such as a function of the fitted preference margin?

## Decision B: exact nonconformity score

Specify the scalar calibration score, including:

- whether it predicts reward error, preference margin error, or a parameter-space residual,
- whether it uses a label,
- and how \(V\) enters.

## Decision C: policy-shift adaptation rule

Specify precisely how current rollout/reference differences alter calibration:

- importance weights,
- local neighborhoods,
- kernel weights,
- density ratio,
- or another construction.

## Decision D: weighted quantile and guarantee

Specify:

- target \(\alpha\),
- finite-sample correction,
- normalization,
- treatment of a test-point weight if applicable,
- assumptions under which coverage is claimed.

## Decision E: update cadence and memory

Choose:

- every rollout batch,
- every PPO step,
- every \(k\) steps,
- and current-batch versus rolling-window state.

## Decision F: proposed robust policy objective

Choose whether calibrated uncertainty is used as:

- a sample-wise pessimistic reward,
- an AdvPO-style batch confidence-set radius,
- a constraint,
- a gating/abstention rule,
- or another exact objective.

## Decision G: reference responses

Confirm the recommended frozen-SFT reference generation for AlpacaFarm unlabeled PPO prompts, including decoding settings.

## Decision H: primary experiment track

Recommended primary:

```text
Coste-native assets and trainer
+ clean proxy-RM labels
+ beta = 0 overoptimization stress setting
+ PPO / AdvPO / proposed-static / proposed-adaptive
```

Confirm whether the initial full run should instead use 25% label noise or the closer AdvPO 30% noise setting.

---

# 25. Source-of-truth references

## AdvPO

- NeurIPS abstract:  
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/94bbcb744bbada8808fda05b9d9290d6-Abstract-Conference.html
- Official PDF:  
  https://proceedings.neurips.cc/paper_files/paper/2024/file/94bbcb744bbada8808fda05b9d9290d6-Paper-Conference.pdf
- OpenReview ID:  
  https://openreview.net/forum?id=kYio3xH6eb
- arXiv lineage:  
  https://arxiv.org/abs/2403.05171

Relevant paper locations:

- Section 3.2: CI uncertainty and \(M_D\)
- Section 4: AdvPO objective, reference responses, Eq. (8), Eq. (9)
- Section 5.1: uncertainty diagnostic
- Section 5.2: overoptimization mitigation
- Appendix A.2/A.3: reward scaling, \(B\) tuning, synthetic setup, PPO details
- Appendix C.1: Pearson correlation table

## Coste et al.

- ICLR abstract:  
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/dda7f9378a210c25e470e19304cce85d-Abstract-Conference.html
- arXiv:  
  https://arxiv.org/abs/2310.02743
- Code:  
  https://github.com/tlc4418/llm_optimization
- Preference dataset:  
  https://huggingface.co/datasets/tlc4418/1.4b-policy_preference_data_gold_labelled
- Initial policy:  
  https://huggingface.co/tlc4418/pythia_1.4b_sft_policy
- Proxy-RM SFT base:  
  https://huggingface.co/tlc4418/pythia_70m_sft

## Upstream infrastructure

- AlpacaFarm:  
  https://github.com/tatsu-lab/alpaca_farm
- Open-Assistant:  
  https://github.com/LAION-AI/Open-Assistant
- trlx:  
  https://github.com/CarperAI/trlx
- Hugging Face Accelerate:  
  https://github.com/huggingface/accelerate
- DeepSpeed:  
  https://github.com/microsoft/DeepSpeed
- Hugging Face Datasets:  
  https://github.com/huggingface/datasets

---

# 26. Final implementation principle

The desired contribution is not a new monolithic RLHF framework.

The desired contribution is a carefully controlled addition to a validated synthetic-RM overoptimization pipeline:

```text
Coste data/models/PPO/gold evaluation
             +
exact AdvPO 5.1/5.2 baseline
             +
pairwise conformal uncertainty
             +
policy-adapted calibration
             =
controlled evidence about uncertainty under PPO policy shift
```

When existing code can perform a task, use it. When a new component is scientifically necessary, keep it modular, mathematically explicit, independently tested, and auditable.
