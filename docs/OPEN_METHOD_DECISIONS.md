# CPDPO decision status

The 2026-08-27 specifications replace the earlier AdvPO/adaptive-conformal
decision table. No equation needed by the v1 CPDPO online reward path is open.

## Frozen for v1

- exact scalar-head input feature;
- source-order pair difference `a - b`;
- full float64 normalized Gram geometry on `D_rm_train`;
- trace-scaled ridge with the specified zero-trace fallback;
- Cholesky uncertainty solve;
- hinge-like misranking calibration score;
- alpha `0.10`, epsilon `1e-8`, and finite-sample higher order statistic;
- one fixed threshold for the entire PPO run;
- exactly two stochastic responses per scheduled prompt;
- PairPPO reward `R=m`;
- CPDPO strict certification and robust-margin reward;
- no GAE or value loss for pair methods;
- response-token sum followed by pair averaging;
- separate common KL term, with beta `0.0` for the primary stress track;
- comparison of PPO, PairPPO, and CPDPO only;
- gold evaluation in a separate process on persisted common responses.

`alpha=0.10` remains frozen for the v1 main result. A user-declared alternate
alpha is a named sensitivity ablation, not a replacement main configuration.
Each such run must rebuild the finite-sample calibration threshold before
training, keep it fixed throughout training, use an alpha-specific artifact
and run identity, and be compared against the unchanged PPO/PairPPO controls.
Gold reward cannot be used to select alpha.

## Configurable, not scientifically unresolved

- Total rollout horizon. The experiment PDF requires it to be equal across
  methods but does not prescribe a number. Full launch commands must set it.
- Checkpoint interval. Default example: every 10 rollouts; it must remain equal
  across methods.
- Number of seeds. Minimum 3, preferred 5.
- A later nonzero common KL beta track.
- Predeclared CPDPO alpha sensitivity values, provided they are reported as
  ablations and never selected using gold reward.

## Out of scope for v1

- AdvPO;
- adaptive or rollout-weighted conformal thresholds;
- online geometry updates;
- fallback scalar rewards for uncertified pairs;
- best-of-K or resampling until certification;
- a learned pairwise critic or GAE over `+/-R`;
- gold-driven selection, calibration, early stopping, or optimization.

AdvPO remains out of scope for the frozen v1 comparison. The user's later
2026-08-30 request authorizes it only as an additive, separately named
comparison branch.

## Frozen for the additive CPDPOv2 exploratory track

- method identity `cpdpo_v2`; v1 `cpdpo` remains unchanged;
- one stochastic SFT response cached once per unique scheduled prompt;
- reference-generation seed `base_seed + 40000` and rollout generation settings;
- two current-policy responses and 512 trainable trajectories per main rollout;
- the same full v1 `V`, fixed `q_alpha`, and alpha identity;
- directed orientation `current - fixed_SFT`;
- continuous `R_v2 = m_ref - q_alpha*u_ref`, including negative values;
- ordinary scalar PPO value head, GAE, and clipped loss;
- no ratio, advantage, gradient, or regeneration for the cached response;
- no additional reward normalization in the first v2 diagnostic;
- separate v2 artifact/run/evaluation names and gold isolation.

Applying source-pair calibration to current/SFT pairs assumes exchangeability.
Report v2 as an exploratory robust proxy method, not a gold-reward guarantee.

## Frozen for the additive AdvPO branch

- paper Eq. (4) confidence matrix from individual `D_rm_train` response
  features, with no `1/n` normalization and no CPDPO pair differences;
- paper Eq. (6)--(7) fixed-reference max-min objective and its one shared
  batch-level adversarial projection direction;
- fixed SFT-generated references, which the paper explicitly permits and
  which fit the prompt-disjoint AlpacaFarm PPO prompts;
- ordinary scalar PPO value loss, GAE, and clipped loss;
- no conformal calibration, certification gate, pair loss, or gold access;
- one declared `B=b^2` per named run; the paper's reported grid is
  `[1, 5, 10, 15]`;
- dynamic scaling that restores the adversarial reward's running mean to the
  original proxy reward running mean, with the factor persisted in metrics;
- separate confidence/reference artifacts, run identity, smoke/full jobs,
  evaluation provenance, and optional plots.

The paper does not publish a numerical `ridge_lambda` for `M_D` or authors'
code. `ridge_lambda` is therefore an explicit recorded experiment parameter.
There is no default for a scientific artifact: the launch requires a declared
value, changing it creates a different artifact fingerprint, and the selected
value must be reported. The smoke-only template explicitly uses `1.0` and does
not freeze that value for a scientific run.
