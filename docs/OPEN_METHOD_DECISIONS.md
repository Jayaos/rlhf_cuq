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

## Configurable, not scientifically unresolved

- Total rollout horizon. The experiment PDF requires it to be equal across
  methods but does not prescribe a number. Full launch commands must set it.
- Checkpoint interval. Default example: every 10 rollouts; it must remain equal
  across methods.
- Number of seeds. Minimum 3, preferred 5.
- A later nonzero common KL beta track.

## Out of scope for v1

- AdvPO;
- adaptive or rollout-weighted conformal thresholds;
- online geometry updates;
- fallback scalar rewards for uncertified pairs;
- best-of-K or resampling until certification;
- a learned pairwise critic or GAE over `+/-R`;
- gold-driven selection, calibration, early stopping, or optimization.
