#!/usr/bin/env bash
# Source this after changing to the repository root. It resolves only the
# initial/reference causal LM. Proxy-RM selection remains a separate setting.

POLICY_VARIANT="${RLHF_POLICY_VARIANT:-1p4b}"
case "$POLICY_VARIANT" in
  1p4b)
    POLICY_DEFAULT_PATH="assets/initial_sft_policy"
    POLICY_OUTPUT_TAG="policy_1p4b"
    ;;
  70m)
    POLICY_DEFAULT_PATH="assets/proxy_rm_sft_base"
    POLICY_OUTPUT_TAG="policy_70m"
    ;;
  *)
    echo "ERROR: RLHF_POLICY_VARIANT must be exactly '1p4b' or '70m'; found '$POLICY_VARIANT'" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

POLICY_PATH="${CPDPO_POLICY_PATH:-$POLICY_DEFAULT_PATH}"
export POLICY_VARIANT POLICY_PATH POLICY_OUTPUT_TAG

# Optional, explicitly recorded optimization overrides. Keeping these empty
# preserves the audited 1.4B settings. They are useful for a separately named
# 70M stabilization profile after the literal 1.4B profile proved unstable.
POLICY_TRAINING_ARGS=()
if [[ -n "${RLHF_POLICY_LEARNING_RATE:-}" ]]; then
  POLICY_TRAINING_ARGS+=(--optimizer-learning-rate "$RLHF_POLICY_LEARNING_RATE")
fi
if [[ -n "${RLHF_POLICY_NUM_LAYERS_UNFROZEN:-}" ]]; then
  POLICY_TRAINING_ARGS+=(--num-layers-unfrozen "$RLHF_POLICY_NUM_LAYERS_UNFROZEN")
fi
if [[ -n "${RLHF_POLICY_MAX_GRAD_NORM:-}" ]]; then
  POLICY_TRAINING_ARGS+=(--max-grad-norm "$RLHF_POLICY_MAX_GRAD_NORM")
fi
