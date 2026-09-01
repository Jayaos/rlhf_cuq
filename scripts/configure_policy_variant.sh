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
