#!/usr/bin/env bash
# End-to-end CPDPOv2 smoke: cache two SFT anchors, then train one rollout.
# Requires the existing two-prompt smoke schedule and a matching v1 artifact set.
#   sbatch scripts/slurm/smoke_cpdpo_v2.sbatch

#SBATCH -J cpdpo_v2_smoke
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-cpdpo-v2-smoke-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
SCHEDULE="${CPDPO_SMOKE_SCHEDULE:-artifacts/prompt_schedules/smoke_seed_1.jsonl}"
CPDPO_ALPHA="${CPDPO_ALPHA:-0.10}"
PROXY_BATCH_SIZE="${CPDPO_PROXY_BATCH_SIZE:-64}"
if [[ ! "$PROXY_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CPDPO_PROXY_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
source scripts/configure_policy_variant.sh
if [[ -n "${CPDPO_V2_SMOKE_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$CPDPO_V2_SMOKE_OUTPUT_ROOT"
elif [[ "$POLICY_VARIANT" == "70m" ]]; then
  OUTPUT_ROOT="$JOB_STORAGE_ROOT/outputs/reward_overoptimization_v2_smoke_policy_70m"
else
  OUTPUT_ROOT="$JOB_STORAGE_ROOT/outputs/reward_overoptimization_v2_smoke"
fi
if [[ -n "${CPDPO_V2_SMOKE_REFERENCE_DIR:-}" ]]; then
  REFERENCE_DIR="$CPDPO_V2_SMOKE_REFERENCE_DIR"
elif [[ "$POLICY_VARIANT" == "70m" ]]; then
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/cpdpo_v2_smoke/policy_70m/seed_1"
else
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/cpdpo_v2_smoke/seed_1"
fi
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ALPHA_TAG="$(python -c 'import sys; from src.cpdpo.spec import alpha_tag; print(alpha_tag(float(sys.argv[1])))' "$CPDPO_ALPHA")"
if [[ -n "${CPDPO_ARTIFACT_DIR:-}" ]]; then
  ARTIFACT_DIR="$CPDPO_ARTIFACT_DIR"
elif [[ "$ALPHA_TAG" == "0p1" ]]; then
  ARTIFACT_DIR="artifacts/cpdpo/proxy_rm_seed1"
else
  ARTIFACT_DIR="artifacts/cpdpo/proxy_rm_seed1_alpha_${ALPHA_TAG}"
fi
ALLOW_ARGS=()
if [[ "${CPDPO_ALLOW_SMOKE_ARTIFACTS:-0}" == "1" ]]; then
  ALLOW_ARGS+=(--allow-smoke-artifacts)
fi

python scripts/validate_legacy_sources.py
python -m pip check
python scripts/validate_policy_variant.py --variant "$POLICY_VARIANT" --policy-model "$POLICY_PATH"
test -f "$SCHEDULE"
test -d "$POLICY_PATH"
test -d "$PROXY_RM_PATH"

srun python scripts/prepare_cpdpo_v2_reference_cache.py \
  --prompt-schedule "$SCHEDULE" \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --policy-model "$POLICY_PATH" \
  --policy-variant "$POLICY_VARIANT" \
  "${POLICY_TRAINING_ARGS[@]}" \
  --proxy-rm "$PROXY_RM_PATH" \
  --geometry "$ARTIFACT_DIR/pair_geometry.pt" \
  --calibration "$ARTIFACT_DIR/conformal_calibration.json" \
  --output-dir "$REFERENCE_DIR" \
  --base-seed 1 \
  --alpha "$CPDPO_ALPHA" \
  --batch-size 2 \
  --proxy-batch-size "$PROXY_BATCH_SIZE" \
  --device cuda:0 \
  --dtype bf16 \
  "${ALLOW_ARGS[@]}"

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_reward_overoptimization.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual prompt_disjoint_data_split_v1 \
  --method cpdpo_v2 \
  --prompt-schedule "$SCHEDULE" \
  --rollout-steps 1 \
  --prompts-per-rollout 2 \
  --checkpoint-every-rollouts 1 \
  --pair-batch-size 1 \
  --pair-chunk-size 4 \
  --proxy-batch-size "$PROXY_BATCH_SIZE" \
  --base-seed 1 \
  --policy-model "$POLICY_PATH" \
  --policy-variant "$POLICY_VARIANT" \
  --proxy-rm "$PROXY_RM_PATH" \
  --geometry "$ARTIFACT_DIR/pair_geometry.pt" \
  --calibration "$ARTIFACT_DIR/conformal_calibration.json" \
  --reference-cache "$REFERENCE_DIR/reference_cache.pt" \
  --alpha "$CPDPO_ALPHA" \
  --output-root "$OUTPUT_ROOT" \
  "${ALLOW_ARGS[@]}"

RUN_VARIANT="$(python -c 'import sys; from src.cpdpo.spec import method_run_name; print(method_run_name("cpdpo_v2", float(sys.argv[1])))' "$CPDPO_ALPHA")"
test -f "$OUTPUT_ROOT/seed_1/$RUN_VARIANT/run_metadata.json"
test -f "$OUTPUT_ROOT/seed_1/$RUN_VARIANT/rollout_metrics.jsonl"
test -d "$OUTPUT_ROOT/seed_1/$RUN_VARIANT/model"
echo "PASS CPDPOv2 end-to-end smoke policy=$POLICY_VARIANT alpha=$CPDPO_ALPHA"
