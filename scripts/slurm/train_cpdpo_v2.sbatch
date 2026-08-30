#!/usr/bin/env bash
# Train only the additive CPDPOv2 branch. Override the array for a seed-1 run:
#   sbatch --array=1 --export=ALL,ROLLOUT_STEPS=100 scripts/slurm/train_cpdpo_v2.sbatch

#SBATCH -J cpdpo_v2
#SBATCH --array=1-3
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-cpdpo-v2-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail
: "${ROLLOUT_STEPS:?Export a common positive ROLLOUT_STEPS value when submitting}"
if [[ ! "$ROLLOUT_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ROLLOUT_STEPS must be a positive integer" >&2
  exit 2
fi

SEED="${SLURM_ARRAY_TASK_ID:?missing seed array task ID}"
PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
POLICY_PATH="${CPDPO_POLICY_PATH:-assets/initial_sft_policy}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
SCHEDULE="artifacts/prompt_schedules/seed_${SEED}.jsonl"
OUTPUT_ROOT="${CPDPO_OUTPUT_ROOT:-$JOB_STORAGE_ROOT/outputs/reward_overoptimization}"
CPDPO_ALPHA="${CPDPO_ALPHA:-0.10}"

cd "$PROJECT_ROOT"
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
if [[ -n "${CPDPO_V2_REFERENCE_DIR:-}" ]]; then
  REFERENCE_DIR="$CPDPO_V2_REFERENCE_DIR"
elif [[ "$ALPHA_TAG" == "0p1" ]]; then
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/cpdpo_v2/seed_${SEED}"
else
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/cpdpo_v2/seed_${SEED}_alpha_${ALPHA_TAG}"
fi

python scripts/validate_legacy_sources.py
python -m pip check
test -f "$SCHEDULE"
test -f "$REFERENCE_DIR/reference_cache.pt"
test -f "$REFERENCE_DIR/reference_cache_metadata.json"
test -f "$REFERENCE_DIR/reference_responses.jsonl"

EXTRA_ARGS=()
if [[ -n "${CPDPO_RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$CPDPO_RESUME_CHECKPOINT")
fi

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_reward_overoptimization.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual prompt_disjoint_data_split_v1 \
  --method cpdpo_v2 \
  --prompt-schedule "$SCHEDULE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --prompts-per-rollout 256 \
  --checkpoint-every-rollouts 10 \
  --pair-batch-size 32 \
  --pair-chunk-size 64 \
  --base-seed "$SEED" \
  --policy-model "$POLICY_PATH" \
  --proxy-rm "$PROXY_RM_PATH" \
  --geometry "$ARTIFACT_DIR/pair_geometry.pt" \
  --calibration "$ARTIFACT_DIR/conformal_calibration.json" \
  --reference-cache "$REFERENCE_DIR/reference_cache.pt" \
  --alpha "$CPDPO_ALPHA" \
  --output-root "$OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}"

echo "PASS method=cpdpo_v2 seed=$SEED rollouts=$ROLLOUT_STEPS alpha=$CPDPO_ALPHA"
