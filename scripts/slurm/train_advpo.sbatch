#!/usr/bin/env bash
# Train only paper-equation AdvPO. B is required and becomes part of the path.
#   sbatch --array=1 --export=ALL,ROLLOUT_STEPS=100,ADVPO_B=1 scripts/slurm/train_advpo.sbatch

#SBATCH -J advpo
#SBATCH --array=1-3
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-advpo-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail
: "${ROLLOUT_STEPS:?Export a common positive ROLLOUT_STEPS value}"
: "${ADVPO_B:?Export AdvPO B=b^2; the paper grid is 1,5,10,15}"
: "${ADVPO_RIDGE_LAMBDA:?Export the ridge used to build the confidence artifact}"
if [[ ! "$ROLLOUT_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ROLLOUT_STEPS must be a positive integer" >&2
  exit 2
fi

SEED="${SLURM_ARRAY_TASK_ID:?missing seed array task ID}"
PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
SCHEDULE="artifacts/prompt_schedules/seed_${SEED}.jsonl"
RIDGE_LAMBDA="$ADVPO_RIDGE_LAMBDA"
PROXY_BATCH_SIZE="${CPDPO_PROXY_BATCH_SIZE:-64}"
if [[ ! "$PROXY_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CPDPO_PROXY_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
source scripts/configure_policy_variant.sh
if [[ -n "${CPDPO_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$CPDPO_OUTPUT_ROOT"
elif [[ "$POLICY_VARIANT" == "70m" ]]; then
  OUTPUT_ROOT="$JOB_STORAGE_ROOT/outputs/reward_overoptimization_policy_70m"
else
  OUTPUT_ROOT="$JOB_STORAGE_ROOT/outputs/reward_overoptimization"
fi
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RIDGE_TAG="$(python -c 'import sys; from src.advpo.spec import number_tag; print(number_tag(float(sys.argv[1])))' "$RIDGE_LAMBDA")"
CONFIDENCE_DIR="${ADVPO_CONFIDENCE_DIR:-$JOB_STORAGE_ROOT/artifacts/advpo/proxy_rm_seed1_ridge_${RIDGE_TAG}}"
if [[ -n "${ADVPO_REFERENCE_DIR:-}" ]]; then
  REFERENCE_DIR="$ADVPO_REFERENCE_DIR"
elif [[ "$POLICY_VARIANT" == "70m" ]]; then
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/advpo/references/policy_70m_${POLICY_PRECISION}/seed_${SEED}"
else
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/advpo/references/seed_${SEED}"
fi

python scripts/validate_legacy_sources.py
python -m pip check
python scripts/validate_policy_variant.py --variant "$POLICY_VARIANT" --policy-model "$POLICY_PATH"
test -f "$SCHEDULE"
test -f "$CONFIDENCE_DIR/confidence_matrix.pt"
test -f "$CONFIDENCE_DIR/confidence_matrix_metadata.json"
test -f "$REFERENCE_DIR/reference_cache.pt"
test -f "$REFERENCE_DIR/reference_cache_metadata.json"
test -f "$REFERENCE_DIR/reference_responses.jsonl"

EXTRA_ARGS=()
if [[ -n "${ADVPO_RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$ADVPO_RESUME_CHECKPOINT")
fi

srun accelerate launch --config_file "$POLICY_ACCELERATE_CONFIG" \
  src/ppo/trainer_reward_overoptimization.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual prompt_disjoint_data_split_v1 \
  --method advpo \
  --prompt-schedule "$SCHEDULE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --prompts-per-rollout 256 \
  --checkpoint-every-rollouts 10 \
  --pair-batch-size 32 \
  --pair-chunk-size 64 \
  --proxy-batch-size "$PROXY_BATCH_SIZE" \
  --base-seed "$SEED" \
  --policy-model "$POLICY_PATH" \
  --policy-variant "$POLICY_VARIANT" \
  "${POLICY_TRAINING_ARGS[@]}" \
  --proxy-rm "$PROXY_RM_PATH" \
  --advpo-confidence "$CONFIDENCE_DIR/confidence_matrix.pt" \
  --reference-cache "$REFERENCE_DIR/reference_cache.pt" \
  --advpo-B "$ADVPO_B" \
  --output-root "$OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}"

echo "PASS method=advpo seed=$SEED policy=$POLICY_VARIANT rollouts=$ROLLOUT_STEPS B=$ADVPO_B ridge=$RIDGE_LAMBDA"
