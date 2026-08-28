#!/usr/bin/env bash
# Three methods x three seeds. ROLLOUT_STEPS is deliberately required because
# the revised experiment specification does not freeze a numerical horizon.
# Example after materializing seed_1..seed_3 schedules:
#   sbatch --export=ALL,ROLLOUT_STEPS=100 --array=0-8 scripts/slurm/train_reward_overoptimization.sbatch

#SBATCH -J reward_overopt
#SBATCH --array=0-8
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-reward-overopt-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

: "${ROLLOUT_STEPS:?Export a common positive ROLLOUT_STEPS value when submitting}"
if [[ ! "$ROLLOUT_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ROLLOUT_STEPS must be a positive integer" >&2
  exit 2
fi
METHODS=(ppo pairppo cpdpo)
METHOD="${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]}"
SEED="$((SLURM_ARRAY_TASK_ID / 3 + 1))"
PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
POLICY_PATH="${CPDPO_POLICY_PATH:-assets/initial_sft_policy}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
SCHEDULE="artifacts/prompt_schedules/seed_${SEED}.jsonl"
ARTIFACT_DIR="${CPDPO_ARTIFACT_DIR:-artifacts/cpdpo/proxy_rm_seed1}"
OUTPUT_ROOT="${CPDPO_OUTPUT_ROOT:-outputs/reward_overoptimization}"

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/validate_legacy_sources.py
python -m pip check
test -f "$SCHEDULE"
test -d "$POLICY_PATH"
test -d "$PROXY_RM_PATH"
EXTRA_ARGS=()
if [[ "$METHOD" == "cpdpo" ]]; then
  EXTRA_ARGS+=(--geometry "$ARTIFACT_DIR/pair_geometry.pt")
  EXTRA_ARGS+=(--calibration "$ARTIFACT_DIR/conformal_calibration.json")
fi
if [[ -n "${CPDPO_RESUME_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$CPDPO_RESUME_CHECKPOINT")
fi

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_reward_overoptimization.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual prompt_disjoint_data_split_v1 \
  --method "$METHOD" \
  --prompt-schedule "$SCHEDULE" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --prompts-per-rollout 256 \
  --checkpoint-every-rollouts 10 \
  --pair-batch-size 32 \
  --pair-chunk-size 64 \
  --base-seed "$SEED" \
  --policy-model "$POLICY_PATH" \
  --proxy-rm "$PROXY_RM_PATH" \
  --output-root "$OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}"

echo "PASS method=$METHOD seed=$SEED rollouts=$ROLLOUT_STEPS"
