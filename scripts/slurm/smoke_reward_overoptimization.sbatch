#!/usr/bin/env bash
# One-rollout smoke matrix. Build the shared schedule and CPDPO artifacts first.
#   python scripts/build_prompt_schedule.py --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json --output artifacts/prompt_schedules/smoke_seed_1.jsonl --base-seed 1 --rollout-steps 1 --prompts-per-rollout 2
#   sbatch --array=0-2 scripts/slurm/smoke_reward_overoptimization.sbatch

#SBATCH -J cpdpo_smoke
#SBATCH --array=0-2
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-cpdpo-smoke-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

METHODS=(ppo pairppo cpdpo)
METHOD="${METHODS[${SLURM_ARRAY_TASK_ID:?missing array task ID}]}"
PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
POLICY_PATH="${CPDPO_POLICY_PATH:-assets/initial_sft_policy}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
SCHEDULE="${CPDPO_SMOKE_SCHEDULE:-artifacts/prompt_schedules/smoke_seed_1.jsonl}"
OUTPUT_ROOT="${CPDPO_SMOKE_OUTPUT_ROOT:-outputs/reward_overoptimization_smoke}"
CPDPO_ALPHA="${CPDPO_ALPHA:-0.10}"
PROXY_BATCH_SIZE="${CPDPO_PROXY_BATCH_SIZE:-64}"
if [[ ! "$PROXY_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CPDPO_PROXY_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

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
RUN_VARIANT="$(python -c 'import sys; from src.cpdpo.spec import method_run_name; print(method_run_name(sys.argv[1], float(sys.argv[2])))' "$METHOD" "$CPDPO_ALPHA")"

python scripts/validate_legacy_sources.py
python -m pip check
test -f "$SCHEDULE"
test -d "$POLICY_PATH"
test -d "$PROXY_RM_PATH"

EXTRA_ARGS=()
if [[ "$METHOD" == "cpdpo" ]]; then
  EXTRA_ARGS+=(--geometry "$ARTIFACT_DIR/pair_geometry.pt")
  EXTRA_ARGS+=(--calibration "$ARTIFACT_DIR/conformal_calibration.json")
  EXTRA_ARGS+=(--alpha "$CPDPO_ALPHA")
  # The smoke path may deliberately use reduced-data artifacts. Full training
  # never passes this opt-in and therefore rejects them.
  EXTRA_ARGS+=(--allow-smoke-artifacts)
fi

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_reward_overoptimization.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual prompt_disjoint_data_split_v1 \
  --method "$METHOD" \
  --prompt-schedule "$SCHEDULE" \
  --rollout-steps 1 \
  --prompts-per-rollout 2 \
  --checkpoint-every-rollouts 1 \
  --pair-batch-size 1 \
  --pair-chunk-size 4 \
  --proxy-batch-size "$PROXY_BATCH_SIZE" \
  --base-seed 1 \
  --policy-model "$POLICY_PATH" \
  --proxy-rm "$PROXY_RM_PATH" \
  --output-root "$OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}"

test -f "$OUTPUT_ROOT/seed_1/$RUN_VARIANT/run_metadata.json"
test -d "$OUTPUT_ROOT/seed_1/$RUN_VARIANT/model"
echo "PASS $METHOD one-rollout experiment smoke cpdpo_alpha=$CPDPO_ALPHA"
