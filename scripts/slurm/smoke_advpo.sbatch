#!/usr/bin/env bash
# End-to-end AdvPO smoke: reduced confidence, two SFT refs, one PPO rollout.
#   sbatch scripts/slurm/smoke_advpo.sbatch

#SBATCH -J advpo_smoke
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-advpo-smoke-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
SCHEDULE="${CPDPO_SMOKE_SCHEDULE:-artifacts/prompt_schedules/smoke_seed_1.jsonl}"
ADVPO_B="${ADVPO_B:-1.0}"
RIDGE_LAMBDA="${ADVPO_RIDGE_LAMBDA:-1.0}"
ARTIFACT_BATCH_SIZE="${ADVPO_ARTIFACT_BATCH_SIZE:-64}"
PROXY_BATCH_SIZE="${CPDPO_PROXY_BATCH_SIZE:-64}"
if [[ ! "$ARTIFACT_BATCH_SIZE" =~ ^[1-9][0-9]*$ || ! "$PROXY_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ADVPO_ARTIFACT_BATCH_SIZE and CPDPO_PROXY_BATCH_SIZE must be positive integers" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
source scripts/configure_policy_variant.sh
if [[ -n "${ADVPO_SMOKE_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$ADVPO_SMOKE_OUTPUT_ROOT"
elif [[ "$POLICY_VARIANT" == "70m" ]]; then
  OUTPUT_ROOT="$JOB_STORAGE_ROOT/outputs/advpo_smoke_policy_70m"
else
  OUTPUT_ROOT="$JOB_STORAGE_ROOT/outputs/advpo_smoke"
fi
if [[ -n "${ADVPO_SMOKE_CONFIDENCE_DIR:-}" ]]; then
  CONFIDENCE_DIR="$ADVPO_SMOKE_CONFIDENCE_DIR"
elif [[ "$POLICY_VARIANT" == "70m" ]]; then
  CONFIDENCE_DIR="$JOB_STORAGE_ROOT/artifacts/advpo_smoke/policy_70m/confidence"
else
  CONFIDENCE_DIR="$JOB_STORAGE_ROOT/artifacts/advpo_smoke/confidence"
fi
if [[ -n "${ADVPO_SMOKE_REFERENCE_DIR:-}" ]]; then
  REFERENCE_DIR="$ADVPO_SMOKE_REFERENCE_DIR"
elif [[ "$POLICY_VARIANT" == "70m" ]]; then
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/advpo_smoke/policy_70m_${POLICY_PRECISION}/references_seed_1"
else
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/advpo_smoke/references_seed_1"
fi
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/validate_legacy_sources.py
python -m pip check
python scripts/validate_policy_variant.py --variant "$POLICY_VARIANT" --policy-model "$POLICY_PATH"
test -f "$SCHEDULE"
test -d "$POLICY_PATH"
test -d "$PROXY_RM_PATH"

srun python scripts/prepare_advpo_confidence.py \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --proxy-rm "$PROXY_RM_PATH" \
  --output-dir "$CONFIDENCE_DIR" \
  --ridge-lambda "$RIDGE_LAMBDA" \
  --batch-size "$ARTIFACT_BATCH_SIZE" \
  --max-rm-pairs 64 \
  --device cuda:0

srun python scripts/prepare_advpo_references.py \
  --prompt-schedule "$SCHEDULE" \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --policy-model "$POLICY_PATH" \
  --policy-variant "$POLICY_VARIANT" \
  --proxy-rm "$PROXY_RM_PATH" \
  --output-dir "$REFERENCE_DIR" \
  --base-seed 1 \
  --batch-size 2 \
  --proxy-batch-size "$PROXY_BATCH_SIZE" \
  --artifact-scope smoke \
  --device cuda:0 \
  --dtype "$POLICY_PRECISION"

srun accelerate launch --config_file "$POLICY_ACCELERATE_CONFIG" \
  src/ppo/trainer_reward_overoptimization.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual prompt_disjoint_data_split_v1 \
  --method advpo \
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
  "${POLICY_TRAINING_ARGS[@]}" \
  --proxy-rm "$PROXY_RM_PATH" \
  --advpo-confidence "$CONFIDENCE_DIR/confidence_matrix.pt" \
  --reference-cache "$REFERENCE_DIR/reference_cache.pt" \
  --advpo-B "$ADVPO_B" \
  --output-root "$OUTPUT_ROOT" \
  --allow-smoke-artifacts

RUN_VARIANT="$(python -c 'import sys; from src.advpo.spec import advpo_run_name; print(advpo_run_name(float(sys.argv[1])))' "$ADVPO_B")"
test -f "$OUTPUT_ROOT/seed_1/$RUN_VARIANT/run_metadata.json"
test -f "$OUTPUT_ROOT/seed_1/$RUN_VARIANT/rollout_metrics.jsonl"
test -d "$OUTPUT_ROOT/seed_1/$RUN_VARIANT/model"
echo "PASS AdvPO end-to-end smoke policy=$POLICY_VARIANT B=$ADVPO_B ridge=$RIDGE_LAMBDA"
