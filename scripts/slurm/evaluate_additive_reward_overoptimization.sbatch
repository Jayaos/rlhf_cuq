#!/usr/bin/env bash
# Gold-isolated evaluation array for the additive CPDPOv2 and AdvPO tracks.
# Mapping: task 0=CPDPOv2 seed 1, 1=AdvPO seed 1, 2=CPDPOv2 seed 2,
# 3=AdvPO seed 2, 4=CPDPOv2 seed 3, 5=AdvPO seed 3.
# For the current seed-1 diagnostic:
#   sbatch --array=0-1 --export=ALL,GOLD_RM_PATH=/path/to/gold,ADVPO_B=1 \
#     scripts/slurm/evaluate_additive_reward_overoptimization.sbatch

#SBATCH -J additive_eval
#SBATCH --array=0-5
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-additive-reward-eval-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail
: "${GOLD_RM_PATH:?Export the reconstructed AlpacaFarm GOLD_RM_PATH}"

METHODS=(cpdpo_v2 advpo)
TASK_ID="${SLURM_ARRAY_TASK_ID:?missing array task ID}"
METHOD="${METHODS[$((TASK_ID % 2))]}"
SEED="$((TASK_ID / 2 + 1))"
PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
EVAL_SPLIT="${CPDPO_EVAL_SPLIT:-D_rl_val_prompts}"
NUM_PROMPTS="${CPDPO_EVAL_NUM_PROMPTS:-0}"
CPDPO_ALPHA="${CPDPO_ALPHA:-0.10}"
ADVPO_B="${ADVPO_B:-}"
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
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ "$METHOD" == "cpdpo_v2" ]]; then
  RUN_VARIANT="$(python -c 'import sys; from src.cpdpo.spec import method_run_name; print(method_run_name("cpdpo_v2", float(sys.argv[1])))' "$CPDPO_ALPHA")"
  METHOD_DETAIL="alpha=$CPDPO_ALPHA"
else
  if [[ -z "$ADVPO_B" ]]; then
    echo "ERROR: Export ADVPO_B for AdvPO evaluation array tasks" >&2
    exit 2
  fi
  RUN_VARIANT="$(python -c 'import sys; from src.advpo.spec import advpo_run_name; print(advpo_run_name(float(sys.argv[1])))' "$ADVPO_B")"
  METHOD_DETAIL="B=$ADVPO_B"
fi
RUN_DIR="$OUTPUT_ROOT/seed_${SEED}/$RUN_VARIANT"

python scripts/validate_legacy_sources.py
python scripts/validate_policy_variant.py --variant "$POLICY_VARIANT" --policy-model "$POLICY_PATH"
test -f "$RUN_DIR/run_metadata.json"
test -d "$GOLD_RM_PATH"

srun python scripts/evaluate_policy_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --initial-policy "$POLICY_PATH" \
  --reference-policy "$POLICY_PATH" \
  --policy-variant "$POLICY_VARIANT" \
  --proxy-rm "$PROXY_RM_PATH" \
  --gold-rm "$GOLD_RM_PATH" \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --split "$EVAL_SPLIT" \
  --num-prompts "$NUM_PROMPTS" \
  --proxy-batch-size "$PROXY_BATCH_SIZE" \
  --device cuda:0 \
  --dtype bf16

echo "PASS additive offline evaluation method=$METHOD seed=$SEED policy=$POLICY_VARIANT split=$EVAL_SPLIT $METHOD_DETAIL"
