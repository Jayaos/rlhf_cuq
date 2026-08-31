#!/usr/bin/env bash
# Offline gold-isolated evaluation for a named AdvPO B run.
#   sbatch --array=1 --export=ALL,GOLD_RM_PATH=/path/to/gold,ADVPO_B=1 scripts/slurm/evaluate_advpo.sbatch

#SBATCH -J advpo_eval
#SBATCH --array=1-3
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-advpo-eval-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail
: "${GOLD_RM_PATH:?Export the reconstructed AlpacaFarm GOLD_RM_PATH}"
: "${ADVPO_B:?Export the B value used by the AdvPO run}"

SEED="${SLURM_ARRAY_TASK_ID:?missing seed array task ID}"
PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
POLICY_PATH="${CPDPO_POLICY_PATH:-assets/initial_sft_policy}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
OUTPUT_ROOT="${CPDPO_OUTPUT_ROOT:-$JOB_STORAGE_ROOT/outputs/reward_overoptimization}"
EVAL_SPLIT="${CPDPO_EVAL_SPLIT:-D_rl_val_prompts}"
NUM_PROMPTS="${CPDPO_EVAL_NUM_PROMPTS:-0}"

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_VARIANT="$(python -c 'import sys; from src.advpo.spec import advpo_run_name; print(advpo_run_name(float(sys.argv[1])))' "$ADVPO_B")"
RUN_DIR="$OUTPUT_ROOT/seed_${SEED}/$RUN_VARIANT"
python scripts/validate_legacy_sources.py
test -f "$RUN_DIR/run_metadata.json"
test -d "$GOLD_RM_PATH"

srun python scripts/evaluate_policy_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --initial-policy "$POLICY_PATH" \
  --reference-policy "$POLICY_PATH" \
  --proxy-rm "$PROXY_RM_PATH" \
  --gold-rm "$GOLD_RM_PATH" \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --split "$EVAL_SPLIT" \
  --num-prompts "$NUM_PROMPTS" \
  --device cuda:0 \
  --dtype bf16

echo "PASS offline evaluation method=advpo seed=$SEED split=$EVAL_SPLIT B=$ADVPO_B"
