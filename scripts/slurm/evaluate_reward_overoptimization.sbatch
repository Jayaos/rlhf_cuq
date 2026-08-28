#!/usr/bin/env bash
# Offline common-response evaluation for three methods x three seeds.
# GOLD_RM_PATH must name a licensed, reconstructed, validated AlpacaFarm
# reward-model-human checkpoint.
#   sbatch --export=ALL,GOLD_RM_PATH=/path/to/reward-model-human --array=0-8 scripts/slurm/evaluate_reward_overoptimization.sbatch

#SBATCH -J reward_eval
#SBATCH --array=0-8
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-reward-eval-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail
: "${GOLD_RM_PATH:?Export the reconstructed AlpacaFarm GOLD_RM_PATH}"

METHODS=(ppo pairppo cpdpo)
METHOD="${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]}"
SEED="$((SLURM_ARRAY_TASK_ID / 3 + 1))"
PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
POLICY_PATH="${CPDPO_POLICY_PATH:-assets/initial_sft_policy}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
OUTPUT_ROOT="${CPDPO_OUTPUT_ROOT:-outputs/reward_overoptimization}"
EVAL_SPLIT="${CPDPO_EVAL_SPLIT:-D_rl_val_prompts}"
NUM_PROMPTS="${CPDPO_EVAL_NUM_PROMPTS:-0}"
RUN_DIR="$OUTPUT_ROOT/seed_${SEED}/$METHOD"

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

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

echo "PASS offline evaluation method=$METHOD seed=$SEED split=$EVAL_SPLIT"
