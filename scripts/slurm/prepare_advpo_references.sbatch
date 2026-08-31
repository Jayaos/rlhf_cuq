#!/usr/bin/env bash
# Prepare fixed SFT references for one full AdvPO seed schedule.
#   sbatch --export=ALL,ADVPO_SEED=1 scripts/slurm/prepare_advpo_references.sbatch

#SBATCH -J advpo_refs
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-advpo-refs-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
SEED="${ADVPO_SEED:-1}"
POLICY_PATH="${CPDPO_POLICY_PATH:-assets/initial_sft_policy}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
SCHEDULE="${ADVPO_SCHEDULE:-artifacts/prompt_schedules/seed_${SEED}.jsonl}"
REFERENCE_DIR="${ADVPO_REFERENCE_DIR:-$JOB_STORAGE_ROOT/artifacts/advpo/references/seed_${SEED}}"
GENERATION_BATCH_SIZE="${ADVPO_GENERATION_BATCH_SIZE:-16}"
PROXY_BATCH_SIZE="${CPDPO_PROXY_BATCH_SIZE:-64}"
if [[ ! "$GENERATION_BATCH_SIZE" =~ ^[1-9][0-9]*$ || ! "$PROXY_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ADVPO_GENERATION_BATCH_SIZE and CPDPO_PROXY_BATCH_SIZE must be positive integers" >&2
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

python scripts/validate_legacy_sources.py
python -m pip check
test -f "$SCHEDULE"
test -d "$POLICY_PATH"
test -d "$PROXY_RM_PATH"

srun python scripts/prepare_advpo_references.py \
  --prompt-schedule "$SCHEDULE" \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --policy-model "$POLICY_PATH" \
  --proxy-rm "$PROXY_RM_PATH" \
  --output-dir "$REFERENCE_DIR" \
  --base-seed "$SEED" \
  --batch-size "$GENERATION_BATCH_SIZE" \
  --proxy-batch-size "$PROXY_BATCH_SIZE" \
  --device cuda:0 \
  --dtype bf16

echo "PASS AdvPO references seed=$SEED path=$REFERENCE_DIR"
