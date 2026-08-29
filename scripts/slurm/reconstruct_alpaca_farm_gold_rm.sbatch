#!/usr/bin/env bash
# Reconstruct the licensed AlpacaFarm SFT10k and reward-model-human locally.
# Submit only after authorized LLaMA-v1 use has been confirmed:
#   sbatch --export=ALL,GOLD_LLAMA_LICENSE_ACKNOWLEDGED=yes \
#     scripts/slurm/reconstruct_alpaca_farm_gold_rm.sbatch

#SBATCH -J gold_rm_recover
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-gold-rm-recover-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
GOLD_ASSET_ROOT="${GOLD_ASSET_ROOT:-${JOB_STORAGE_ROOT}/gold-assets}"
LLAMA7B_PATH="${GOLD_LLAMA7B_PATH:-${GOLD_ASSET_ROOT}/llama-7b-hf}"
SFT_DIFF_PATH="${GOLD_SFT10K_DIFF_PATH:-${GOLD_ASSET_ROOT}/alpaca_farm_sft10k_weight_diff}"
GOLD_DIFF_PATH="${GOLD_RM_DIFF_PATH:-${GOLD_ASSET_ROOT}/alpaca_farm_gold_rm_weight_diff}"
MODELS_ROOT="${GOLD_MODELS_ROOT:-${JOB_STORAGE_ROOT}/alpaca_farm_models}"

if [[ "${GOLD_LLAMA_LICENSE_ACKNOWLEDGED:-}" != "yes" ]]; then
  echo "ERROR: set GOLD_LLAMA_LICENSE_ACKNOWLEDGED=yes only after confirming authorized LLaMA-v1 use." >&2
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
test -d "$LLAMA7B_PATH"
test -d "$SFT_DIFF_PATH"
test -d "$GOLD_DIFF_PATH"
python scripts/download_assets.py \
  --asset-root "$GOLD_ASSET_ROOT" \
  --asset alpaca_farm_sft10k_weight_diff \
  --asset alpaca_farm_gold_rm_weight_diff \
  --verify-only

srun python scripts/reconstruct_alpaca_farm_gold_rm.py \
  --llama-7b-hf-dir "$LLAMA7B_PATH" \
  --sft10k-weight-diff "$SFT_DIFF_PATH" \
  --gold-rm-weight-diff "$GOLD_DIFF_PATH" \
  --output-dir "$MODELS_ROOT" \
  --device cuda:0 \
  --llama-license-acknowledged

echo "PASS offline gold-RM reconstruction: $MODELS_ROOT/reward-model-human"
