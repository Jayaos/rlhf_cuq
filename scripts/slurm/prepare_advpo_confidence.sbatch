#!/usr/bin/env bash
# Build the shared full AdvPO M_D confidence matrix.
#   sbatch --export=ALL,ADVPO_RIDGE_LAMBDA=1.0 scripts/slurm/prepare_advpo_confidence.sbatch

#SBATCH -J advpo_conf
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-advpo-confidence-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail
: "${ADVPO_RIDGE_LAMBDA:?Export an explicit ridge; the paper does not disclose one}"

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
RIDGE_LAMBDA="$ADVPO_RIDGE_LAMBDA"
ARTIFACT_BATCH_SIZE="${ADVPO_ARTIFACT_BATCH_SIZE:-64}"
if [[ ! "$ARTIFACT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ADVPO_ARTIFACT_BATCH_SIZE must be a positive integer" >&2
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

RIDGE_TAG="$(python -c 'import sys; from src.advpo.spec import number_tag; print(number_tag(float(sys.argv[1])))' "$RIDGE_LAMBDA")"
CONFIDENCE_DIR="${ADVPO_CONFIDENCE_DIR:-$JOB_STORAGE_ROOT/artifacts/advpo/proxy_rm_seed1_ridge_${RIDGE_TAG}}"

python scripts/validate_legacy_sources.py
python -m pip check
test -d "$PROXY_RM_PATH"

srun python scripts/prepare_advpo_confidence.py \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --proxy-rm "$PROXY_RM_PATH" \
  --output-dir "$CONFIDENCE_DIR" \
  --ridge-lambda "$RIDGE_LAMBDA" \
  --batch-size "$ARTIFACT_BATCH_SIZE" \
  --device cuda:0

echo "PASS AdvPO confidence ridge_lambda=$RIDGE_LAMBDA path=$CONFIDENCE_DIR"
