#!/usr/bin/env bash
# Build the frozen CPDPO geometry/calibration once from the full proxy RM.
# Submit from the repository root:
#   sbatch scripts/slurm/prepare_cpdpo_artifacts.sbatch

#SBATCH -J cpdpo_artifacts
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-cpdpo-artifacts-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
MANIFEST_PATH="${CPDPO_MANIFEST_PATH:-data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json}"
ARTIFACT_DIR="${CPDPO_ARTIFACT_DIR:-artifacts/cpdpo/proxy_rm_seed1}"

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/validate_legacy_sources.py
python scripts/build_data_manifest.py --config configs/data_split_prompt_disjoint_v1.yaml --verify-only
test -d "$PROXY_RM_PATH"
if [[ -d "$ARTIFACT_DIR" ]] && [[ -n "$(find "$ARTIFACT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: CPDPO artifact directory is not empty: $ARTIFACT_DIR" >&2
  exit 2
fi

srun python scripts/prepare_cpdpo_artifacts.py \
  --manifest "$MANIFEST_PATH" \
  --proxy-rm "$PROXY_RM_PATH" \
  --output-dir "$ARTIFACT_DIR" \
  --device cuda:0 \
  --batch-size 64

echo "PASS CPDPO offline artifacts: $ARTIFACT_DIR"
