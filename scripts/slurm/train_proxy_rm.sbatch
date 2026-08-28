#!/usr/bin/env bash
# Train one manifest-backed 44M proxy reward model on a Phoenix GPU node.
#
# Submit from the repository root with:
#   sbatch --account="$PACE_ACCOUNT" scripts/slurm/train_proxy_rm.sbatch
#
# Override RM_SEED for one seed, or submit an array for seeds 1--5:
#   sbatch --account="$PACE_ACCOUNT" --array=1-5 scripts/slurm/train_proxy_rm.sbatch

#SBATCH -J proxy_rm_pythia_44m
#SBATCH --account=gts-yxie77-paid
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this file with sbatch; do not run it on a login node." >&2
  exit 2
fi

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
RM_SEED="${SLURM_ARRAY_TASK_ID:-${RM_SEED:-1}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
OUTPUT_DIR="models/rm-pythia-44m-prompt-disjoint_seed${RM_SEED}"

if [[ ! "$RM_SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: RM_SEED/SLURM_ARRAY_TASK_ID must be a positive integer: $RM_SEED" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
if [[ ! -f pyproject.toml || ! -f scripts/configure_cluster_storage.sh ]]; then
  echo "ERROR: RLHF_PROJECT_ROOT/SLURM_SUBMIT_DIR is not the repository root: $PROJECT_ROOT" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not on PATH. Activate rlhf-cuq before sbatch, or load the site's Conda module." >&2
  exit 2
fi
# Conda activation hooks may read unset variables (for example,
# MKL_INTERFACE_LAYER), so temporarily disable Bash nounset.
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u

# Keep build, Hugging Face, and temporary caches off quota-limited home storage.
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Repository: $PROJECT_ROOT"
echo "Git revision: $(git rev-parse HEAD)"
echo "Conda environment: $CONDA_PREFIX"
echo "Scratch/cache root: $JOB_STORAGE_ROOT"
echo "RM seed: $RM_SEED"
echo "Expected output: $OUTPUT_DIR"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported(), 'The checked RM configuration requires BF16 support'"

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: output directory is not empty: $OUTPUT_DIR" >&2
  echo "Move it aside or select a new seed; this script will not overwrite a prior run." >&2
  exit 2
fi

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/reward_modeling/training/trainer_rm_manifest.py \
  --configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-split \
  --rng_seed "$RM_SEED"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "ERROR: training returned success but the expected output is missing: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p artifacts/checksums
find "$OUTPUT_DIR" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "artifacts/checksums/rm-pythia-44m-prompt-disjoint_seed${RM_SEED}.sha256"

echo "PASS proxy RM seed $RM_SEED"
echo "Checkpoint: $OUTPUT_DIR"
echo "Checksums: artifacts/checksums/rm-pythia-44m-prompt-disjoint_seed${RM_SEED}.sha256"
