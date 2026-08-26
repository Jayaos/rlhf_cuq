#!/usr/bin/env bash
# Fast non-scientific proxy-RM smoke compatible with V100/RTX6000/A100 GPUs.
# It uses FP16 and standard attention; submit from the repository root with:
#   sbatch scripts/slurm/smoke_proxy_rm_any_gpu.sbatch

#SBATCH -J proxy_rm_44m_smoke_fp16
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t30
#SBATCH --gres=gpu:1
#SBATCH -qinferno
#SBATCH -o Report-smoke-fp16-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this file with sbatch; do not run it on a login node." >&2
  exit 2
fi

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
RM_SEED="${RM_SEED:-1}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
OUTPUT_DIR="models/rm-pythia-44m-prompt-disjoint-smoke-fp16_seed${RM_SEED}"
CHECKSUM_FILE="artifacts/checksums/rm-pythia-44m-prompt-disjoint-smoke-fp16_seed${RM_SEED}.sha256"

if [[ ! "$RM_SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: RM_SEED must be a positive integer: $RM_SEED" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
if [[ ! -f pyproject.toml || ! -f scripts/configure_cluster_storage.sh ]]; then
  echo "ERROR: submit from the repository root or set RLHF_PROJECT_ROOT: $PROJECT_ROOT" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not on PATH. Activate rlhf-cuq before sbatch." >&2
  exit 2
fi
# Conda activation hooks may read unset variables (for example,
# MKL_INTERFACE_LAYER), so temporarily disable Bash nounset.
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u

source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
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
echo "Smoke seed: $RM_SEED"
echo "Expected output: $OUTPUT_DIR"
echo "Plan: generic GPU, FP16, standard attention, 512 pairs, 16 optimizer steps"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))"

python scripts/download_assets.py \
  --asset-root assets \
  --asset proxy_rm_sft_base \
  --verify-only
python scripts/build_data_manifest.py \
  --config configs/data_split_prompt_disjoint_v1.yaml \
  --verify-only

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: FP16 smoke output is not empty: $OUTPUT_DIR" >&2
  echo "Move the old smoke output aside before rerunning; this job will not overwrite it." >&2
  exit 2
fi

srun accelerate launch --config_file configs/accelerate_config_fp16.yaml \
  src/reward_modeling/training/trainer_rm_manifest.py \
  --configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-smoke rm-pythia-44m-cluster-smoke-fp16 \
  --rng_seed "$RM_SEED"

python -c "import json, math, pathlib, sys; root = pathlib.Path(sys.argv[1]); config = json.loads((root / 'config.json').read_text()); mean, std = config.get('mean'), config.get('std'); assert isinstance(mean, (int, float)) and math.isfinite(mean), f'invalid reward mean: {mean}'; assert isinstance(std, (int, float)) and math.isfinite(std) and std > 0, f'invalid reward std: {std}'; assert (root / 'tokenizer_config.json').is_file(), 'tokenizer_config.json missing'; assert any(path.suffix in {'.bin', '.safetensors'} for path in root.iterdir()), 'model weights missing'; print('PASS finite reward normalization', 'mean=', mean, 'std=', std)" "$OUTPUT_DIR"

mkdir -p artifacts/checksums
find "$OUTPUT_DIR" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$CHECKSUM_FILE"

echo "PASS generic-GPU FP16 proxy-RM pipeline smoke"
echo "Smoke checkpoint (use only for baseline_smoke PPO integration): $OUTPUT_DIR"
echo "Checksums: $CHECKSUM_FILE"
