#!/usr/bin/env bash
# Non-scientific end-to-end check of the manifest-backed proxy-RM pipeline.
# Submit from the repository root with:
#   sbatch scripts/slurm/smoke_proxy_rm.sbatch

#SBATCH -J proxy_rm_44m_smoke
#SBATCH --account=gts-yxie77-paid
#SBATCH --mem=16G
#SBATCH -t20
#SBATCH --gres=gpu:1 -C A100
#SBATCH -qinferno
#SBATCH -o Report-smoke-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
conda activate mfg_flow

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this file with sbatch; do not run it on a login node." >&2
  exit 2
fi

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
RM_SEED="${RM_SEED:-1}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
OUTPUT_DIR="models/rm-pythia-44m-prompt-disjoint-smoke_seed${RM_SEED}"
CHECKSUM_FILE="artifacts/checksums/rm-pythia-44m-prompt-disjoint-smoke_seed${RM_SEED}.sha256"

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
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

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
echo "Smoke seed: $RM_SEED"
echo "Expected output: $OUTPUT_DIR"
echo "Plan: 512 training pairs, 1 epoch, effective batch 32, 16 optimizer steps"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported(), 'The checked RM configuration requires BF16 support'"

python scripts/download_assets.py \
  --asset-root assets \
  --asset proxy_rm_sft_base \
  --verify-only
python scripts/build_data_manifest.py \
  --config configs/data_split_prompt_disjoint_v1.yaml \
  --verify-only

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: smoke output is not empty: $OUTPUT_DIR" >&2
  echo "Move the old smoke output aside before rerunning; this job will not overwrite it." >&2
  exit 2
fi

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/reward_modeling/training/trainer_rm_manifest.py \
  --configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-smoke \
  --rng_seed "$RM_SEED"

python -c "import json, math, pathlib, sys; root = pathlib.Path(sys.argv[1]); config = json.loads((root / 'config.json').read_text()); mean, std = config.get('mean'), config.get('std'); assert isinstance(mean, (int, float)) and math.isfinite(mean), f'invalid reward mean: {mean}'; assert isinstance(std, (int, float)) and math.isfinite(std) and std > 0, f'invalid reward std: {std}'; assert (root / 'tokenizer_config.json').is_file(), 'tokenizer_config.json missing'; assert any(path.suffix in {'.bin', '.safetensors'} for path in root.iterdir()), 'model weights missing'; print('PASS finite reward normalization', 'mean=', mean, 'std=', std)" "$OUTPUT_DIR"

mkdir -p artifacts/checksums
find "$OUTPUT_DIR" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$CHECKSUM_FILE"

echo "PASS proxy-RM pipeline smoke"
echo "Smoke checkpoint (do not use for PPO): $OUTPUT_DIR"
echo "Checksums: $CHECKSUM_FILE"
