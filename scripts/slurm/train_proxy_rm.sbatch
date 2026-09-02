#!/usr/bin/env bash
# Train one manifest-backed 44M proxy reward model on a Phoenix GPU node.
#
# Submit from the repository root with:
#   sbatch --account="$PACE_ACCOUNT" scripts/slurm/train_proxy_rm.sbatch
#
# Override RM_SEED for one seed, or submit an array for seeds 1--5:
#   sbatch --account="$PACE_ACCOUNT" --array=1-5 scripts/slurm/train_proxy_rm.sbatch
# Opt-in 30% training-label-noise stress run (the noiseless default is unchanged):
#   sbatch --export=ALL,RM_LABEL_NOISE_RATE=0.30,RM_LABEL_NOISE_SEED=1 \
#     scripts/slurm/train_proxy_rm.sbatch

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
RM_LABEL_NOISE_RATE="${RM_LABEL_NOISE_RATE:-0.0}"
RM_LABEL_NOISE_SEED="${RM_LABEL_NOISE_SEED:-1}"
if [[ ! "$RM_LABEL_NOISE_SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: RM_LABEL_NOISE_SEED must be a nonnegative integer" >&2
  exit 2
fi
NOISE_TAG="$(python -c 'from decimal import Decimal; import sys; value=Decimal(sys.argv[1]); assert value.is_finite() and 0 <= value <= 1; text=format(value.normalize(), "f"); print(text.replace(".", "p"))' "$RM_LABEL_NOISE_RATE")" || {
  echo "ERROR: RM_LABEL_NOISE_RATE must be a finite decimal in [0, 1]" >&2
  exit 2
}
if [[ "$NOISE_TAG" == "0" ]]; then
  DEFAULT_OUTPUT_BASE="models/rm-pythia-44m-prompt-disjoint"
  CHECKSUM_STEM="rm-pythia-44m-prompt-disjoint"
else
  DEFAULT_OUTPUT_BASE="models/rm-pythia-44m-prompt-disjoint-label-noise-${NOISE_TAG}"
  CHECKSUM_STEM="rm-pythia-44m-prompt-disjoint-label-noise-${NOISE_TAG}"
fi
OUTPUT_BASE="${RM_OUTPUT_BASE:-$DEFAULT_OUTPUT_BASE}"
OUTPUT_DIR="${OUTPUT_BASE}_seed${RM_SEED}"
RESUME_CHECKPOINT="${RM_RESUME_CHECKPOINT:-}"
CHECKSUM_DIR="${RM_CHECKSUM_DIR:-artifacts/checksums}"
CHECKSUM_FILE="$CHECKSUM_DIR/${CHECKSUM_STEM}_seed${RM_SEED}.sha256"

if [[ ! "$RM_SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: RM_SEED/SLURM_ARRAY_TASK_ID must be a positive integer: $RM_SEED" >&2
  exit 2
fi
if [[ -z "$OUTPUT_BASE" || "$OUTPUT_BASE" == "/" ]]; then
  echo "ERROR: RM_OUTPUT_BASE must name a model-output base path, not an empty path or /." >&2
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
export WANDB_DIR="$JOB_STORAGE_ROOT/wandb"
mkdir -p "$WANDB_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Repository: $PROJECT_ROOT"
echo "Git revision: $(git rev-parse HEAD)"
echo "Conda environment: $CONDA_PREFIX"
echo "Scratch/cache root: $JOB_STORAGE_ROOT"
echo "RM seed: $RM_SEED"
echo "RM training-label noise: rate=$RM_LABEL_NOISE_RATE seed=$RM_LABEL_NOISE_SEED"
echo "Output base: $OUTPUT_BASE"
echo "Expected output: $OUTPUT_DIR"
echo "Checksum output: $CHECKSUM_FILE"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported(), 'The checked RM configuration requires BF16 support'"

RESUME_ARGS=()
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: resume requested, but the expected output directory is missing: $OUTPUT_DIR" >&2
    exit 2
  fi
  if [[ ! -d "$RESUME_CHECKPOINT" ]]; then
    echo "ERROR: requested resume checkpoint is missing: $RESUME_CHECKPOINT" >&2
    exit 2
  fi

  OUTPUT_DIR_PATH="$(realpath -e -- "$OUTPUT_DIR")"
  RESUME_CHECKPOINT_PATH="$(realpath -e -- "$RESUME_CHECKPOINT")"
  RESUME_CHECKPOINT_NAME="$(basename -- "$RESUME_CHECKPOINT_PATH")"

  if [[ "$(dirname -- "$RESUME_CHECKPOINT_PATH")" != "$OUTPUT_DIR_PATH" ]]; then
    echo "ERROR: resume checkpoint must be an immediate child of $OUTPUT_DIR_PATH" >&2
    echo "Received: $RESUME_CHECKPOINT_PATH" >&2
    exit 2
  fi
  if [[ ! "$RESUME_CHECKPOINT_NAME" =~ ^checkpoint-([1-9][0-9]*)$ ]]; then
    echo "ERROR: invalid resume checkpoint name: $RESUME_CHECKPOINT_NAME" >&2
    exit 2
  fi
  RESUME_STEP="${BASH_REMATCH[1]}"

  LATEST_CHECKPOINT="$(
    find "$OUTPUT_DIR_PATH" -mindepth 1 -maxdepth 1 -type d \
      -name 'checkpoint-[0-9]*' -print \
      | sort -V \
      | tail -n 1
  )"
  if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "ERROR: no checkpoints were found under $OUTPUT_DIR_PATH" >&2
    exit 2
  fi
  LATEST_CHECKPOINT="$(realpath -e -- "$LATEST_CHECKPOINT")"
  if [[ "$RESUME_CHECKPOINT_PATH" != "$LATEST_CHECKPOINT" ]]; then
    echo "ERROR: the legacy trainer resumes only the latest checkpoint." >&2
    echo "Requested: $RESUME_CHECKPOINT_PATH" >&2
    echo "Latest:    $LATEST_CHECKPOINT" >&2
    exit 2
  fi

  python - "$RESUME_CHECKPOINT_PATH" "$RESUME_STEP" <<'PY'
import gc
import json
import sys
from pathlib import Path

import torch


root = Path(sys.argv[1])
expected_step = int(sys.argv[2])
required = (
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
)
for name in required:
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"resume checkpoint is incomplete: {path}")

state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
if int(state.get("global_step", -1)) != expected_step:
    raise RuntimeError(
        f"checkpoint name/state mismatch: name={expected_step}, "
        f"trainer_state={state.get('global_step')}"
    )

weight_files = sorted(root.glob("pytorch_model*.bin")) + sorted(
    root.glob("*.safetensors")
)
if not weight_files:
    raise RuntimeError(f"resume checkpoint has no model weights: {root}")
for path in weight_files:
    if path.stat().st_size == 0:
        raise RuntimeError(f"resume checkpoint has an empty weight file: {path}")

# Deserializing the mutable state catches truncated files before requesting a
# GPU training continuation. Model weights are loaded by Trainer immediately
# afterward; their presence and nonzero size are checked here.
for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
    value = torch.load(root / name, map_location="cpu")
    del value
    gc.collect()

print(
    "PASS resumable RM checkpoint",
    f"path={root}",
    f"global_step={expected_step}",
    f"epoch={state.get('epoch')}",
)
PY

  RESUME_ARGS+=(--resume_from_checkpoint)
  echo "Resume checkpoint: $RESUME_CHECKPOINT_PATH"
else
  if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: output directory is not empty: $OUTPUT_DIR" >&2
    echo "Move it aside, select a new seed, or set RM_RESUME_CHECKPOINT to the latest complete checkpoint." >&2
    exit 2
  fi
fi

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/reward_modeling/training/trainer_rm_manifest.py \
  --configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-split \
  --rng_seed "$RM_SEED" \
  --output_dir "$OUTPUT_BASE" \
  --rm_label_noise_rate "$RM_LABEL_NOISE_RATE" \
  --rm_label_noise_seed "$RM_LABEL_NOISE_SEED" \
  "${RESUME_ARGS[@]}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "ERROR: training returned success but the expected output is missing: $OUTPUT_DIR" >&2
  exit 1
fi
if [[ "$NOISE_TAG" != "0" ]]; then
  python scripts/validate_rm_label_noise.py \
    --model-dir "$OUTPUT_DIR" \
    --rate "$RM_LABEL_NOISE_RATE" \
    --seed "$RM_LABEL_NOISE_SEED"
fi

mkdir -p "$CHECKSUM_DIR"
find "$OUTPUT_DIR" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$CHECKSUM_FILE"

echo "PASS proxy RM seed $RM_SEED"
echo "Checkpoint: $OUTPUT_DIR"
echo "Checksums: $CHECKSUM_FILE"
