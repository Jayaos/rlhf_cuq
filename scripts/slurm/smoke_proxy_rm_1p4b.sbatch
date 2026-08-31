#!/usr/bin/env bash
# Exercise conversion, a few RM updates, final validation, normalization, and
# saving with the real 1.4B SFT asset. This output is smoke-only.

#SBATCH -J proxy_rm_1p4b_smoke
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-proxy-rm-1p4b-smoke-%j.out
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
OUTPUT_BASE="${RM_OUTPUT_BASE:-$JOB_STORAGE_ROOT/models/rm-pythia-1p4b-prompt-disjoint-smoke}"
OUTPUT_DIR="${OUTPUT_BASE}_seed${RM_SEED}"
CHECKSUM_DIR="${RM_CHECKSUM_DIR:-$JOB_STORAGE_ROOT/checksums}"
CHECKSUM_FILE="$CHECKSUM_DIR/rm-pythia-1p4b-prompt-disjoint-smoke_seed${RM_SEED}.sha256"

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DIR="$JOB_STORAGE_ROOT/wandb"
mkdir -p "$WANDB_DIR" "$(dirname -- "$OUTPUT_BASE")" "$CHECKSUM_DIR"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print('GPU', torch.cuda.get_device_name(0))"
python scripts/download_assets.py --asset-root assets --asset initial_sft_policy --verify-only
python scripts/build_data_manifest.py --config configs/data_split_prompt_disjoint_v1.yaml --verify-only

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: prior 1.4B RM smoke output is not empty: $OUTPUT_DIR" >&2
  exit 2
fi

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/reward_modeling/training/trainer_rm_manifest.py \
  --configs defaults_rm rm-pythia-44m rm-pythia-1p4b-cluster-smoke \
  --rng_seed "$RM_SEED" \
  --output_dir "$OUTPUT_BASE"

python - "$OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "final_eval_results.json").read_text(encoding="utf-8"))
mean, std = config.get("mean"), config.get("std")
assert isinstance(mean, (int, float)) and math.isfinite(mean)
assert isinstance(std, (int, float)) and math.isfinite(std) and std > 0
assert any(path.suffix in {".bin", ".safetensors"} for path in root.iterdir())
assert any(key.endswith("accuracy") for key in metrics), metrics
print("PASS 1.4B RM smoke validation", f"mean={mean}", f"std={std}")
PY

find "$OUTPUT_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$CHECKSUM_FILE"
echo "PASS matched-capacity 1.4B proxy-RM pipeline smoke"
echo "Smoke-only checkpoint: $OUTPUT_DIR"
