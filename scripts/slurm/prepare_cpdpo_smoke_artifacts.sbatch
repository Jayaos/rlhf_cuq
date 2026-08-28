#!/usr/bin/env bash
# Fast, non-scientific artifact preparation for exercising the CPDPO pipeline.
# It uses small deterministic prefixes of D_rm_train and D_cal. The resulting
# artifacts are tagged smoke-only and are rejected by the full training job.
# Submit from the repository root:
#   sbatch scripts/slurm/prepare_cpdpo_smoke_artifacts.sbatch

#SBATCH -J cpdpo_art_smoke
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-cpdpo-artifacts-smoke-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint-smoke_seed1}"
MANIFEST_PATH="${CPDPO_MANIFEST_PATH:-data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json}"
ARTIFACT_DIR="${CPDPO_ARTIFACT_DIR:-artifacts/cpdpo/proxy_rm_smoke_seed1_quick}"
RM_PAIRS="${CPDPO_SMOKE_RM_PAIRS:-64}"
CAL_PAIRS="${CPDPO_SMOKE_CAL_PAIRS:-64}"

if [[ ! "$RM_PAIRS" =~ ^[1-9][0-9]*$ ]] || [[ ! "$CAL_PAIRS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CPDPO_SMOKE_RM_PAIRS and CPDPO_SMOKE_CAL_PAIRS must be positive integers" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python scripts/validate_legacy_sources.py
python scripts/build_data_manifest.py --config configs/data_split_prompt_disjoint_v1.yaml --verify-only
test -d "$PROXY_RM_PATH"
if [[ -d "$ARTIFACT_DIR" ]] && [[ -n "$(find "$ARTIFACT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "ERROR: CPDPO smoke artifact directory is not empty: $ARTIFACT_DIR" >&2
  exit 2
fi

srun python scripts/prepare_cpdpo_artifacts.py \
  --manifest "$MANIFEST_PATH" \
  --proxy-rm "$PROXY_RM_PATH" \
  --output-dir "$ARTIFACT_DIR" \
  --device cuda:0 \
  --batch-size 32 \
  --max-rm-pairs "$RM_PAIRS" \
  --max-cal-pairs "$CAL_PAIRS"

python - "$ARTIFACT_DIR" "$RM_PAIRS" "$CAL_PAIRS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_rm = int(sys.argv[2])
expected_cal = int(sys.argv[3])
geometry = json.loads((root / "pair_geometry_metadata.json").read_text())
calibration = json.loads((root / "conformal_calibration.json").read_text())
assert geometry["artifact_scope"] == calibration["artifact_scope"] == "smoke"
assert geometry["n_rm"] == min(expected_rm, geometry["available_n_rm"])
assert calibration["n_cal"] == min(expected_cal, calibration["available_n_cal"])
print("PASS reduced CPDPO artifact provenance")
PY

echo "PASS CPDPO smoke-only offline artifacts: $ARTIFACT_DIR"
