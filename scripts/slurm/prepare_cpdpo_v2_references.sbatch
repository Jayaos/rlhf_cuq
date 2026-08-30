#!/usr/bin/env bash
# Prepare one immutable SFT anchor per unique prompt in one full seed schedule.
# Example:
#   sbatch --export=ALL,CPDPO_V2_SEED=1 scripts/slurm/prepare_cpdpo_v2_references.sbatch

#SBATCH -J cpdpo_v2_refs
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-cpdpo-v2-refs-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3
set -euo pipefail

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
SEED="${CPDPO_V2_SEED:-1}"
POLICY_PATH="${CPDPO_POLICY_PATH:-assets/initial_sft_policy}"
PROXY_RM_PATH="${CPDPO_PROXY_RM_PATH:-models/rm-pythia-44m-prompt-disjoint_seed1}"
SCHEDULE="${CPDPO_V2_SCHEDULE:-artifacts/prompt_schedules/seed_${SEED}.jsonl}"
CPDPO_ALPHA="${CPDPO_ALPHA:-0.10}"

cd "$PROJECT_ROOT"
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u
source scripts/configure_cluster_storage.sh "$JOB_STORAGE_ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ALPHA_TAG="$(python -c 'import sys; from src.cpdpo.spec import alpha_tag; print(alpha_tag(float(sys.argv[1])))' "$CPDPO_ALPHA")"
if [[ -n "${CPDPO_ARTIFACT_DIR:-}" ]]; then
  ARTIFACT_DIR="$CPDPO_ARTIFACT_DIR"
elif [[ "$ALPHA_TAG" == "0p1" ]]; then
  ARTIFACT_DIR="artifacts/cpdpo/proxy_rm_seed1"
else
  ARTIFACT_DIR="artifacts/cpdpo/proxy_rm_seed1_alpha_${ALPHA_TAG}"
fi
if [[ -n "${CPDPO_V2_REFERENCE_DIR:-}" ]]; then
  REFERENCE_DIR="$CPDPO_V2_REFERENCE_DIR"
elif [[ "$ALPHA_TAG" == "0p1" ]]; then
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/cpdpo_v2/seed_${SEED}"
else
  REFERENCE_DIR="$JOB_STORAGE_ROOT/artifacts/cpdpo_v2/seed_${SEED}_alpha_${ALPHA_TAG}"
fi

python scripts/validate_legacy_sources.py
python -m pip check
test -f "$SCHEDULE"
test -d "$POLICY_PATH"
test -d "$PROXY_RM_PATH"
test -f "$ARTIFACT_DIR/pair_geometry.pt"
test -f "$ARTIFACT_DIR/conformal_calibration.json"

srun python scripts/prepare_cpdpo_v2_reference_cache.py \
  --prompt-schedule "$SCHEDULE" \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --policy-model "$POLICY_PATH" \
  --proxy-rm "$PROXY_RM_PATH" \
  --geometry "$ARTIFACT_DIR/pair_geometry.pt" \
  --calibration "$ARTIFACT_DIR/conformal_calibration.json" \
  --output-dir "$REFERENCE_DIR" \
  --base-seed "$SEED" \
  --alpha "$CPDPO_ALPHA" \
  --device cuda:0 \
  --dtype bf16

echo "PASS CPDPOv2 reference preparation seed=$SEED alpha=$CPDPO_ALPHA path=$REFERENCE_DIR"
