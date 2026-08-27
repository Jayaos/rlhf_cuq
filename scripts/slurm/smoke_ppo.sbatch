#!/usr/bin/env bash
# Non-scientific, one-update PPO integration check using one local proxy RM.
# Submit from the repository root with:
#   sbatch scripts/slurm/smoke_ppo.sbatch
#
# Override the proxy RM when needed, for example:
#   sbatch --export=ALL,PPO_SMOKE_RM_PATH=models/rm-pythia-44m-prompt-disjoint-smoke-fp16_seed1 scripts/slurm/smoke_ppo.sbatch

#SBATCH -J ppo_1step_smoke
#SBATCH --account=gts-yxie77-paid
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100|H200"
#SBATCH -qinferno
#SBATCH -o Report-ppo-smoke-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlee3541@gatech.edu

module load anaconda3

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit this file with sbatch; do not run it on a login node." >&2
  exit 2
fi

PROJECT_ROOT="${RLHF_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
CONDA_ENV_NAME="${RLHF_CONDA_ENV:-rlhf-cuq}"
JOB_STORAGE_ROOT="${RLHF_JOB_STORAGE_ROOT:-/storage/scratch1/0/${USER}/rlhf-cuq}"
POLICY_PATH="${PPO_SMOKE_POLICY_PATH:-assets/initial_sft_policy}"
PROXY_RM_PATH="${PPO_SMOKE_RM_PATH:-models/rm-pythia-44m-prompt-disjoint-smoke_seed1}"
PPO_SEED="${PPO_SMOKE_SEED:-1}"
OUTPUT_ROOT="runs/ppo_smoke"
CHECKPOINT_ROOT="runs/ppo_smoke_checkpoints"
PROMPT_PREVIEW="output.txt"
CHECKSUM_FILE="artifacts/checksums/ppo-smoke-${SLURM_JOB_ID}.sha256"

if [[ ! "$PPO_SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: PPO_SMOKE_SEED must be a nonnegative integer: $PPO_SEED" >&2
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
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Repository: $PROJECT_ROOT"
echo "Git revision: $(git rev-parse HEAD)"
echo "Conda environment: $CONDA_PREFIX"
echo "Scratch/cache root: $JOB_STORAGE_ROOT"
echo "Policy: $POLICY_PATH"
echo "Proxy RM: $PROXY_RM_PATH"
echo "PPO seed: $PPO_SEED"
echo "Plan: 2 rollouts, 1 PPO epoch, 1 optimizer update, no gold model"

python scripts/validate_legacy_sources.py

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported()); assert torch.cuda.is_bf16_supported(), 'The checked PPO smoke configuration requires BF16 support'"

python scripts/download_assets.py \
  --asset-root assets \
  --asset initial_sft_policy \
  --verify-only
python scripts/build_data_manifest.py \
  --config configs/data_split_prompt_disjoint_v1.yaml \
  --verify-only

if [[ ! -d "$POLICY_PATH" ]]; then
  echo "ERROR: local policy directory does not exist: $POLICY_PATH" >&2
  exit 2
fi
if [[ ! -d "$PROXY_RM_PATH" ]]; then
  echo "ERROR: local proxy-RM directory does not exist: $PROXY_RM_PATH" >&2
  exit 2
fi
python -c "import json, math, pathlib, sys; root = pathlib.Path(sys.argv[1]); config = json.loads((root / 'config.json').read_text()); mean, std = config.get('mean'), config.get('std'); assert isinstance(mean, (int, float)) and math.isfinite(mean), f'invalid reward mean: {mean}'; assert isinstance(std, (int, float)) and math.isfinite(std) and std > 0, f'invalid reward std: {std}'; assert (root / 'tokenizer_config.json').is_file(), 'tokenizer_config.json missing'; assert any(path.suffix in {'.bin', '.safetensors'} for path in root.iterdir()), 'model weights missing'; print('PASS proxy-RM checkpoint validation', 'mean=', mean, 'std=', std)" "$PROXY_RM_PATH"

for path in "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"; do
  if [[ -d "$path" ]] && [[ -n "$(find "$path" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: prior PPO smoke artifact directory is not empty: $path" >&2
    echo "Move it aside before rerunning; this job will not overwrite it." >&2
    exit 2
  fi
done
if [[ -e "$PROMPT_PREVIEW" ]]; then
  echo "ERROR: trainer prompt preview already exists: $PROMPT_PREVIEW" >&2
  echo "Move it aside before running this smoke; the legacy trainer overwrites that path." >&2
  exit 2
fi

srun accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_rl.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual \
    prompt_disjoint_data_split_v1 baseline_smoke \
  --rng_seed "$PPO_SEED" \
  --run_gold_evaluation false \
  --policy_model_path_override "$POLICY_PATH" \
  --proxy_rm_path_override "$PROXY_RM_PATH"

if [[ ! -d "$OUTPUT_ROOT" ]]; then
  echo "ERROR: PPO returned success but output root is missing: $OUTPUT_ROOT" >&2
  exit 1
fi
RESULT_MARKER="$(find "$OUTPUT_ROOT" -type f -name rm_model_names.txt -print -quit)"
if [[ -z "$RESULT_MARKER" ]]; then
  echo "ERROR: PPO returned success but rm_model_names.txt is missing under $OUTPUT_ROOT" >&2
  exit 1
fi
RESULT_DIR="$(dirname "$RESULT_MARKER")"
FINAL_MODEL_DIR="$RESULT_DIR/model"
CHECKPOINT_MODEL_DIR="$CHECKPOINT_ROOT/checkpoint_1/hf_model"

if [[ ! -d "$FINAL_MODEL_DIR" ]] || ! find "$FINAL_MODEL_DIR" -type f \( -name '*.bin' -o -name '*.safetensors' \) -print -quit | grep -q .; then
  echo "ERROR: final PPO smoke model weights are missing: $FINAL_MODEL_DIR" >&2
  exit 1
fi
if [[ ! -d "$CHECKPOINT_MODEL_DIR" ]] || ! find "$CHECKPOINT_MODEL_DIR" -type f \( -name '*.bin' -o -name '*.safetensors' \) -print -quit | grep -q .; then
  echo "ERROR: step-1 PPO smoke checkpoint is missing: $CHECKPOINT_MODEL_DIR" >&2
  exit 1
fi

python -c "import json, math, pathlib, sys; root = pathlib.Path(sys.argv[1]); eval_dir = root / 'eval'; files = sorted(eval_dir.glob('eval-*.json')); assert files, f'no evaluation JSON under {eval_dir}'; payload = json.loads(files[-1].read_text()); assert payload, f'empty evaluation payload: {files[-1]}'; rewards = [score for row in payload for score in row.get('proxy_scores', [])]; assert rewards and all(isinstance(x, (int, float)) and math.isfinite(x) for x in rewards), f'invalid proxy scores: {rewards}'; kl = payload[0].get('kl_list', []); assert kl and all(isinstance(x, (int, float)) and math.isfinite(x) for x in kl), f'invalid KL list: {kl}'; print('PASS finite PPO smoke metrics', 'eval=', files[-1], 'proxy_scores=', rewards, 'kl=', kl)" "$RESULT_DIR"

if [[ -f "$PROMPT_PREVIEW" ]]; then
  mv "$PROMPT_PREVIEW" "$RESULT_DIR/prompt_preview.txt"
fi

mkdir -p artifacts/checksums
find "$RESULT_DIR" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$CHECKSUM_FILE"

echo "PASS one-update PPO pipeline smoke"
echo "Final model: $FINAL_MODEL_DIR"
echo "Step-1 checkpoint: $CHECKPOINT_MODEL_DIR"
echo "Results: $RESULT_DIR"
echo "Checksums: $CHECKSUM_FILE"
echo "This run is an integration test only; never report it as an experiment."
