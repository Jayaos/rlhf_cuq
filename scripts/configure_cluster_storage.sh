#!/usr/bin/env bash
# Source before Conda/pip/Hugging Face operations, for example:
#   source scripts/configure_cluster_storage.sh /storage/scratch1/0/$USER/rlhf-cuq

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this script so its exports remain in your shell." >&2
  echo "Usage: source scripts/configure_cluster_storage.sh /absolute/scratch/path" >&2
  exit 2
fi

if [[ $# -ne 1 || -z "$1" ]]; then
  echo "ERROR: provide one absolute scratch/project cache root." >&2
  return 2
fi

case "$1" in
  /*) ;;
  *)
    echo "ERROR: storage root must be an absolute path: $1" >&2
    return 2
    ;;
esac

if [[ "$1" == "/" || "$1" == "$HOME" || "$1" == "$HOME/"* ]]; then
  echo "ERROR: storage root must not be the filesystem root or home directory." >&2
  return 2
fi

# A clean Conda install must not resolve imports or dependency satisfaction
# through packages inherited from ~/.local or a previously activated prefix.
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

export RLHF_STORAGE_ROOT="${1%/}"
export PIP_CACHE_DIR="$RLHF_STORAGE_ROOT/pip-cache"
export TMPDIR="$RLHF_STORAGE_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$RLHF_STORAGE_ROOT/xdg-cache"
export TORCH_EXTENSIONS_DIR="$RLHF_STORAGE_ROOT/torch-extensions"
export HF_HOME="$RLHF_STORAGE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export CONDA_PKGS_DIRS="$RLHF_STORAGE_ROOT/conda-pkgs"

mkdir -p -- \
  "$PIP_CACHE_DIR" \
  "$TMPDIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$HF_DATASETS_CACHE" \
  "$CONDA_PKGS_DIRS"

echo "Cluster cache/build storage configured under: $RLHF_STORAGE_ROOT"
echo "pip cache: $PIP_CACHE_DIR"
echo "temporary builds: $TMPDIR"
echo "Hugging Face cache: $HF_HOME"
echo "Python user-site packages disabled: PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
