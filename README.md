# LLM Overoptimization and Reward Model Ensembles (ICLR 2024)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


This repository contains the code related to **ICLR 2024** paper "[Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/abs/2310.02743)". 

In particular, it provides the following:
- An easy way to perform LLM instruction supervised fine-tuning (SFT) as laid out in the paper
- An easy way to create and train one or more reward models to be used in reward model ensembles (or on their own)
- A best-of-*n* inference pipeline, both with individual reward models and ensembles
- A PPO-based RLHF training pipeline, both for individual reward models and ensembles
- Some crucial models and datasets to facilitate future work and experiments akin to those in the paper

We hope you can find the code, models, and datasets provided helpful for your own research. 

Note: SFT, RM, and PPO training backends are based on the [Open-Assistant](https://github.com/LAION-AI/Open-Assistant) and [trlx](https://github.com/CarperAI/trlx) libraries. Base models are taken from the open-source [Pythia](https://github.com/EleutherAI/pythia) suite.


## Cluster setup and experiment runbook

Use this section for the audited cluster workflow. The historical upstream instructions below are useful as a code reference, but they use moving Hub identifiers, assume a proxy checkpoint that is not released, and enable a gold-evaluation path that is not currently validated.

### What can be run now

| Run | Purpose | Inputs | Status |
| --- | --- | --- | --- |
| Source audit and unit tests | Check source revisions, file hashes, configs, and local helpers | CPU; no model payloads | Runnable |
| Proxy RM, seed 1 | Train the Coste 44M proxy RM from the pinned 70M SFT base | RM base + preference dataset | Runnable on one GPU; required before PPO |
| Additional proxy-RM seeds | Optional Coste ensemble study | Same as above | Not required for CPDPO; the target experiment freezes one shared seed-1 proxy RM |
| PPO integration smoke | One optimizer update using the real 1.4B policy and one RM | Policy + prompts + structurally valid full or smoke RM | Runnable on one GPU; setup check only |
| Checked-code single-RM PPO | Run the vendored `configs/ppo_config.yaml` on the strict split for 3,000 steps | Same as smoke | Runnable only after the smoke gate; gold must stay off |
| Coste ensemble PPO | Reproduce Coste mean/WCO/UWO behavior | Policy + prompts + five trained RMs | Retained as an optional legacy path; out of scope for the primary CPDPO experiment |
| CPDPO offline artifacts | Build pair geometry on `D_rm_train` and fixed calibration on `D_cal` | Full seed-1 proxy RM | Implemented; GPU smoke still required |
| PPO / PairPPO / CPDPO | Controlled reward-overoptimization comparison | Shared policy, proxy RM, prompt schedule, and response budget | Implemented as additive trainers; cluster smoke still required |
| CPDPOv2 exploratory track | Fixed-SFT reference-anchored robust proxy PPO | Existing full CPDPO artifacts plus an immutable per-schedule SFT reference cache | Implemented additively; run the dedicated cluster smoke before a full launch |
| AdvPO additive comparison | Paper Eq. (4), (6), and (7) with a batch-shared adversarial RM head | Full proxy RM, AdvPO confidence matrix, and immutable SFT references | Implemented separately; run the dedicated cluster smoke before a full launch |
| Offline common evaluation | Score the same saved response with proxy RM, gold RM, and sampled KL | Reconstructed AlpacaFarm 7B RM | Code implemented; licensed gold reconstruction/checksum remains an external prerequisite |

The revised primary experiment uses one seed-1 proxy RM, frozen and shared by
PPO, PairPPO, and CPDPO. AdvPO remains outside that frozen three-method
specification, but is now implemented as a separately named additive
comparison at the user's 2026-08-30 request.
PPO tests the ordinary scalar objective, PairPPO isolates the effect of the
paired objective with `R=m`, and CPDPO adds the fixed conformal robust margin.
All three methods receive the same prompt schedule, two responses per prompt,
and the same total optimizer-update budget.
PairPPO and CPDPO use a pairwise PPO-style surrogate with fixed `+R/-R`
coefficients; they are not standard scalar-reward PPO with two independent
returns.

The old `configs/ppo_config.yaml` remains a regression baseline. The revised
experiment uses `configs/ppo_config_reward_overoptimization.yaml`, 256 prompts
and two responses per prompt, four PPO epochs, and a primary KL coefficient of
zero. See [the source audit](docs/SOURCE_AUDIT.md), [implementation
plan](docs/IMPLEMENTATION_PLAN.md), [decision status](docs/OPEN_METHOD_DECISIONS.md),
and [environment lock notes](docs/ENVIRONMENT_LOCK.md).

### Cluster prerequisites

The candidate compatibility target is Linux x86-64, Python 3.10, PyTorch 2.0.1, and CUDA 11.8. PyTorch documents this exact CUDA 11.8 Conda combination on its [previous versions page](https://docs.pytorch.org/get-started/previous-versions/). The environment also installs CUDA 11.8 `nvcc`, because the pinned `flash-attn==2.0.8` build imports PyTorch and compiles CUDA code.

Before setup, ensure the cluster has:

- an NVIDIA driver compatible with CUDA 11.8 and an Ampere-or-newer GPU for the checked BF16 configs;
- a compatible host C/C++ compiler (a site GCC 11 module is a reasonable choice);
- Conda or Mamba, Git, outbound access on a login/data-transfer node, and at least 10 GiB for the online baseline;
- substantially more space for generated checkpoints; reserve over 100 GiB if staging the 50.2-GiB pair of gold weight-difference snapshots, a licensed LLaMA-7B base, and reconstructed models.

If the site uses environment modules, load its compiler/driver modules first. Module names are cluster-specific; the Conda YAML supplies the CUDA toolkit rather than assuming commands such as `module load cuda/11.8` exist.

### Create the Conda environment

From the repository root on a login or build node:

```bash
export PROJECT_ROOT="$(pwd)"
export SCRATCH_ROOT="/absolute/path/on/shared/scratch/$USER/rlhf-cuq"
source scripts/configure_cluster_storage.sh "$SCRATCH_ROOT"

# These must show scratch-backed caches and disabled user-site packages.
python -m pip cache dir
echo "$TMPDIR"
echo "$PYTHONNOUSERSITE"

conda env create --file environment.cluster.yml
conda activate rlhf-cuq
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

python -m pip install --no-cache-dir \
  --constraint requirements/legacy-conda.constraints.txt \
  --requirement requirements/legacy-build.txt
python -m pip install --no-build-isolation \
  --constraint requirements/legacy-conda.constraints.txt \
  --requirement requirements/legacy-runtime.txt
python -m pip install --no-build-isolation --no-deps \
  --src "$CONDA_PREFIX/legacy-src" \
  --requirement requirements/legacy-sources.txt
python -m pip install --no-build-isolation --no-deps --editable .
python -m pip check
```

`environment.cluster.yml` installs the compiled foundation first. The build
requirements step then installs Python distribution metadata for CMake and lit,
which `triton==2.0.0` requires, and makes `pybind11` importable before fastText
runs under `--no-build-isolation`. The staged commands are intentional:
FlashAttention must see the installed Torch/CUDA toolchain, while pinned Open
Assistant must be installed editably with `--no-deps` because its wheel metadata
omits imported subpackages and contains moving VCS dependencies. Conda supports
creating an environment from a YAML file with `conda env create -f ...`; see the
[official command reference](https://docs.conda.io/projects/conda/en/latest/commands/env/create.html).

On PACE Phoenix, a concrete cache root is
`/storage/scratch1/<shard>/$USER/rlhf-cuq`; use the exact path printed by
`pace-quota`. Scratch is suitable for caches and temporary builds, not the only
copy of trained checkpoints. Source `configure_cluster_storage.sh` again in
every new login shell and Slurm job.

The helper also unsets inherited `PYTHONHOME`/`PYTHONPATH` and exports
`PYTHONNOUSERSITE=1`. This is required before the first pip transaction: a
Phoenix trial otherwise allowed packages under `~/.local` and a stale Conda
prefix to satisfy dependencies in the nominally clean environment.

Validate imports on the build node:

```bash
python --version
nvcc --version
python -c "import os, site, sys; actual=os.path.realpath(sys.prefix); expected=os.path.realpath(os.environ['CONDA_PREFIX']); assert actual == expected, (actual, expected); assert site.ENABLE_USER_SITE is False; print(sys.executable, actual, 'user_site=False')"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import pybind11; print(pybind11.__version__, pybind11.get_include())"
python -c "import pathtools, threadpoolctl; print(pathtools.__file__); print(threadpoolctl.__version__, threadpoolctl.__file__)"
python -m pip show cmake lit pybind11 pathtools pytest threadpoolctl
python -m pip check
python scripts/validate_legacy_sources.py
python -c "from model_training.custom_datasets.formatting import format_pairs; from model_training.models.reward_model import GPTNeoXRewardModel; from model_training.utils.utils import read_yamls; import alpaca_farm, oasst_data, trlx; assert callable(trlx.train); print('All imports and trlx.train passed')"
python -m pytest -q tests
python scripts/audit_assets.py --offline
```

Always pass `tests` to pytest. A bare `python -m pytest` also discovers test
suites inside the editable VCS checkouts that pip places below `src/`; those
upstream suites are not this repository's environment acceptance test.

Then, inside an allocated GPU job, require this check to print a visible GPU and `True`:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.is_available())"
```

This is a candidate legacy environment, not yet a fully resolved GPU lock. After the first successful smoke, save `python -m pip freeze --all`, `python -m torch.utils.collect_env`, the scheduler resource request, and all checkpoint checksums with the run artifacts.

If fastText fails with `ModuleNotFoundError: pybind11`, the build-requirements
step was skipped or ran in a different environment. If `pip check` says Triton
requires CMake/lit, rerun that same build-requirements step; a Conda CMake
executable alone does not provide the distribution metadata checked by pip. If
wheel output mentions `$HOME/.cache/pip`, source the storage helper before
retrying. A failed runtime install may have built wheels without installing the
transaction, so rerun the complete runtime requirements command after fixing
the prerequisite. If `pip check` reports missing `pathtools`, or an import
resolves through `~/.local`, source the storage helper and rerun the runtime
transaction; `pathtools==0.1.2` is now an explicit requirement for pinned
`wandb==0.15.8`.

If `build_data_manifest.py` reports `Invalid pattern: '**' can only be an
entire path component`, the legacy `datasets==2.14.4` install resolved an
incompatible newer `fsspec`. The runtime requirements now pin
`fsspec[http]==2023.9.2`; rerun the runtime transaction above rather than
upgrading `datasets` or the legacy trainer stack.

If PPO reports `module 'trlx' has no attribute 'train'`, Python is not loading
the pinned editable `CarperAI/trlx` source: that revision exports
`trlx.train`. A validator result such as `namespace without __file__` at
`$PROJECT_ROOT/src/trlx` means pip put the VCS checkout inside this project's
Python package tree, where the checkout root shadows its nested `trlx` package.
Uninstall the distribution, verify and move that exact checkout to recoverable
scratch quarantine, then reinstall outside the project tree:

```bash
source scripts/configure_cluster_storage.sh \
  "/storage/scratch1/0/$USER/rlhf-cuq"
conda activate rlhf-cuq

# These must identify CarperAI/trlx and the pinned 3340c2f... revision.
git -C "$PROJECT_ROOT/src/trlx" remote get-url origin
git -C "$PROJECT_ROOT/src/trlx" rev-parse HEAD
python -m pip uninstall -y trlx
mkdir -p "$RLHF_STORAGE_ROOT/quarantine"
mv "$PROJECT_ROOT/src/trlx" \
  "$RLHF_STORAGE_ROOT/quarantine/trlx-shadow-20260827"

python -m pip install --no-build-isolation --no-deps \
  --src "$CONDA_PREFIX/legacy-src" --editable \
  "git+https://github.com/CarperAI/trlx.git@3340c2f3a56d1d14fdd5f13ad575121fa26b6d92#egg=trlx"
python scripts/validate_legacy_sources.py
python -m pip check
```

This repair does not reinstall Torch, CUDA packages, Open Assistant, or the
trained models. Do not work around the failure by installing an arbitrary
PyPI `trlx` release; the PPO implementation is pinned by commit. If the
quarantine target already exists, choose a new explicit name instead of
overwriting it.

To repair an already-created `rlhf-cuq` environment after either observed
failure, do not recreate it. From the repository root run:

```bash
source scripts/configure_cluster_storage.sh \
  "/absolute/path/on/shared/scratch/$USER/rlhf-cuq"
conda activate rlhf-cuq
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"

python -m pip install --no-cache-dir \
  --constraint requirements/legacy-conda.constraints.txt \
  --requirement requirements/legacy-build.txt
python -m pip install --no-build-isolation \
  --constraint requirements/legacy-conda.constraints.txt \
  --requirement requirements/legacy-runtime.txt
python -m pip install --no-cache-dir \
  --constraint requirements/legacy-conda.constraints.txt \
  pytest==7.4.0
python -m pip check
```

On a login node without a visible GPU, the import check may make
`bitsandbytes==0.41.1` print a CPU-library warning. The checked RM configs use
`quantization: false` and PPO uses `adamw`, so bitsandbytes GPU functionality
is not an installation gate for these runs. Enabling 8-bit quantization or a
bitsandbytes optimizer would require a separate GPU-enabled validation.

### Download and verify data/checkpoints

Run downloads on the site-approved login or data-transfer node. The downloader reads [the source manifest](artifacts/source_manifest.json), passes each immutable revision to `huggingface_hub.snapshot_download`, and verifies every required local file by size and Git-blob SHA-1 or LFS SHA-256. The default download is about 2.93 GiB and includes:

- `assets/initial_sft_policy`: the 1.4B policy;
- `assets/proxy_rm_sft_base`: the 70M SFT base used to train proxy RMs;
- `assets/coste_preference_dataset`: RM training/validation data;
- `assets/alpaca_farm_prompt_dataset`: PPO training and validation prompts.

Review the license fields in the manifest before downloading: the AlpacaFarm prompt data is CC-BY-NC-4.0, while the Coste Hub model/dataset cards do not declare licenses in their metadata.

```bash
conda activate rlhf-cuq
cd "$PROJECT_ROOT"
python scripts/audit_assets.py
python scripts/download_assets.py --asset-root assets
python scripts/download_assets.py --asset-root assets --verify-only
```

The Hub supports revision-pinned full-repository downloads and local directories through `snapshot_download`; see the [official download guide](https://huggingface.co/docs/huggingface_hub/guides/download). If a partial/corrupt transfer is reported, rerun with `--force-download`. Do not hand-edit a verified asset directory.

### Build and verify the controlled data split

Materialize the logical roles once, before training any RM. The builder verifies
the pinned source payloads, loads only the exact JSON files declared in the
split config, validates their raw row counts, creates content-derived
record/prompt IDs, preserves exact duplicate source rows with deterministic
occurrence ordinals, keeps
duplicate prompts in one role, uses deterministic hash assignment with exact
largest-remainder quotas, audits overlap, and writes SHA-256-protected JSONL and
ID files. It never overwrites an existing split bundle.

```bash
python scripts/build_data_manifest.py --config configs/data_split_prompt_disjoint_v1.yaml
python scripts/build_data_manifest.py --config configs/data_split_prompt_disjoint_v1.yaml --verify-only
sha256sum data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json
```

The primary split reserves AlpacaFarm `unlabeled` exclusively for PPO. The RM
pool uses Coste's generated preference pairs from `human_pref`, `sft`,
`synth_pref`, and `val`; it deliberately excludes the preference
`unlabelled.json` file derived from PPO prompts. A pinned audit found zero
prompt-ID overlap between these source pools.

For the audited source sizes, the primary controlled roles are:

| Source pool | Logical role | Fraction | Rows |
| --- | --- | ---: | ---: |
| Non-`unlabelled` Coste pairs (31,382) | `D_rm_train` | 90% | 28,244 |
| Non-`unlabelled` Coste pairs | `D_rm_val` | 5% | 1,569 |
| Non-`unlabelled` Coste pairs | `D_cal` | 5% | 1,569 |
| Accepted AlpacaFarm `unlabeled` prompts (19,993) | `D_rl_train_prompts` | 80% | 15,995 |
| Accepted AlpacaFarm `unlabeled` prompts | `D_rl_val_prompts` | 10% | 1,999 |
| Accepted AlpacaFarm `unlabeled` prompts | `D_rl_test_prompts` | 10% | 1,999 |

The pinned raw AlpacaFarm file has 20,001 rows; the legacy Coste/Open-Assistant
content filter rejects eight before assignment. The two RM 5% roles together
form the requested 10% RM validation/calibration reservation. The manifest
enforces zero RM/PPO prompt intersection, and duplicate prompt groups cannot be
split across roles. Archive its hash with every run.

For a Coste-native overlap replication, separately build
`configs/data_split_coste_v1.yaml`. That manifest intentionally includes the
Coste `train/unlabelled.json` preference pairs and permits/reports their prompt
overlap with PPO. Do not pool its results with the strict primary protocol.

Do not replace the explicit `data_files` entries with a snapshot-directory
load. A Hub snapshot contains several JSON datasets/configurations. Recursive
discovery can duplicate the Coste preference rows and can combine AlpacaFarm
instruction data with evaluation JSON that has extra metadata columns.
Exact duplicates already present inside a verified source file are different:
they are retained to preserve the original sample multiplicity, assigned
distinct occurrence-qualified record IDs, and counted in manifest provenance.

The RM adapter exposes only `D_rm_train` and `D_rm_val`; the PPO adapter exposes
only `D_rl_train_prompts` and `D_rl_val_prompts`. `D_cal`, test, and external
validation roles cannot enter either online trainer through this adapter. The
offline CPDPO artifact builder reads `D_cal` explicitly to compute the frozen
finite-sample conformal threshold; `D_cal` still cannot enter proxy-RM fitting
or online PPO rollouts. The threshold is fixed for the complete run—there is no
adaptive update.

Use the prompt-disjoint manifest route for current and future controlled
experiments. For an exact legacy-data regression only, run the original
`trainer_rm.py` with `rm-pythia-44m-cluster`, or omit the split overlay from PPO
and supply `--rl_dataset_path_override assets/alpaca_farm_prompt_dataset`.
Label those runs `legacy_split`; they use the original implicit
first-N/source-validation selection and are not directly interchangeable with
either manifest protocol.

After verification, compute jobs can be network-independent:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

There is no released Coste 44M proxy-RM checkpoint. `assets/proxy_rm_sft_base` is not a reward model; train the scalar reward head as the next step.

Gold assets are intentionally excluded. Only after accepting AlpacaFarm's CC-BY-NC-4.0 terms and obtaining the original LLaMA-7B weights under their license, stage the two pinned weight differences with:

```bash
python scripts/download_assets.py --asset-root assets --include-gold
python scripts/download_assets.py --asset-root assets --include-gold --verify-only
```

This does not download LLaMA-7B or reconstruct the gold RM. After obtaining
authorized access to the original LLaMA-v1 7B base, place its Hugging Face
checkpoint on scratch and verify that `config.json`, `tokenizer.model`, the
weight index, and every indexed weight shard are present.  Do not pass the
license acknowledgement below merely because a mirror is technically
downloadable; retain the authorization record with the experiment audit.

The upstream AlpacaFarm recovery command resolves moving Hub repositories and
cannot consume the pinned local differences.  Reconstruct both required
stages offline with the checked wrapper instead:

```bash
export JOB_ROOT=/storage/scratch1/0/$USER/rlhf-cuq
export GOLD_LLAMA7B_PATH="$JOB_ROOT/gold-assets/llama-7b-hf"
mkdir -p "$JOB_ROOT/logs"

sbatch \
  --export=ALL,GOLD_LLAMA_LICENSE_ACKNOWLEDGED=yes,GOLD_LLAMA7B_PATH="$GOLD_LLAMA7B_PATH" \
  --output="$JOB_ROOT/logs/gold-rm-recover-%j.out" \
  scripts/slurm/reconstruct_alpaca_farm_gold_rm.sbatch
```

The job consumes the two manifest-pinned difference directories, reconstructs
`$JOB_ROOT/alpaca_farm_models/sft10k` and then
`$JOB_ROOT/alpaca_farm_models/reward-model-human`, requires both upstream
`model_sum.txt` checks to pass, fingerprints all inputs/outputs, and writes
`reconstruction_metadata.json`.  It is fully offline and refuses to overwrite
a nonempty model directory.  Use
`$JOB_ROOT/alpaca_farm_models/reward-model-human` as `GOLD_RM_PATH` only after
the job ends with all three `PASS reconstructed`/metadata messages.

The revised evaluator bypasses the broken legacy Alpaca formatting branch and
uses the pinned AlpacaFarm `v0_inputs_noinputs.json` templates, with a
regression test that rejects literal placeholders. Never commit or
redistribute licensed/reconstructed weights.

### Experiment 1: train proxy reward model(s)

#### Experiment 1a: short proxy-RM pipeline smoke

When the A100 queue is long, use the generic-GPU FP16 smoke job. It intentionally
removes the A100 constraint, disables FlashAttention, and can run on a V100,
RTX6000, A100, or another CUDA GPU with FP16 support:

```bash
conda activate rlhf-cuq
cd "$PROJECT_ROOT"
bash -n scripts/slurm/smoke_proxy_rm_any_gpu.sbatch
JOB_ID=$(sbatch --parsable scripts/slurm/smoke_proxy_rm_any_gpu.sbatch | cut -d';' -f1)
echo "Submitted generic-GPU smoke job: $JOB_ID"
```

The smoke overlay deterministically samples 512 pairs from `D_rm_train`, uses
128 `D_rm_val` pairs, keeps the production batch size 8 and gradient
accumulation 4, and runs one epoch (16 optimizer steps). It evaluates at steps
8 and 16, performs final reward mean/std normalization, and saves once. The
generic job requests one unconstrained GPU, four CPUs, 16 GiB CPU RAM, and 30
minutes. Expect roughly 10--25 minutes on a V100 or 5--15 minutes on an A100
after the job starts.

Monitor it with:

```bash
squeue -j "$JOB_ID"
tail -f "Report-smoke-fp16-${JOB_ID}.out"
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,ExitCode,AllocTRES
```

Success requires `State=COMPLETED`, `ExitCode=0:0`, and the final log line
`PASS generic-GPU FP16 proxy-RM pipeline smoke`. The isolated outputs are:

```text
models/rm-pythia-44m-prompt-disjoint-smoke-fp16_seed1/
artifacts/checksums/rm-pythia-44m-prompt-disjoint-smoke-fp16_seed1.sha256
```

To test the exact A100/BF16/FlashAttention path instead, submit
`scripts/slurm/smoke_proxy_rm.sbatch`; that job retains the A100 constraint and
currently requests 16 GiB CPU RAM for 20 minutes.

This checkpoint is deliberately trained on too little data. It may be used only
for the one-update PPO integration smoke selected by `baseline_smoke`; it must
not be used for checked-code/full PPO, CPDPO method comparisons, or reported
results. The job refuses to overwrite a nonempty prior smoke directory; move
that directory aside if a failed run must be repeated.

#### Experiment 1b: full proxy-RM seed

After the smoke passes, first train seed 1. The final overlay selects the local base model and the
manifest-backed `D_rm_train`/`D_rm_val`; all Coste RM hyperparameters still come
from the vendored `rm-pythia-44m` entry. Run the training command inside a
scheduled GPU allocation or batch job, not directly on a Phoenix login node.

```bash
conda activate rlhf-cuq
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/reward_modeling/training/trainer_rm_manifest.py \
  --configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-split \
  --rng_seed 1
```

The manifest wrapper adds the repository root to Python's import path before
loading the local `src` namespace; no manual `PYTHONPATH` setting or pip
package named `src` is required. If this command reports
`ModuleNotFoundError: No module named 'src'`, the cluster checkout predates
this launcher fix and must be updated before retrying.

The neutral local directory name `assets/proxy_rm_sft_base` also hides the
architecture from the pinned Open-Assistant code, which historically infers
Pythia handling from the model-name string. The cluster overlay therefore sets
`model_family: pythia`; the wrapper validates that the pinned local
`config.json` declares `model_type: gpt_neox`, then delegates tokenizer and
reward-model construction to the unchanged legacy functions. If the old
`Cannot find any tokeniser configuration` error appears, update the checkout
and confirm `grep -n model_family configs/config_rm_cluster.yaml` prints the
three cluster overlays.

Expected output: `models/rm-pythia-44m-prompt-disjoint_seed1`. Confirm that it contains model/tokenizer files and a finite saved reward normalization mean/std, then hash it:

The manifest wrapper performs one explicit validation pass immediately after
the final optimizer update. The log must contain
`PASS final RM validation` with `final_eval_D_rm_val_accuracy`; the complete
metric mapping is also saved as `final_eval_results.json` in the output
directory. This final pass is separate from the inherited evaluations every
300 steps and does not update the model or optimizer.

```bash
mkdir -p artifacts/checksums
find models/rm-pythia-44m-prompt-disjoint_seed1 -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > artifacts/checksums/rm-pythia-44m-prompt-disjoint_seed1.sha256
```

The repository includes a Phoenix Slurm job for this full GPU step. From the
repository root on a login node, activate the environment, use `pace-quota` to
confirm that the checked-in `gts-yxie77-paid` account is available, and submit
seed 1:

```bash
conda activate rlhf-cuq
cd "$PROJECT_ROOT"
pace-quota
sbatch scripts/slurm/train_proxy_rm.sbatch
```

The checked-in job requests one A100, H100, or H200 GPU, 32 GiB RAM, four
hours, and Phoenix `inferno` QOS. Slurm selects any eligible node carrying one
of those GPU feature labels. The job derives the repository from the directory
where `sbatch` is invoked and places caches under
`/storage/scratch1/0/$USER/rlhf-cuq`. Override those defaults at submission
only when needed:

```bash
sbatch --account=your_alternative_charge_account \
  --export=ALL,RLHF_PROJECT_ROOT=/absolute/repository/path,RLHF_JOB_STORAGE_ROOT=/absolute/scratch/path \
  scripts/slurm/train_proxy_rm.sbatch
```

`sbatch` prints the job ID. Monitor the job and its output/error log:

```bash
squeue -u "$USER"
tail -f Report-JOB_ID.out
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,AllocTRES
```

The job refuses to run outside Slurm, confirms CUDA and BF16 support, refuses
to overwrite a nonempty seed output during a fresh run, trains the RM, and
writes checkpoint hashes under `artifacts/checksums/`.

To resume an interrupted seed-1 run from its latest complete checkpoint, keep
the existing output directory and submit the checkpoint explicitly. For
example, to continue from step 2000:

```bash
sbatch \
  --export=ALL,RM_RESUME_CHECKPOINT=models/rm-pythia-44m-prompt-disjoint_seed1/checkpoint-2000 \
  scripts/slurm/train_proxy_rm.sbatch
```

Resume mode resolves paths before launching training and requires the requested
checkpoint to be the latest immediate `checkpoint-N` child of the expected
seed output. It checks that `trainer_state.json` reports the same step, that
model weights exist, and that the optimizer, scheduler, and RNG files can be
deserialized. The legacy Coste trainer then restores that checkpoint and skips
the already-completed training steps. If `RM_RESUME_CHECKPOINT` is omitted, the
original non-overwrite protection remains active.

If project storage is near its research-group quota, keep the repository and
manifest in project storage but place mutable RM checkpoints on scratch. The
`RM_OUTPUT_BASE` value is the base before the trainer appends `_seed1`; it must
therefore not include that suffix. For an interrupted seed-1 run, first move
the complete partial output directory without deleting it:

```bash
mkdir -p \
  /storage/scratch1/0/$USER/rlhf-cuq/models \
  /storage/scratch1/0/$USER/rlhf-cuq/checksums \
  /storage/scratch1/0/$USER/rlhf-cuq/logs
mv -- \
  models/rm-pythia-44m-prompt-disjoint_seed1 \
  /storage/scratch1/0/$USER/rlhf-cuq/models/
```

Then resume with both paths on scratch:

```bash
sbatch \
  --export=ALL,RM_SEED=1,RM_OUTPUT_BASE=/storage/scratch1/0/$USER/rlhf-cuq/models/rm-pythia-44m-prompt-disjoint,RM_RESUME_CHECKPOINT=/storage/scratch1/0/$USER/rlhf-cuq/models/rm-pythia-44m-prompt-disjoint_seed1/checkpoint-2000,RM_CHECKSUM_DIR=/storage/scratch1/0/$USER/rlhf-cuq/checksums \
  --output=/storage/scratch1/0/$USER/rlhf-cuq/logs/Report-%j.out \
  scripts/slurm/train_proxy_rm.sbatch
```

The job also directs W&B's offline run files to the scratch job-storage root.
Scratch is not backed up: after a successful run, retain the checksum, use the
absolute scratch checkpoint path for downstream jobs, and copy the frozen final
model to backed-up project storage once the research group has restored enough
quota headroom.

Seed 1 is the only proxy RM required for the primary PPO/PairPPO/CPDPO
comparison. Do not submit an RM job array for the primary track. Keep that exact
checkpoint frozen and use it for every optimization method and policy seed.

If a Coste ensemble study is added later, train additional RMs as a separately
declared optional comparison. It is not part of the CPDPO experiment.
The Slurm script supports independent array tasks, for example:

```bash
sbatch --array=2-3 \
  scripts/slurm/train_proxy_rm.sbatch
```

This optional command is not part of the primary runbook. Do not launch one
multi-GPU RM job: each array element is an independent one-GPU seed. Phoenix
uses QOS and assigns the resource pool from the request; do not add an arbitrary
partition copied from another cluster.

#### Additive matched-capacity ablation: 1.4B proxy RM

This ablation branches the already downloaded `assets/initial_sft_policy`
checkpoint into a **separate** `GPTNeoXRewardModel`, adds/trains its scalar
reward head, and preference-trains the full 1.4B branch on the same
`D_rm_train`. It validates on the same `D_rm_val`. The untouched causal LM is
not a usable reward model, and the policy and RM branches do not share live
weights after initialization.

The configuration retains the 44M track's Coste RM loss, learning rate, five
epochs, and effective batch size 32. For memory, it uses microbatch 1,
accumulation 32, gradient checkpointing, and scoring batch 1. It is a named RM
capacity ablation; it does not replace the checked 44M experiment.

First run the real-model smoke. Both smoke and full outputs default to scratch
because a 1.4B optimizer checkpoint is much larger than the 44M checkpoint:

```bash
cd /storage/project/r-yxie77-0/$USER/projects/rlhf_cuq
export JOB_ROOT=/storage/scratch1/0/$USER/rlhf-cuq

SMOKE_JOB=$(sbatch --parsable \
  --export=ALL,RLHF_JOB_STORAGE_ROOT="$JOB_ROOT" \
  scripts/slurm/smoke_proxy_rm_1p4b.sbatch | cut -d';' -f1)
echo "$SMOKE_JOB"
```

Success ends with `PASS matched-capacity 1.4B proxy-RM pipeline smoke`. The
smoke checkpoint is reduced-data integration evidence only and must not be
used for a scientific policy run. Then submit the full seed-1 RM:

```bash
RM_JOB=$(sbatch --parsable \
  --export=ALL,RLHF_JOB_STORAGE_ROOT="$JOB_ROOT",RM_SEED=1 \
  scripts/slurm/train_proxy_rm_1p4b.sbatch | cut -d';' -f1)
echo "$RM_JOB"
```

The full job requests one A100/H100/H200, 64 GiB host RAM, and 48 hours. Its
default frozen output and checksum are:

```text
$JOB_ROOT/models/rm-pythia-1p4b-prompt-disjoint_seed1
$JOB_ROOT/checksums/rm-pythia-1p4b-prompt-disjoint_seed1.sha256
```

Require a completed job, a final validation record, finite normalization, and
model weights before using it:

```bash
export PROXY_RM_1P4B=$JOB_ROOT/models/rm-pythia-1p4b-prompt-disjoint_seed1

sacct -j "$RM_JOB" --format=JobID,State,ExitCode,Elapsed,MaxRSS
test -f "$PROXY_RM_1P4B/config.json"
test -f "$PROXY_RM_1P4B/final_eval_results.json"
test -f "$JOB_ROOT/checksums/rm-pythia-1p4b-prompt-disjoint_seed1.sha256"
```

Resume works like the 44M job, but the checkpoint and output both remain under
the 1.4B scratch directory:

```bash
sbatch \
  --export=ALL,RLHF_JOB_STORAGE_ROOT="$JOB_ROOT",RM_RESUME_CHECKPOINT="$PROXY_RM_1P4B/checkpoint-2000" \
  scripts/slurm/train_proxy_rm_1p4b.sbatch
```

Do not reuse the 44M CPDPO or AdvPO artifacts. Build a newly fingerprinted
1.4B geometry/calibration set with a smaller scoring batch:

```bash
export CPDPO_1P4B_ARTIFACTS=$JOB_ROOT/artifacts/cpdpo/proxy_rm_1p4b_seed1

sbatch \
  --export=ALL,CPDPO_PROXY_RM_PATH="$PROXY_RM_1P4B",CPDPO_ARTIFACT_DIR="$CPDPO_1P4B_ARTIFACTS",CPDPO_ARTIFACT_BATCH_SIZE=2 \
  scripts/slurm/prepare_cpdpo_artifacts.sbatch
```

The larger feature dimension makes this artifact distinct even though the
equations and RM data are unchanged. For a one-rollout PPO/PairPPO/CPDPO
integration check, use a new output root and proxy scoring batch 2:

```bash
export OUTPUT_1P4B=$JOB_ROOT/outputs/reward_overoptimization_proxy_rm_1p4b_smoke

sbatch --array=0-2 \
  --export=ALL,CPDPO_PROXY_RM_PATH="$PROXY_RM_1P4B",CPDPO_ARTIFACT_DIR="$CPDPO_1P4B_ARTIFACTS",CPDPO_PROXY_BATCH_SIZE=2,CPDPO_SMOKE_OUTPUT_ROOT="$OUTPUT_1P4B" \
  scripts/slurm/smoke_reward_overoptimization.sbatch
```

After that smoke passes, a seed-1 full comparison uses the same proxy and
capacity-specific artifact paths for every method:

```bash
export OUTPUT_1P4B=$JOB_ROOT/outputs/reward_overoptimization_proxy_rm_1p4b

sbatch --array=0-2 \
  --export=ALL,ROLLOUT_STEPS=100,CPDPO_PROXY_RM_PATH="$PROXY_RM_1P4B",CPDPO_ARTIFACT_DIR="$CPDPO_1P4B_ARTIFACTS",CPDPO_PROXY_BATCH_SIZE=2,CPDPO_OUTPUT_ROOT="$OUTPUT_1P4B" \
  scripts/slurm/train_reward_overoptimization.sbatch
```

The same rule applies to AdvPO and CPDPOv2: pass the 1.4B proxy path, use
capacity-specific confidence/reference directories, and set
`CPDPO_PROXY_BATCH_SIZE=2`. Never compare a 1.4B-proxy treatment against a
44M-proxy control as though proxy capacity were held fixed. Offline evaluation
must also receive the same `CPDPO_PROXY_RM_PATH`, output root, and reduced
proxy batch.

For the existing seed-1 AdvPO comparison, declare the ridge and use new
confidence/reference locations. Reusing the same declared ridge as the 44M run
isolates one configuration choice, but the paper does not publish a canonical
ridge and representation scale can change with RM capacity:

```bash
export ADVPO_RIDGE_LAMBDA=1.0
export ADVPO_1P4B_CONF=$JOB_ROOT/artifacts/advpo/proxy_rm_1p4b_seed1_ridge_1
export ADVPO_1P4B_REFS=$JOB_ROOT/artifacts/advpo/proxy_rm_1p4b_references/seed_1

sbatch \
  --export=ALL,ADVPO_RIDGE_LAMBDA="$ADVPO_RIDGE_LAMBDA",CPDPO_PROXY_RM_PATH="$PROXY_RM_1P4B",ADVPO_CONFIDENCE_DIR="$ADVPO_1P4B_CONF",ADVPO_ARTIFACT_BATCH_SIZE=2 \
  scripts/slurm/prepare_advpo_confidence.sbatch

sbatch \
  --export=ALL,ADVPO_SEED=1,CPDPO_PROXY_RM_PATH="$PROXY_RM_1P4B",ADVPO_REFERENCE_DIR="$ADVPO_1P4B_REFS",CPDPO_PROXY_BATCH_SIZE=2 \
  scripts/slurm/prepare_advpo_references.sbatch

sbatch --array=1 \
  --export=ALL,ROLLOUT_STEPS=100,ADVPO_B=1,ADVPO_RIDGE_LAMBDA="$ADVPO_RIDGE_LAMBDA",CPDPO_PROXY_RM_PATH="$PROXY_RM_1P4B",ADVPO_CONFIDENCE_DIR="$ADVPO_1P4B_CONF",ADVPO_REFERENCE_DIR="$ADVPO_1P4B_REFS",CPDPO_PROXY_BATCH_SIZE=2,CPDPO_OUTPUT_ROOT="$OUTPUT_1P4B" \
  scripts/slurm/train_advpo.sbatch
```

Evaluate only after the relevant policy jobs complete. The evaluator rejects a
proxy fingerprint that does not match the training metadata:

```bash
sbatch --array=0-2 \
  --export=ALL,GOLD_RM_PATH="$GOLD_RM_PATH",CPDPO_PROXY_RM_PATH="$PROXY_RM_1P4B",CPDPO_PROXY_BATCH_SIZE=2,CPDPO_OUTPUT_ROOT="$OUTPUT_1P4B" \
  scripts/slurm/evaluate_reward_overoptimization.sbatch

sbatch --array=1 \
  --export=ALL,GOLD_RM_PATH="$GOLD_RM_PATH",ADVPO_B=1,CPDPO_PROXY_RM_PATH="$PROXY_RM_1P4B",CPDPO_PROXY_BATCH_SIZE=2,CPDPO_OUTPUT_ROOT="$OUTPUT_1P4B" \
  scripts/slurm/evaluate_advpo.sbatch
```

### Experiment 2: one-update PPO smoke

The checked-in batch job requests one A100/H100/H200-class GPU, four CPUs,
32 GiB host RAM, and 45 minutes. It defaults to the A100/BF16 proxy-RM smoke
checkpoint. Submit it from the repository root:

```bash
conda activate rlhf-cuq
cd "$PROJECT_ROOT"
bash -n scripts/slurm/smoke_ppo.sbatch
JOB_ID=$(sbatch --parsable scripts/slurm/smoke_ppo.sbatch | cut -d';' -f1)
echo "Submitted PPO smoke job: $JOB_ID"
```

To use the generic-GPU FP16 proxy-RM smoke checkpoint instead:

```bash
sbatch --export=ALL,PPO_SMOKE_RM_PATH=models/rm-pythia-44m-prompt-disjoint-smoke-fp16_seed1 \
  scripts/slurm/smoke_ppo.sbatch
```

To use the full seed-1 proxy RM, set
`PPO_SMOKE_RM_PATH=models/rm-pythia-44m-prompt-disjoint_seed1` in the same way.
The PPO job itself remains BF16 and therefore excludes V100. Command-line
`sbatch` options can override the checked-in account, QOS, time, or constraint
when the Phoenix allocation requires it.

The batch job runs the following command. Use it directly only inside an
existing Ampere-or-newer GPU allocation:

```bash
accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_rl.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual prompt_disjoint_data_split_v1 baseline_smoke \
  --policy_model_path_override assets/initial_sft_policy \
  --proxy_rm_path_override models/rm-pythia-44m-prompt-disjoint_seed1
```

For the A100/BF16 proxy-RM smoke output, replace the last argument with:

```text
models/rm-pythia-44m-prompt-disjoint-smoke_seed1
```

For the generic-GPU FP16 proxy-RM smoke output, use:

```text
models/rm-pythia-44m-prompt-disjoint-smoke-fp16_seed1
```

This uses two rollouts, one PPO epoch, one optimizer update, two evaluation prompts, and no gold load. It is an integration test, not a scientific experiment. The smoke YAML explicitly sets `tracker: null`: pinned trlx otherwise defaults to online W&B and fails in a noninteractive batch job without an API key. A proxy-RM smoke checkpoint validates loading, reward callback execution, PPO update, and artifact writing, but it says nothing about reward quality or overoptimization. Success ends with `PASS one-update PPO pipeline smoke`; the job checks finite proxy/KL values, the final policy, and `runs/ppo_smoke_checkpoints/checkpoint_1/hf_model`, then writes checksums under `artifacts/checksums/`. It refuses to overwrite nonempty `runs/ppo_smoke`, `runs/ppo_smoke_checkpoints`, or the legacy `output.txt` prompt preview. Move prior artifacts aside before retrying. Repeat the PPO smoke with the full seed-1 RM before any full run. The legacy evaluator can still generate up to 256 tokens even though smoke rollouts are capped at 16.

### Revised target experiment: PPO versus PairPPO versus CPDPO

The new pipeline deliberately reuses the full seed-1 proxy RM and the policy,
manifest, Accelerate launcher, optimizer, reference-policy, and checkpoint
machinery exercised by the earlier smoke tests. It adds a paired rollout store
and loss instead of modifying the baseline trainer.

#### 1. Build the fixed CPDPO artifacts

This is a one-time GPU job for a particular proxy-RM checkpoint and split
manifest:

```bash
sbatch scripts/slurm/prepare_cpdpo_artifacts.sbatch
```

It reads only `D_rm_train` and `D_cal` and writes:

```text
artifacts/cpdpo/proxy_rm_seed1/
  pair_geometry.pt
  pair_geometry_metadata.json
  calibration_scores.pt
  conformal_calibration.json
```

The geometry is the full normalized pair-difference Gram matrix with the
specified trace-scaled ridge and a float64 Cholesky factor. The calibration
uses `alpha=0.10`, the non-interpolated finite-sample order statistic, and one
fixed `q_alpha` for every CPDPO policy update. The job refuses to overwrite an
existing artifact directory.

The main run uses `--geometry-mode full --reward-variant robust_margin`.
Required ablations are explicit rather than alternate defaults: build a
separate artifact directory with `prepare_cpdpo_artifacts.py --geometry-mode
unit` for the `u=1` ablation, and launch CPDPO with `--reward-variant
sign_only` for the sign-only certified update. Never mix a unit-geometry
calibration artifact with a full-geometry run; fingerprint validation rejects
that mismatch.

`alpha=0.10` remains the specification-defined main result. To run a
predeclared alpha sensitivity ablation, export the same value while preparing
artifacts and training. For example:

```bash
ALPHA=0.20
sbatch --export=ALL,CPDPO_ALPHA="$ALPHA" \
  scripts/slurm/prepare_cpdpo_artifacts.sbatch

sbatch --array=2 \
  --export=ALL,ROLLOUT_STEPS=100,CPDPO_ALPHA="$ALPHA" \
  scripts/slurm/train_reward_overoptimization.sbatch
```

The artifact job automatically writes a non-default value to a separate path,
for example `artifacts/cpdpo/proxy_rm_seed1_alpha_0p2/`. The training job
automatically writes `seed_1/cpdpo_alpha_0p2/`; `seed_1/cpdpo/` remains the
untouched `alpha=0.10` main run. Submit only array index `2` because PPO and
PairPPO do not use alpha and their existing controls are reused. The loader
requires the calibration alpha to equal the training alpha and fingerprints
the selected threshold, so an alpha run cannot resume from or load another
alpha's artifacts. Increasing alpha generally selects a lower conformal order
statistic and therefore makes certification less conservative, but alpha must
not be selected using gold reward.

#### 2. Materialize the shared prompt schedules

The experiment specification does not freeze a numerical rollout horizon, so
choose it before launching and use the same value for every method and seed.
The following example uses 100 rollout steps and the minimum three seeds:

```bash
ROLLOUT_STEPS=100
for SEED in 1 2 3; do
  python scripts/build_prompt_schedule.py \
    --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
    --output "artifacts/prompt_schedules/seed_${SEED}.jsonl" \
    --base-seed "$SEED" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --prompts-per-rollout 256
done
```

Each schedule is shuffled deterministically with `base_seed+30000` and stores
prompt IDs. Every trainer expands each row to adjacent `a` and `b` generations.
PPO flattens the resulting 512 trajectories. PairPPO and CPDPO retain 256
atomic pairs. With batch units 64 responses for PPO and 32 pairs for the pair
methods, all branches perform 32 optimizer updates per rollout.

#### 3. Run the one-rollout three-method smoke

For the fastest end-to-end check, use the proxy RM produced by the earlier RM
smoke and build deliberately reduced CPDPO artifacts. This scores only 64
`D_rm_train` pairs and 64 `D_cal` pairs instead of all 29,813 pairs:

```bash
SMOKE_RM=models/rm-pythia-44m-prompt-disjoint-smoke_seed1
SMOKE_ARTIFACTS=artifacts/cpdpo/proxy_rm_smoke_seed1_quick
SMOKE_OUTPUT=outputs/reward_overoptimization_quick_smoke

ARTIFACT_JOB=$(sbatch --parsable \
  --export=ALL,CPDPO_PROXY_RM_PATH="$SMOKE_RM",CPDPO_ARTIFACT_DIR="$SMOKE_ARTIFACTS" \
  scripts/slurm/prepare_cpdpo_smoke_artifacts.sbatch)
ARTIFACT_JOB=${ARTIFACT_JOB%%;*}
echo "CPDPO smoke artifact job: $ARTIFACT_JOB"
```

The reduced artifacts carry `artifact_scope: smoke`. The full experiment job
rejects them; only the smoke trainer passes the explicit
`--allow-smoke-artifacts` opt-in. They test loading, feature extraction,
geometry, Cholesky factorization, calibration, and fingerprint validation but
are not statistically meaningful.

Create the tiny one-rollout schedule once:

```bash
python scripts/build_prompt_schedule.py \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --output artifacts/prompt_schedules/smoke_seed_1.jsonl \
  --base-seed 1 \
  --rollout-steps 1 \
  --prompts-per-rollout 2

sbatch \
  --dependency=afterok:"$ARTIFACT_JOB" \
  --array=0-2 \
  --export=ALL,CPDPO_PROXY_RM_PATH="$SMOKE_RM",CPDPO_ARTIFACT_DIR="$SMOKE_ARTIFACTS",CPDPO_SMOKE_OUTPUT_ROOT="$SMOKE_OUTPUT" \
  scripts/slurm/smoke_reward_overoptimization.sbatch
```

Array indices map to `0=ppo`, `1=pairppo`, and `2=cpdpo`. This checks the real
policy/proxy models, two independent responses, the ordinary PPO control, the
pair store/loss, CPDPO artifact validation, and checkpoint output. It never
loads the gold RM and is not reportable data.

For a generic-GPU FP16 RM smoke checkpoint, set `SMOKE_RM` to
`models/rm-pythia-44m-prompt-disjoint-smoke-fp16_seed1`. To test only CPDPO,
submit `--array=2`. Each retry must use empty artifact and output directories;
the jobs intentionally refuse to overwrite previous smoke evidence.

After this fast check passes, repeat the same one-rollout matrix with the full
seed-1 RM and the full artifacts from step 1 before starting reportable runs.

#### 4. Launch the full controlled matrix

After the three smoke tasks pass, launch three methods by three seeds:

```bash
sbatch --export=ALL,ROLLOUT_STEPS=100 --array=0-8 \
  scripts/slurm/train_reward_overoptimization.sbatch
```

Change `100` only as a predeclared common design decision. Do not silently
copy the AdvPO paper's 1,500-step horizon: the frozen v1 specification leaves
the horizon configurable, and an additive AdvPO run must use the same declared
horizon as its controls. The full script
uses 256 prompts, two responses per prompt, temperature/top-p 1.0, 128 new
tokens, clip epsilon 0.2, four PPO epochs, and `beta=0.0`. A later practical
track may use one common nonzero beta across all methods.

Training entry points do not accept a gold-model argument. Run metadata records
all four seed namespaces, response/proxy-call budgets, prompt schedule hash,
initial/reference/proxy fingerprints, CPDPO alpha/run identity, and CPDPO
artifact fingerprints.
PairPPO and CPDPO additionally write atomic
`pair_rollouts/rollout_XXXXXX.pt` records containing both response token
sequences and masks, old/reference token log-probabilities, frozen pair
signals, behavior-policy step, and proxy/geometry/calibration fingerprints.

Each rollout-boundary checkpoint also saves the Accelerator model, optimizer,
scheduler, and RNG state plus `experiment_state.json`. To resume one interrupted
array task, resubmit only that task with the same arguments and its checkpoint:

```bash
sbatch --export=ALL,ROLLOUT_STEPS=100,CPDPO_RESUME_CHECKPOINT="$PWD/outputs/reward_overoptimization/seed_1/cpdpo/checkpoints/checkpoint_0320" \
  --array=2 scripts/slurm/train_reward_overoptimization.sbatch
```

The checkpoint must be the latest checkpoint in that exact method/seed output directory. Resume
rejects a changed horizon, code revision, schedule, manifest, policy, proxy RM,
alpha, geometry, or calibration. If the interrupted process logged rollouts newer than
the restored checkpoint, their JSONL is first copied to a timestamped
`rollout_metrics.before_resume_*.jsonl` archive and the active log is rewound to
the checkpoint boundary. Completed runs cannot be resumed.

#### 5. Evaluate checkpoints offline

Gold scoring is a separate job/process after the licensed AlpacaFarm gold RM
has been reconstructed and validated. For each method/seed run:

```bash
python scripts/evaluate_policy_checkpoints.py \
  --run-dir outputs/reward_overoptimization/seed_1/cpdpo \
  --initial-policy assets/initial_sft_policy \
  --reference-policy assets/initial_sft_policy \
  --proxy-rm models/rm-pythia-44m-prompt-disjoint_seed1 \
  --gold-rm /absolute/path/to/reward-model-human \
  --manifest data/processed/alpaca_farm_prompt_disjoint_v1/manifest.json \
  --split D_rl_val_prompts
```

The command evaluates checkpoint 0 and every stored rollout-boundary
checkpoint. For each checkpoint it generates one fresh response per fixed
prompt, persists that response before scoring, and attaches proxy reward, gold
reward, and sampled current/reference KL to that exact response ID. Use
`D_rl_val_prompts` for trajectory/model selection and reserve
`D_rl_test_prompts` for the final locked evaluation.

To visualize a completed seed-1 comparison without launching more policy
seeds, create an explicitly labelled diagnostic:

```bash
python scripts/aggregate_and_plot_reward_overoptimization.py \
  --output-root /storage/scratch1/0/$USER/rlhf-cuq/outputs/reward_overoptimization \
  --split D_rl_val_prompts \
  --diagnostic-seed 1
```

An environment created before the plotting dependency was added can be
repaired without reinstalling the training stack:

```bash
source scripts/configure_cluster_storage.sh \
  "/storage/scratch1/0/$USER/rlhf-cuq"
python -m pip install --no-cache-dir \
  --constraint requirements/legacy-conda.constraints.txt \
  matplotlib==3.7.2
python -m pip check
```

This writes `reward_vs_rollout_step.{png,pdf}` and
`reward_vs_sqrt_kl.{png,pdf}` under `diagnostics/seed_1/`. It compares PPO,
PairPPO, and CPDPO using both proxy and gold rewards but deliberately has no
uncertainty band. It is a single-seed diagnostic, not an across-seed estimate.
The standard reportable aggregation below still requires at least three
identical policy seeds.

After evaluating all three methods and at least three identical seeds:

```bash
python scripts/aggregate_and_plot_reward_overoptimization.py \
  --output-root outputs/reward_overoptimization \
  --split D_rl_val_prompts
```

This consolidates the raw per-run records into
`evaluations/checkpoint_metrics.jsonl` and `.csv`, produces
`mean_by_checkpoint.csv`, `mean_by_kl.csv`, and Figure 2-style PDF/PNG curves
for proxy/gold reward versus rollout step and versus square-root evaluation KL.
Plotting rejects internal PairPPO/CPDPO pair rewards as policy quality metrics.

For the complete three-method/three-seed validation matrix, submit the checked
offline evaluator rather than running the Python command on a login node:

```bash
sbatch --export=ALL,GOLD_RM_PATH=/absolute/path/to/reward-model-human \
  --array=0-8 scripts/slurm/evaluate_reward_overoptimization.sbatch
```

Set `CPDPO_EVAL_SPLIT=D_rl_test_prompts` only for the final locked evaluation.

For the non-default alpha example, evaluate only its CPDPO task and then plot
it against the already evaluated PPO and PairPPO controls:

```bash
sbatch --array=2 \
  --export=ALL,GOLD_RM_PATH=/absolute/path/to/reward-model-human,CPDPO_ALPHA=0.20 \
  scripts/slurm/evaluate_reward_overoptimization.sbatch

python scripts/aggregate_and_plot_reward_overoptimization.py \
  --output-root /storage/scratch1/0/$USER/rlhf-cuq/outputs/reward_overoptimization \
  --split D_rl_val_prompts \
  --diagnostic-seed 1 \
  --cpdpo-alpha 0.20
```

The ablation plots are isolated under
`alpha_ablations/alpha_0p2/diagnostics/seed_1/`; they do not overwrite the
main-alpha plots. The legend records the selected alpha. Do not use
`D_rl_test_prompts` to choose an alpha.

### Exploratory CPDPOv2: fixed SFT reference anchor

CPDPOv2 is additive: it does not modify the PPO, PairPPO, or CPDPOv1 runs
above. It compares each current response with one immutable response sampled
once from the initial SFT policy and optimizes the continuous terminal reward

```text
(proxy_current - proxy_SFT) - q_alpha * uncertainty(current - SFT).
```

The cached SFT response has no PPO ratio, advantage, or gradient. Every
rollout still generates two current responses for each of 256 prompts, so v2
uses the same 512 trainable trajectories, 512 online proxy calls, and 32
optimizer updates as scalar PPO. The one-time SFT generations and proxy calls
are recorded separately in the reference-cache metadata.

First run the complete end-to-end smoke. It prepares a two-prompt cache and
performs one real PPO rollout:

```bash
sbatch scripts/slurm/smoke_cpdpo_v2.sbatch
```

Both its reference-cache and output directories must be empty before a retry.
The default uses the full seed-1 v1 geometry/calibration. If deliberately using
the reduced smoke artifacts, export their path and the explicit opt-in:

```bash
sbatch --export=ALL,CPDPO_ARTIFACT_DIR=artifacts/cpdpo/proxy_rm_smoke_seed1_quick,CPDPO_ALLOW_SMOKE_ARTIFACTS=1 \
  scripts/slurm/smoke_cpdpo_v2.sbatch
```

For the full seed-1 schedule, prepare the immutable references once and then
launch only v2. These commands reuse the existing policy, proxy RM, manifest,
prompt schedule, and full CPDPOv1 artifacts:

```bash
sbatch --export=ALL,CPDPO_V2_SEED=1 \
  scripts/slurm/prepare_cpdpo_v2_references.sbatch

sbatch --array=1 --export=ALL,ROLLOUT_STEPS=100 \
  scripts/slurm/train_cpdpo_v2.sbatch
```

`train_cpdpo_v2.sbatch` uses seed numbers as array IDs (`1`, `2`, `3`), unlike
the three-method v1 array. For an alpha ablation, export the same
`CPDPO_ALPHA` while building v1 artifacts, preparing references, training, and
evaluation. Fingerprints reject a schedule, policy, proxy, alpha, geometry, or
calibration mismatch.

Evaluate seed 1 with the existing isolated evaluator and add v2 to the current
diagnostic plots:

```bash
sbatch --array=1 \
  --export=ALL,GOLD_RM_PATH="$GOLD_RM_PATH" \
  scripts/slurm/evaluate_cpdpo_v2.sbatch

python scripts/aggregate_and_plot_reward_overoptimization.py \
  --output-root /storage/scratch1/0/$USER/rlhf-cuq/outputs/reward_overoptimization \
  --split D_rl_val_prompts \
  --diagnostic-seed 1 \
  --include-cpdpo-v2
```

The four-method diagnostic is isolated under
`cpdpo_v2_comparison/diagnostics/seed_1/` and does not overwrite the v1 plot.
The conformal threshold was calibrated on Coste source preference pairs, not
current/SFT pairs. Its use in v2 therefore assumes exchangeability and should
be reported as an exploratory reference-anchored robust proxy margin, not as a
finite-sample guarantee on gold reward.

### Additive AdvPO: paper-equation method in the Coste/Pythia pipeline

This branch implements the disclosed AdvPO equations, not CPDPO under another
name. Its offline confidence matrix is

```text
M_D = ridge_lambda I + sum_(D_rm_train) [e_chosen e_chosen^T + e_rejected e_rejected^T],
```

and each 64-response PPO scoring batch uses one shared adversarial projection
derived from the current/reference mean feature difference. It does not load
CPDPO pair geometry, conformal calibration, pair loss, or gold reward.

The paper permits reference responses generated by the SFT policy. That is
the compatible choice here because `D_rl_train_prompts` comes from the
AlpacaFarm unlabeled split and has no annotated chosen response. Consequently,
this is the exact disclosed AdvPO method on the current Coste assets and fair
budget, not an exact reproduction of the paper's Section 5.2 LLaMA-7B/data
configuration or unpublished authors' code. The paper does not report a
numeric confidence-matrix ridge; this repository therefore requires an
explicit value for every scientific artifact and records it in the artifact
fingerprint. The smoke-only job explicitly uses `1.0`, which is not a frozen
scientific choice.

First run the end-to-end two-prompt/one-rollout smoke:

```bash
sbatch scripts/slurm/smoke_advpo.sbatch
```

The job builds a 64-preference smoke-only confidence matrix, generates two
fixed SFT references, performs one real scalar PPO rollout, and checks the
saved model/metadata. It refuses to overwrite prior smoke artifacts.

For a full seed-1 run, build the shared confidence matrix once, prepare the
seed schedule's references once, then choose `B=b^2`. The paper's reported
search grid is `1, 5, 10, 15`; do not select `B` using the offline gold RM.

```bash
sbatch --export=ALL,ADVPO_RIDGE_LAMBDA=1.0 \
  scripts/slurm/prepare_advpo_confidence.sbatch

sbatch --export=ALL,ADVPO_SEED=1 \
  scripts/slurm/prepare_advpo_references.sbatch

sbatch --array=1 \
  --export=ALL,ROLLOUT_STEPS=100,ADVPO_B=1,ADVPO_RIDGE_LAMBDA=1.0 \
  scripts/slurm/train_advpo.sbatch
```

The run is saved under
`$CPDPO_OUTPUT_ROOT/seed_1/advpo_B_1/`. Different `B` values cannot overwrite
one another. Resume only from the latest rollout-boundary checkpoint:

```bash
sbatch --array=1 \
  --export=ALL,ROLLOUT_STEPS=100,ADVPO_B=1,ADVPO_RIDGE_LAMBDA=1.0,ADVPO_RESUME_CHECKPOINT=/absolute/path/to/advpo_B_1/checkpoints/checkpoint_0320 \
  scripts/slurm/train_advpo.sbatch
```

Evaluate through the same separate gold-isolated process, then add AdvPO to
the already evaluated PPO/PairPPO/CPDPO diagnostic:

```bash
sbatch --array=1 \
  --export=ALL,GOLD_RM_PATH="$GOLD_RM_PATH",ADVPO_B=1 \
  scripts/slurm/evaluate_advpo.sbatch

python scripts/aggregate_and_plot_reward_overoptimization.py \
  --output-root /storage/scratch1/0/$USER/rlhf-cuq/outputs/reward_overoptimization \
  --split D_rl_val_prompts \
  --diagnostic-seed 1 \
  --include-advpo \
  --advpo-B 1
```

For a reportable comparison, prepare references and train/evaluate the same
predeclared seeds for every method (minimum three). AdvPO metadata records
`B`, `b`, ridge, matrix/reference fingerprints, the batch-level Mahalanobis
shift, adversarial correction, uncertainty, and dynamic scaling factor.
Gold remains unavailable to every preparation and training command.

### Legacy experiment: checked-code PPO baseline on the strict split

Run the single-RM checked-code baseline only after the smoke gate:

```bash
accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_rl.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual prompt_disjoint_data_split_v1 \
  --run_gold_evaluation false \
  --policy_model_path_override assets/initial_sft_policy \
  --proxy_rm_path_override models/rm-pythia-44m-prompt-disjoint_seed1
```

This selects `configs/ppo_config.yaml` (3,000 steps, four rollouts, chunk size two, four PPO epochs, KL coefficient 0.1). Do not describe it as a faithful paper reproduction. The legacy DeepSpeed launcher remains unvalidated because its `auto` batch fields are not wired cleanly through the custom trainer, so the commands here deliberately use the one-process launcher.

The following is retained only as a Coste compatibility example. Do not run it
for the primary CPDPO track. If a later study explicitly adds
the five-member Coste ensemble-optimization baselines, the configured
ensemble-mean run is:

```bash
accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_rl.py \
  --configs defaults defaults_rlhf pythia_rlhf_ensemble prompt_disjoint_data_split_v1 \
  --run_gold_evaluation false \
  --policy_model_path_override assets/initial_sft_policy
```

`pythia_rlhf_ensemble` reads
`models/rm-pythia-44m-prompt-disjoint_seed1` through `seed5` and currently has
`objective_name: mean`. WCO and UWO are separate Coste baseline experiments:
create and archive separate config entries with `objective_name: WCO` or `UWO`,
and set/record `uwo_weight` for UWO, before submitting each run. Do not use
`--proxy_rm_path_override` for an ensemble; that override intentionally accepts
only one RM.

Keep gold scoring in a later, separate GPU process. Once its blockers are resolved, the entry point will be `python src/ppo/run_ppo_gold_eval.py --help`; until then, proxy-only curves are diagnostic and must not be reported as gold-regret results.

### Reproducibility record for every scheduled run

Save the following beside the scheduler stdout/stderr and result directory:

- exact command, merged YAML/config copies, random seed, Slurm job ID, hostname, and requested resources;
- `artifacts/source_manifest.json`, the online audit output, and local asset verification output;
- `python -m torch.utils.collect_env`, `nvidia-smi`, `nvcc --version`, and `python -m pip freeze --all`;
- input/checkpoint SHA-256 manifests and the source state/diff;
- whether the run is smoke, checked-code baseline, paper-reproduction attempt, or a later method experiment.

The rest of this README is the original Coste usage reference. Prefer the commands above for this repository's audited cluster workflow.


## Installation
```
git clone https://github.com/tlc4418/llm_optimization.git
cd llm_optimization
pip install -e .
```

## Provided models and datasets
We provide the following models and datasets on HuggingFace to promote and faciliatte reproducing experiments and future work.

### Models
- [tlc4418/pythia_1.4b_sft_policy](https://huggingface.co/tlc4418/pythia_1.4b_sft_policy): 1.4B Pythia model after SFT on the AlpacaFarm "sft" split. Used as the initial policy model in our experiments.
- [tlc4418/pythia_70m_sft](https://huggingface.co/tlc4418/pythia_70m_sft): 70M Pythia model after SFT on the AlpacaFarm "sft" split. Used as the base model for most of our reward model experiments (before RM training). Multiple reward models can be created from this during [reward model training](#reward-model-training) and it is relatively inexpensive to do so.

### Datasets
- [tlc4418/1.4b-policy_preference_data_gold_labelled](https://huggingface.co/datasets/tlc4418/1.4b-policy_preference_data_gold_labelled): Preference dataset using labels from the AlpacaFarm dataset, generated answers from a 1.4b fine-tuned Pythia policy model, and labelled using the AlpacaFarm "reward-model-human" as a gold reward model. Used to train reward models.
- [tlc4418/gold_labelled_gens](https://huggingface.co/datasets/tlc4418/gold_labelled_gens): Dataset of 12600 answer generations from the 1.4B Pythia SFT model (provided above), using the AlpacaFarm dataset "val" split, and labelled with the AlpacaFarm "reward-model-human" to give "gold" scores. Used for best-of-*n* inference. This dataset is particularly expensive to geenrate and we hope it can help other with future work.
- We provide wrappers and functionality for using these datasets, as well as different parts of the [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) dataset. See the [dataset guide](/src/data_utils/README.md) for details on this and how to use your own datasets.


## Supervised fine-tuning (SFT)
You can easily perform SFT on any HuggingFace or local language models by creating a new entry in the [SFT config](/configs/config.yaml). An example is given for a 70M Pythia model. The following are crucial fields:
| Field name   | Type | Description | Example value |
| ------------ | ---- | ----------- | ------------- |
| `model_name` | str  | name of (path to) the model to be trained | EleutherAI/pythia-70m |
| `output_dir` | str  | name of (path to) the directory where the output model should be saved | models/pythia_model_70m_sft |
| `datasets`   | list | list of instruction datasets to use for training (see [dataset guide](/src/data_utils/README.md) for details) | - alpaca_farm |

Any default hyperparameters can also be overwritten. The 70M Pythia model sample config entry shows a few examples.

Once the config has been set, training can be started with the following command, using the new entry name you created (e.g. "pythia-70m"):
```
accelerate launch --config_file configs/accelerate_config.yaml src/sft/trainer_sft.py --configs defaults {your_sft_config_entry}
```


## Reward model training
Very similar to SFT, you can perform reward model training on any registered HuggingFace or local reward model by creating a new entry in the [RM config](/configs/config_rm.yaml). The following are crucial fields:
| Field name   | Type | Description | Example value |
| ------------ | ---- | ----------- | ------------- |
| `model_name` | str  | name of (path to) the model to be trained. Normal Pythia GPTNeoX models will be automatically converted into a reward model in this training process, such that outputs from the previous SFT training step can be directly fed in. | models/pythia_model_70m_sft |
| `output_dir` | str  | name of (path to) the directory where the output model should be saved. "_seed{rng_seed}" will get appended to it. | models/rm-pythia-44m |
| `datasets`   | list | list of preference datasets to use for training (see [dataset guide](/src/data_utils/README.md) for details) | - alpaca_farm_pref |
| `rng_seed`   | int  | seed which controls the RM training (RM head initialisation and dataset order). This is very useful to create different RMs for a reward model ensemble. It can also be set as a command-line option (see below). | 1 |

Again, default hyperparameters can be overwritten.

Reward model training can then be started with this new config entry (e.g. "rm-pythia-44m"). The `--rng_seed` argument is optional and will otherwise be sourced from the RM config. Command to launch training:
```
accelerate launch --config_file configs/accelerate_config.yaml src/reward_modeling/training/trainer_rm.py --configs defaults_rm {your_rm_config_entry} --rng_seed {your_choice_seed}
```

## Best-of-*n* (BoN) inference
Best-of-*n* inference (also known as re-ranking) can be performed using a base policy model (post-SFT) and one or more trained reward models, as follows:
```
python src/bon/run_bon_pipeline.py {your_reward_models_path}
```
This command is customizable with the following arguments and options:
| Argument        | Type | Required |  Description                                              | Default value              |      
| --------------- | ---- | -------- | --------------------------------------------------------- | -------------------------- |
| `proxy_rm_path`             | str         | yes      | generic path to proxy (non-gold) reward models to use. This should be a string with a "{seed}" placeholder, so that multiple reward models can be retrieved, both for ensembles and general convenience. e.g. "models/rm-pythia-44m_seed{seed}"| - |                                                                           
| `output_dir`                | str         | no       | name of (path to) the directory where the output model should be saved. This will go under [runs/](/runs/). | bon_sampling_{curr_time}                                              
| `gold_gens` | str         | no       | name of (path to) BoN dataset containing at least `big_n` answers (see [dataset guide](/src/data_utils/README.md#best-of-n-bon) for details)| tlc4418/gold_labelled_gens |                                     
| `big_n`                     | int         | no       | total number of answers to perform BoN sampling over. 'N' in the unbiased estimator formula. Usually the total number of answers in the dataset, will be used to cut down the dataset otherwise. | 12600 |                                                                       
| `sample_ns`                 | str         | no       | list of indexes (the *n* in best-of-*n*) at which to perform BoN sampling. Comma-separated list of ints. | "1,2,4,8,16,32,64,128,<br>256,512,1024,2048,<br>4096,6144,8192,12500" |
| `seeds`                     | str         | no       | list of seeds corresponding to which reward models to run (will be used to fill the "{seed}" placeholder for the `proxy_rm_path`). If doing BoN for ensembles, the length of this list is also your ensemble cardinality. Comma-separated list of ints. | "1,2,3,4,5"  |                                                                
| `ensembles`                 | bool        | no       | whether to run BoN over ensembles. If set to true, BoN for the three types of ensembles in the paper (mean, WCO, UWO) will be performed in addition to the individual reward models. | True  |                                                                       
| `uwo_weights`               | str         | no       | list of UWO weights to use when doing BoN sampling with the UWO ensemble (if `ensembles` is true). Results will be given for a new ensemble with each weight. Comma-separated list of floats. | "0.5" |                                                                       


A help function is also provided (`--help`) which will display a condensed version of the above, including the appropriate flag names for specifying each parameter.

Our implementation uses an unbiased estimator (see paper for details) for robust and unbiased results.

The relevant result of running this command will be a "bon_sampled_results.json" results file for each run (one for each individual seed and one for each ensemble if desired). This file contains a list of dictionary entries, where each entry contains the sampled index *n*, the proxy reward model score at that *n*, and the corresponding gold reward model score at that *n*, as follows:
```
[
    {
        "n": {some_int},
        "proxy_score": {proxy_score},
        "gold_score": {gold_score}
    },

    ...
]
```
These data points can then be used to plot the BoN performance of different policies according to both proxy and gold reward model score, as a function of *n*.


## RL training (PPO)
RL training with PPO can be performed using a base policy model (post-SFT) and one or more trained reward models. The policy will be trained using PPO and the reward models as a reward function. We provide options for both single reward models and ensembles, along with conservative optimization techniques. Example configs entries for this are provided in the [RL config](/configs/config_rl.yaml). We detail some of the following important fields (which will likely be changed the most) for a new config entry:

| Field name    | Subfield name      | Type  |  Description                                                              | Example value |
| ------------- | ------------------ | ----- | ----------------------------------------------------------------------- | ------------- |
| `output_dir`  | -                  | str   | name of (path to) the directory where the outputs should be saved. | runs/ppo |
| `datasets`    | -                  | list  | list of instruction datasets to use for training (see [dataset guide](/src/data_utils/README.md) for details) | - alpaca_farm |
| `gold_config` | `model_name`       | str   | name of (path to) the 'gold' reward model to use to evaluate policy outputs alongside the proxy reward mdoel rewards | alpaca_farm_models/<br>reward-model-human
|               | `is_alpacafarm_rm` | bool  | whether or not the gold reward model is from AlpacaFarm. Because the input format is different for their models, different dataset processing is needed for evaluation. | true
| `rank_config` | `model_names`      | str   | list of names/paths for the proxy reward models. A single path should be provided for single reward model PPO training, and several paths for ensembles. When providing a single path, it can also contain a '{seed}' placeholder, which can be replaced by passing an additional `--rm_seed {your_seed}` argument to the run command given below. This can be helpful when wanting to run multiple seeds in parallel using job managers, without requiing a different config each time. | - models/rm-pythia-44m_seed1
|               | `objective_name`   | str   | name of the conservative optimization objective to use to combine rewards from the reward model ensemble members. Must be one of "mean", "WCO", or "UWO". Only used if more than one model is provided in `model_names` | mean
|               | `uwo_weight`       | float | weight (&lambda; coefficient) for UWO conservative optimization objective. Only used when `objective_name` is "UWO". | 0.1
| `sft_config`  | `model_name`       | str   | name of (path to) the policy model (post-SFT) to use for PPO training. This is the model that will be trained. | tlc4418/pythia_1.4b_sft_policy


Many default hyperparameters can also be overwritten, both in the [RL config](/configs/config_rl.yaml) and more specific PPO ones in the [PPO config file](/configs/config_ppo.yaml)


PPO training (and subsequent gold reward model evaluation) can be started with the following command:
```
accelerate launch --config_file configs/accelerate_config.yaml src/ppo/trainer_rl.py --configs defaults defaults_rlhf {your_rm_config_entry}
```

Gold reward model evaluation is performed after training is completed so as to avoid loading the large gold reward mdoel during training. KL divergence from the initial policy is also recorded at each evaluation step, such that the evaluation results in the "eval/" output folder can be used to plot both proxy and gold reward model score either as a function of steps or of KL divergence. Please see the [AlpacaFarm respository](https://github.com/tatsu-lab/alpaca_farm) for instructions on downloading the 7B reward model used as the gold reward model in our paper (`reward-model-human`)


## Example usage (full pipeline) - single reward model

### SFT models
Simply use the provided* [tlc4418/pythia_1.4b_sft_policy](https://huggingface.co/tlc4418/pythia_1.4b_sft_policy) and [tlc4418/pythia_70m_sft](https://huggingface.co/tlc4418/pythia_70m_sft) models.

Alternatively, you could run your own SFT training for the policy model and base reward model, following the [SFT instructions](#supervised-fine-tuning-sft) above.

### Reward models
Train a reward model from the base reward model (post-SFT), with any rng seed (here "1"):
```
accelerate launch --config_file configs/accelerate_config.yaml src/reward_modeling/training/trainer_rm.py --configs defaults_rm rm-pythia-44m --rng_seed 1
```

### BoN sampling
Run BoN sampling for the trained reward model:
```
python src/bon/run_bon_pipeline.py models/rm-pythia-44m_seed{seed} --seeds 1
```

### PPO RL training
For PPO, run the following command:
```
accelerate launch --config_file configs/accelerate_config.yaml src/ppo/trainer_rl.py --configs defaults defaults_rlhf pythia_rlhf_individual
```

Here we set the config to already contain the trained reward model under the rank_config's `model_names`, but you will want to change this to match yours when customizing this process.


## Example usage (full pipeline) - ensemble of *k* reward models

### SFT models
Simply use the provided [tlc4418/pythia_1.4b_sft_policy](https://huggingface.co/tlc4418/pythia_1.4b_sft_policy) and [tlc4418/pythia_70m_sft](https://huggingface.co/tlc4418/pythia_70m_sft) models.

Alternatively, you could run your own SFT training for the policy model and base reward model, following the [SFT instructions](#supervised-fine-tuning-sft) above.

### Reward models
Train as many (*k*) reward models as you want from the base reward model (post-SFT). Just run the following *k* times, changing the seed each time (e.g. 1-5):
```
accelerate launch --config_file configs/accelerate_config.yaml src/reward_modeling/training/trainer_rm.py --configs defaults_rm rm-pythia-44m --rng_seed {seed}
```

### BoN sampling
Assuming you have trained 5 reward models above, this will run BoN sampling for all 5 models and the 3 ensembles types (mean, WCO, UWO):
```
python src/bon/run_bon_pipeline.py models/rm-pythia-44m_seed{seed} --seeds 1,2,3,4,5 --ensembles
```

### PPO RL training
For PPO, you will want to run the following command for each ensemble you want to train, making sure the appropriate conservative optimization `objective` and `uwo_weight` (if applicable) are set in the [RL config]((/configs/config_rl.yaml)):
```
accelerate launch --config_file configs/accelerate_config.yaml src/ppo/trainer_rl.py --configs defaults defaults_rlhf pythia_rlhf_ensemble
```

Here we set the config to already contain the 5 trained reward models under the rank_config's `model_names`, but you will want to change these to match yours when customizing this process.




## Citation
If you use the models, data, or code in this repo, please consider citing our work.
```
@article{coste2023reward,
  title={Reward model ensembles help mitigate overoptimization},
  author={Coste, Thomas and Anwar, Usman and Kirk, Robert and Krueger, David},
  journal={arXiv preprint arXiv:2310.02743},
  year={2023}
}
```
