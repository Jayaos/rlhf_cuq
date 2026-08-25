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
| Proxy RM, seeds 1--5 | Produce the five members used by the Coste ensemble | Same as above | Runnable as a job array after seed 1 passes |
| PPO integration smoke | One optimizer update using the real 1.4B policy and one RM | Policy + prompts + trained seed-1 RM | Runnable on one GPU; setup check only |
| Checked Coste single-RM PPO | Run the vendored `configs/ppo_config.yaml` for 3,000 steps | Same as smoke | Runnable only after the smoke gate; gold must stay off |
| Checked Coste ensemble PPO | Run the five local RMs with the configured mean/WCO/UWO objective | Policy + prompts + five trained RMs | Mean is configured; set and record WCO/UWO fields for separate runs |
| Offline gold scoring | Measure reward overoptimization without exposing gold reward online | Reconstructed AlpacaFarm 7B RM | Blocked: licensed base, reconstruction pinning, and prompt formatter validation remain open |
| AdvPO and proposed conformal methods | Section 5.1/5.2 and new uncertainty experiments | Method code and frozen equations | Not implemented yet; there is no valid launch command |

The checked PPO YAML uses `num_rollouts=4` and `chunk_size=2`; the Coste paper reports 256 and 32. Treat this as the checked-code baseline, not a paper-faithful reproduction. Do not start a reportable large run until the one-update smoke saves and resumes a checkpoint and its KL accounting is understood. See [the implementation plan](docs/IMPLEMENTATION_PLAN.md), [open method decisions](docs/OPEN_METHOD_DECISIONS.md), and [environment lock notes](docs/ENVIRONMENT_LOCK.md) for the acceptance gates.

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
export HF_HOME="$SCRATCH_ROOT/huggingface"
export HF_DATASETS_CACHE="$SCRATCH_ROOT/huggingface/datasets"
export PIP_CACHE_DIR="$SCRATCH_ROOT/pip-cache"
mkdir -p "$SCRATCH_ROOT" "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"

conda env create --file environment.cluster.yml
conda activate rlhf-cuq
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

python -m pip install --no-build-isolation \
  --constraint requirements/legacy-conda.constraints.txt \
  --requirement requirements/legacy-runtime.txt
python -m pip install --no-build-isolation --no-deps \
  --requirement requirements/legacy-sources.txt
python -m pip install --no-build-isolation --no-deps --editable .
python -m pip check
```

`environment.cluster.yml` installs the compiled foundation first. The staged pip commands are intentional: FlashAttention must see the installed Torch/CUDA toolchain, while pinned Open Assistant must be installed editably with `--no-deps` because its wheel metadata omits imported subpackages and contains moving VCS dependencies. Conda supports creating an environment from a YAML file with `conda env create -f ...`; see the [official command reference](https://docs.conda.io/projects/conda/en/latest/commands/env/create.html).

Validate imports on the build node:

```bash
python --version
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "from model_training.custom_datasets.formatting import format_pairs; from model_training.models.reward_model import GPTNeoXRewardModel; from model_training.utils.utils import read_yamls; import alpaca_farm, oasst_data, trlx"
python -m unittest discover --start-directory tests --verbose
python scripts/audit_assets.py --offline
```

Then, inside an allocated GPU job, require this check to print a visible GPU and `True`:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name()); print(torch.cuda.is_available())"
```

This is a candidate legacy environment, not yet a fully resolved GPU lock. After the first successful smoke, save `python -m pip freeze --all`, `python -m torch.utils.collect_env`, the scheduler resource request, and all checkpoint checksums with the run artifacts.

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
the pinned source payloads, creates content-derived record/prompt IDs, keeps
duplicate prompts in one role, uses deterministic hash assignment with exact
largest-remainder quotas, audits overlap, and writes SHA-256-protected JSONL and
ID files. It never overwrites an existing split bundle.

```bash
python scripts/build_data_manifest.py --config configs/data_split_coste_v1.yaml
python scripts/build_data_manifest.py --config configs/data_split_coste_v1.yaml --verify-only
sha256sum data/processed/coste_split_v1/manifest.json
```

For the audited source sizes, the controlled roles are:

| Source pool | Logical role | Fraction | Rows |
| --- | --- | ---: | ---: |
| Coste preference `train` (49,383) | `D_rm_train` | 90% | 44,445 |
| Coste preference `train` | `D_rm_val` | 5% | 2,469 |
| Coste preference `train` | `D_cal` | 5% | 2,469 |
| AlpacaFarm `unlabeled` (20,000) | `D_rl_train_prompts` | 80% | 16,000 |
| AlpacaFarm `unlabeled` | `D_rl_val_prompts` | 10% | 2,000 |
| AlpacaFarm `unlabeled` | `D_rl_test_prompts` | 10% | 2,000 |

The original preference `validation` and AlpacaFarm `val` splits are retained
as `D_rm_external_val` and `D_rl_external_val`; they are not mixed into the
percentages. The generated manifest is authoritative if a pinned legacy content
filter rejects any PPO source row. Archive its hash with every run.

The RM adapter exposes only `D_rm_train` and `D_rm_val`; the PPO adapter exposes
only `D_rl_train_prompts` and `D_rl_val_prompts`. `D_cal`, test, and external
validation roles cannot enter either online trainer through this adapter. The
fixed `D_cal` is reserved for later calibration code; no conformal score or
adaptive-update equation is selected by this pipeline.

Use this manifest route for the current and future controlled experiments. For
an exact legacy-data regression only, run the original `trainer_rm.py` with
`rm-pythia-44m-cluster`, or omit `coste_data_split_v1` from PPO and supply
`--rl_dataset_path_override assets/alpaca_farm_prompt_dataset`. Label those
runs `legacy_split`; they use the original implicit first-N/source-validation
selection and are not directly interchangeable with `coste_split_v1` results.

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

This does not download LLaMA-7B and does not make gold scoring ready. The pinned AlpacaFarm recovery utility still needs a reviewed way to consume these exact local revisions, and `src/data_utils/rm_dataset_formatter.py` currently emits literal `{instruction}`/`{input_}` placeholders on its AlpacaFarm branch. Keep `--run_gold_evaluation false` until that formatter is fixed against a pinned fixture and reconstructed `model_sum.txt` validation passes. Never commit or redistribute licensed/reconstructed weights.

### Experiment 1: train proxy reward model(s)

First train seed 1. The final overlay selects the local base model and the
manifest-backed `D_rm_train`/`D_rm_val`; all Coste RM hyperparameters still come
from the vendored `rm-pythia-44m` entry.

```bash
conda activate rlhf-cuq
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/reward_modeling/training/trainer_rm_manifest.py \
  --configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-split \
  --rng_seed 1
```

Expected output: `models/rm-pythia-44m_seed1`. Confirm that it contains model/tokenizer files and a finite saved reward normalization mean/std, then hash it:

```bash
mkdir -p artifacts/checksums
find models/rm-pythia-44m_seed1 -type f -print0 \
  | sort -z | xargs -0 sha256sum \
  > artifacts/checksums/rm-pythia-44m_seed1.sha256
```

Only after seed 1 passes, train seeds 1--5 for ensemble experiments. A generic Slurm array body is:

```bash
#!/bin/bash
#SBATCH --job-name=rm44m
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
source "/path/to/conda/etc/profile.d/conda.sh"
conda activate rlhf-cuq
cd "/absolute/path/to/rlhf-cuq"
export CUDA_HOME="$CONDA_PREFIX"
export HF_HOME="/absolute/path/on/shared/scratch/$USER/rlhf-cuq/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/reward_modeling/training/trainer_rm_manifest.py \
  --configs defaults_rm rm-pythia-44m rm-pythia-44m-cluster-split \
  --rng_seed "$SLURM_ARRAY_TASK_ID"
```

Replace the Conda, repository, account/partition, GPU, memory, and wall-time values for the local scheduler. Do not launch one multi-GPU RM job: each array element is an independent one-GPU seed.

### Experiment 2: one-update PPO smoke

Request one Ampere-or-newer GPU (40 GiB is a conservative starting request), activate the environment, set the offline variables, and run:

```bash
accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_rl.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual coste_data_split_v1 baseline_smoke \
  --policy_model_path_override assets/initial_sft_policy \
  --proxy_rm_path_override models/rm-pythia-44m_seed1
```

This uses two rollouts, one PPO epoch, one optimizer update, two evaluation prompts, and no gold load. It is an integration test, not a scientific experiment. Inspect `runs/ppo_smoke`, verify finite proxy/KL values, verify a checkpoint can be reloaded, and capture the environment before crossing Gate 1. The legacy evaluator can still generate up to 256 tokens even though smoke rollouts are capped at 16.

### Experiment 3: checked Coste PPO baselines

Run the single-RM checked-code baseline only after the smoke gate:

```bash
accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_rl.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual coste_data_split_v1 \
  --run_gold_evaluation false \
  --policy_model_path_override assets/initial_sft_policy \
  --proxy_rm_path_override models/rm-pythia-44m_seed1
```

This selects `configs/ppo_config.yaml` (3,000 steps, four rollouts, chunk size two, four PPO epochs, KL coefficient 0.1). Do not describe it as a faithful paper reproduction. The legacy DeepSpeed launcher remains unvalidated because its `auto` batch fields are not wired cleanly through the custom trainer, so the commands here deliberately use the one-process launcher.

After all five RM directories exist, the configured ensemble-mean run is:

```bash
accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_rl.py \
  --configs defaults defaults_rlhf pythia_rlhf_ensemble coste_data_split_v1 \
  --run_gold_evaluation false \
  --policy_model_path_override assets/initial_sft_policy
```

`pythia_rlhf_ensemble` reads `models/rm-pythia-44m_seed1` through `seed5` and currently has `objective_name: mean`. WCO and UWO are separate Coste baseline experiments: create and archive separate config entries with `objective_name: WCO` or `UWO`, and set/record `uwo_weight` for UWO, before submitting each run. Do not use `--proxy_rm_path_override` for an ensemble; that override intentionally accepts only one RM.

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
