# Environment lock and setup status

## Status

The source graph is pinned, but the GPU environment is not yet a validated lock. Stage 0 ran on Windows with Python 3.13.2 and no PyTorch, Transformers, datasets, Accelerate, DeepSpeed, trlx, Open Assistant, or AlpacaFarm installation. That host can run the standard-library audit/tests but cannot honestly validate the legacy CUDA training stack.

The target compatibility envelope is Linux x86-64, Python 3.10, CUDA 11.8, and PyTorch 2.0.1+cu118. `pyproject.toml` enforces Python `>=3.10,<3.11`. `requirements/legacy-sources.txt` pins every VCS dependency to a commit, `requirements/legacy-runtime.txt` enumerates their direct runtime requirements, and `requirements/legacy-cu118.constraints.txt` freezes the high-risk binary/API packages. This is a candidate setup derived from the pinned trlx requirements and Open Assistant’s stricter package requirements—not a complete, hash-locked transitive environment.

Do not publish GPU results until a compatible host resolves this environment, passes the tiny baseline, and exports a complete lock (`pip freeze` plus wheel/direct-URL hashes or an immutable container digest).

## Pinned source graph

| Component | Package metadata version | Source/revision |
|---|---|---|
| project/Coste snapshot | `llm_optimization` 0.0.1 | `tlc4418/llm_optimization@416b03cc2c3c8125208679acd88891584d9eefd2` |
| Open Assistant | `model_training` 1.0.0; `oasst_data` 1.0.0 | `LAION-AI/Open-Assistant@e1769c102f1597cc0b53a8b915f858239d197aeb` |
| `trlx` | metadata version 0.7.0 (not the tag) | `CarperAI/trlx@3340c2f3a56d1d14fdd5f13ad575121fa26b6d92` |
| `alpaca_farm` | 0.2.0 | `tlc4418/alpaca_farm@f92bd550130975436301ba02137b303d1eb59986` |

The direct source file replaces moving `trlx` and AlpacaFarm references. The selected trlx commit reports package version 0.7.0 but is not the 0.7.0 tag, so commit identity—not the version string—is authoritative.

Do not move these VCS entries back into ordinary root dependencies. The pinned OA model metadata declares its own moving direct `trlx` URL, which conflicts with a SHA-pinned direct URL in modern pip. Its built wheel also contains only the top-level `model_training` package and omits imported `custom_datasets`, `models`, and `utils` subpackages. Installing the pinned source editably and with `--no-deps` retains the source tree and bypasses both defects; the separately installed runtime list supplies dependencies.

## Compatibility constraints

The constraints file selects:

- Python 3.10 only;
- PyTorch 2.0.1+cu118 and Triton 2.0.0;
- Accelerate 0.22.0 and DeepSpeed 0.10.1;
- datasets 2.14.4, PyArrow 13.0.0, and Hugging Face Hub 0.16.4;
- Transformers 4.31.0, Tokenizers 0.13.3, PEFT 0.2.0, and Pydantic 1.10.7, following the pinned OA package rather than trlx’s later 4.32/0.5/1.10.12 environment snapshot;
- the remaining high-risk versions listed in `requirements/legacy-cu118.constraints.txt`.

Open Assistant still declares lower-bounded packages such as flash-attn and bitsandbytes, and AlpacaFarm brings additional dependencies. A successful resolver may therefore still select unrecorded transitive versions. Freeze those only after the target CUDA host proves they build and the smoke run passes. Never reinterpret this constraints file as a complete lock.

## Target-host installation procedure

The preferred cluster path is now the staged Conda procedure in the root
`README.md`: create `environment.cluster.yml`, then install the constrained
runtime and SHA-pinned editable sources. `requirements/legacy-conda.constraints.txt`
uses `torch==2.0.1`, matching Conda's version metadata, while the environment
selects the CUDA 11.8 build.

The following virtual-environment route remains an alternative on a CUDA
11.8-compatible Linux host:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --requirement requirements/legacy-build.txt
python -m pip install \
  --constraint requirements/legacy-cu118.constraints.txt \
  torch ninja cmake packaging
python -m pip install --no-build-isolation \
  --constraint requirements/legacy-cu118.constraints.txt \
  --requirement requirements/legacy-runtime.txt
python -m pip install --no-build-isolation --no-deps --requirement requirements/legacy-sources.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python -c "from model_training.custom_datasets.formatting import format_pairs; from model_training.models.reward_model import GPTNeoXRewardModel; from model_training.utils.utils import read_yamls; import trlx, alpaca_farm, oasst_data"
```

Torch is installed before the remaining runtime specifically because `flash-attn==2.0.8` imports Torch while building. Build isolation is then disabled so that build sees the pinned Torch, Ninja, CMake, and packaging tools. A CUDA toolkit/compiler compatible with the selected PyTorch build is still required. This sequence is structurally installable but remains target-host-unvalidated; its first successful resolution must be frozen completely.

Then record, before training:

```bash
python --version
nvidia-smi
python -m torch.utils.collect_env
python -m pip freeze --all
git diff --no-index upstream_coste .
python scripts/audit_assets.py --output artifacts/source_audit_online.txt
```

The last `git diff --no-index` example is meaningful only if a clean pinned Coste checkout is present outside the deliverable workspace; do not vendor that temporary checkout into experiment artifacts.

## Asset acquisition contract

Run the metadata-only audit first:

```bash
python scripts/audit_assets.py
```

Download model/dataset payloads only at revisions in `artifacts/source_manifest.json`. Verify each required Git/LFS digest after download. Hugging Face `from_pretrained` and `load_dataset` calls in the legacy code do not pass revisions. Stage 0 therefore adds opt-in local-path overrides to the PPO entry point; use them with offline Hub mode for smoke/reportable runs. The defaults retain the original identifiers for baseline comparison.

The repository now provides a manifest-driven downloader that creates the
standard local directories and verifies every required downloaded file:

```bash
python scripts/download_assets.py --asset-root assets
python scripts/download_assets.py --asset-root assets --verify-only
```

`audit_assets.py` verifies remote metadata and the vendored Coste hashes;
`download_assets.py` performs the separate local payload digest check. The
proxy path remains a locally trained/checksummed checkpoint, not a Hub
download.

Before RM or PPO training, build and verify the controlled split bundle:

```bash
python scripts/build_data_manifest.py --config configs/data_split_coste_v1.yaml
python scripts/build_data_manifest.py --config configs/data_split_coste_v1.yaml --verify-only
sha256sum data/processed/coste_split_v1/manifest.json
```

The split builder requires the pinned `datasets` and Open-Assistant installs,
because it loads the local Hub snapshots and applies the same
`_filter_by_words` rule as the Coste loader. Its manifest hash and all explicit
membership-ID files are run inputs. The generated `data/processed/` directory
is ignored by Git and must be retained on shared storage or regenerated from
the identical pinned inputs and config.

The proxy path in `configs/config_rl.yaml` is not a released checkpoint. Train seed 1 from the pinned 70M SFT base using the legacy RM command, then record:

- resolved data manifest hash and row IDs;
- training config and seed;
- base revision;
- complete checkpoint hashes;
- scalar-head weight and bias hashes;
- validation results.

Do not begin baseline PPO while `models/rm-pythia-44m_seed1` is merely a placeholder.

For the gold evaluator, obtain LLaMA-7B under its original access/license terms, record its conversion tool and hashes, then use the pinned AlpacaFarm code to reconstruct pinned `sft10k` followed by pinned `reward-model-human`. The upstream command shape is:

```bash
python -m pretrained_models.recover_model_weights \
  --llama-7b-hf-dir /licensed/path/to/llama-7b-hf \
  --alpaca-farm-model-name reward-model-human \
  --models-save-dir alpaca_farm_models
```

The upstream recovery utility currently resolves weight-diff repositories without explicit revisions. Before use, either populate its local cache from the manifest-pinned snapshots or add a separately reviewed wrapper that supplies the revisions. Require `model_sum.txt` verification and hash the reconstructed model. Never commit or redistribute the licensed base/reconstructed weights through this repository.

## Tiny baseline smoke profile

The opt-in profile changes only scale and skips the separate gold phase:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
accelerate launch --config_file configs/accelerate_config_simple.yaml \
  src/ppo/trainer_rl.py \
  --configs defaults defaults_rlhf pythia_rlhf_individual coste_data_split_v1 baseline_smoke \
  --policy_model_path_override assets/initial_sft_policy \
  --proxy_rm_path_override models/rm-pythia-44m_seed1
```

It selects `configs/ppo_config_smoke.yaml`: two rollouts in one chunk, batch two, one PPO epoch, one optimizer update, two eval prompts, rollout generation capped at 16 new tokens, and a dedicated checkpoint directory. Two samples avoid the legacy trlx singleton-variance/`RunningMoments` NaN. The trainer’s evaluation path hard-codes up to 256 new tokens, so the 16-token cap does not apply to its initial and post-update evaluation. The smoke still uses the real 1.4B policy and locally trained proxy RM, so it is a setup/integration check—not a CPU unit test and not scientific evidence. `run_gold_evaluation=false` prevents a licensed 7B gold model from being loaded in the training process. Run offline gold scoring later, in a separate process, only after the scorer path is corrected and validated.

The smoke command intentionally uses the existing single-process, non-DeepSpeed launcher. The default DeepSpeed JSON contains `auto` batch and accumulation fields, while the custom trainer constructs `Accelerator()` without forwarding the YAML accumulation setting; those values must be captured and validated separately before using the DeepSpeed launcher for reported runs.

Acceptance requires the smoke run to save generations/proxy scores, produce understood KL, save and resume a checkpoint, and prove that a deliberately failing gold loader is never reached online. None of those GPU acceptance claims has been made in Stage 0.

## Producing the final lock

On the first successful target host:

1. Save OS, driver, GPU, CUDA, compiler, and Python reports.
2. Export all resolved packages/direct URLs and hash the export.
3. Record installed VCS commit IDs independently of package version labels.
4. Capture DeepSpeed’s resolved micro/global batch settings and Accelerate world size.
5. Run unit tests, read-only online asset audit, proxy feature identity test, one-update smoke, and checkpoint resume.
6. Build an immutable image or lock file and record its digest in a new experiment manifest.

If any dependency must change, retain this legacy track and document the new stack as a compatibility patch. A later modern PPO port must have separate results and a baseline-equivalence study.
