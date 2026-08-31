"""Train one fair-budget PPO, PairPPO, CPDPO, CPDPOv2, or AdvPO branch."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import transformers
import trlx
from model_training.custom_datasets.formatting import format_pairs
from model_training.utils.utils import init_rng, read_yamls
from trlx.data.configs import TRLConfig

from src.advpo.reward import AdvPORewardCallback
from src.advpo.spec import AdvPOConfig, advpo_run_name
from src.cpdpo.artifacts import atomic_write_json, git_revision, model_fingerprint, sha256_file
from src.cpdpo.pair_reward import PairRewardCallback
from src.cpdpo.prompt_schedule import ExperimentSeeds, load_schedule_as_duplicated_prompts
from src.cpdpo.reference_anchor import ReferenceAnchoredRewardCallback
from src.cpdpo.spec import (
    ALPHA,
    TRAINING_METHODS,
    CPDPOConfig,
    CPDPOV2Config,
    is_main_alpha,
    method_run_name,
)
from src.data_utils.manifest_dataset_loader import get_manifest_dataset
from src.ppo.custom_helpers import get_reward_fn
from src.ppo.custom_trlx_trainers.custom_accelerate_pair_ppo_trainer import (  # noqa: F401
    CustomAcceleratePairPPOTrainer,
)
from src.ppo.custom_trlx_trainers.experiment_ppo_trainer import (  # noqa: F401
    ExperimentAcceleratePPOTrainer,
)
from src.ppo.runtime_config import apply_local_asset_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--method", choices=sorted(TRAINING_METHODS), required=True)
    parser.add_argument("--prompt-schedule", required=True)
    parser.add_argument("--rollout-steps", type=int, required=True)
    parser.add_argument("--prompts-per-rollout", type=int, default=256)
    parser.add_argument("--checkpoint-every-rollouts", type=int, default=10)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--proxy-rm", required=True)
    parser.add_argument("--geometry")
    parser.add_argument("--calibration")
    parser.add_argument("--reference-cache")
    parser.add_argument("--advpo-confidence")
    parser.add_argument(
        "--advpo-B",
        dest="advpo_B",
        type=float,
        help="AdvPO confidence radius squared B=b^2 (paper grid: 1, 5, 10, 15)",
    )
    parser.add_argument("--output-root", default="outputs/reward_overoptimization")
    parser.add_argument("--ppo-config", default="configs/ppo_config_reward_overoptimization.yaml")
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--pair-chunk-size", type=int, default=64)
    parser.add_argument("--proxy-batch-size", type=int, default=64)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="CPDPO conformal alpha; non-default values create a named alpha-ablation run",
    )
    parser.add_argument("--reward-variant", choices=["robust_margin", "sign_only"], default="robust_margin")
    parser.add_argument("--geometry-mode", choices=["full", "unit"], default="full")
    parser.add_argument(
        "--allow-smoke-artifacts",
        action="store_true",
        help="Permit explicitly tagged reduced-data artifacts in a smoke-only robust/additive run",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume an interrupted run from one of its rollout-boundary Accelerator checkpoints",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def merged_training_config(names: list[str]) -> Namespace:
    all_configs = read_yamls("./configs")
    merged = {}
    for name in names:
        if name not in all_configs:
            raise KeyError(f"Unknown configuration overlay: {name}")
        merged.update(all_configs[name])
    return Namespace(**merged)


def validate_schedule(
    path: Path,
    rollout_steps: int,
    prompts_per_rollout: int,
    *,
    expected_manifest: Path,
    expected_base_seed: int,
) -> dict:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prompt schedule metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = rollout_steps * prompts_per_rollout
    if metadata.get("row_count") != expected:
        raise ValueError(f"Schedule has {metadata.get('row_count')} rows; expected {expected}")
    if metadata.get("rollout_steps") != rollout_steps or metadata.get("prompts_per_rollout") != prompts_per_rollout:
        raise ValueError("Schedule dimensions do not match the requested experiment")
    if metadata.get("schedule_sha256") != sha256_file(path):
        raise ValueError("Prompt schedule hash mismatch")
    if metadata.get("manifest_sha256") != sha256_file(expected_manifest):
        raise ValueError("Prompt schedule was built from a different data manifest")
    if metadata.get("base_seed") != expected_base_seed:
        raise ValueError("Prompt schedule base seed does not match this run")
    if metadata.get("resolved_seeds") != asdict(ExperimentSeeds.from_base(expected_base_seed)):
        raise ValueError("Prompt schedule seed namespaces do not match this run")
    if metadata.get("responses_per_prompt") != 2:
        raise ValueError("Prompt schedule must declare exactly two responses per prompt")
    return metadata


def main() -> None:  # noqa: C901
    args = parse_args()
    if args.rollout_steps < 1 or args.prompts_per_rollout < 1:
        raise ValueError("rollout-steps and prompts-per-rollout must be positive")
    if args.checkpoint_every_rollouts < 1:
        raise ValueError("checkpoint interval must be positive")
    cpdpo_methods = {"cpdpo", "cpdpo_v2"}
    if args.method in cpdpo_methods and (not args.geometry or not args.calibration):
        raise ValueError(f"{args.method} requires --geometry and --calibration")
    if args.method not in cpdpo_methods and (args.geometry or args.calibration):
        raise ValueError("Only CPDPO methods may load geometry/calibration artifacts")
    if args.method in {"cpdpo_v2", "advpo"} and not args.reference_cache:
        raise ValueError(f"{args.method} requires --reference-cache")
    if args.method not in {"cpdpo_v2", "advpo"} and args.reference_cache:
        raise ValueError("--reference-cache is valid only for CPDPOv2 or AdvPO")
    if args.method == "advpo":
        if not args.advpo_confidence or args.advpo_B is None:
            raise ValueError("AdvPO requires --advpo-confidence and --advpo-B")
        if args.advpo_B <= 0.0:
            raise ValueError("--advpo-B must be positive")
    elif args.advpo_confidence or args.advpo_B is not None:
        raise ValueError("AdvPO confidence/B arguments are valid only for AdvPO")
    if args.method not in cpdpo_methods and (
        args.reward_variant != "robust_margin" or args.geometry_mode != "full"
    ):
        raise ValueError("CPDPO ablation flags cannot be applied to PPO or PairPPO")
    if args.method == "cpdpo_v2" and (
        args.reward_variant != "robust_margin" or args.geometry_mode != "full"
    ):
        raise ValueError("The first CPDPOv2 diagnostic freezes robust-margin/full-geometry settings")
    if args.allow_smoke_artifacts and args.method not in cpdpo_methods | {"advpo"}:
        raise ValueError("--allow-smoke-artifacts is valid only for robust/additive methods")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if args.method not in cpdpo_methods and not is_main_alpha(args.alpha):
        raise ValueError("--alpha applies only to CPDPO methods; controls are unchanged")

    run_variant = (
        advpo_run_name(args.advpo_B)
        if args.method == "advpo"
        else method_run_name(args.method, args.alpha)
    )
    experiment_track = (
        ("cpdpo_alpha_ablation" if args.method == "cpdpo" else "cpdpo_v2_alpha_ablation")
        if args.method in cpdpo_methods and not is_main_alpha(args.alpha)
        else (
            "cpdpo_v2_exploratory"
            if args.method == "cpdpo_v2"
            else ("advpo_additive" if args.method == "advpo" else "main")
        )
    )

    schedule_path = Path(args.prompt_schedule).resolve()
    training_conf = merged_training_config(args.configs)
    training_conf.rng_seed = args.base_seed
    training_conf.rm_seed = None
    training_conf.local_rank = args.local_rank
    training_conf.policy_model_path_override = args.policy_model
    training_conf.proxy_rm_path_override = args.proxy_rm
    training_conf.rl_dataset_path_override = ""
    rank_config = Namespace(**training_conf.rank_config)
    sft_config = Namespace(**training_conf.sft_config)
    apply_local_asset_overrides(training_conf, rank_config, sft_config)
    init_rng(training_conf)

    data_manifest = Path(training_conf.data_split_manifest_path).resolve()
    schedule_metadata = validate_schedule(
        schedule_path,
        args.rollout_steps,
        args.prompts_per_rollout,
        expected_manifest=data_manifest,
        expected_base_seed=args.base_seed,
    )

    seeds = ExperimentSeeds.from_base(args.base_seed)
    trlx_config = TRLConfig.load_yaml(args.ppo_config)
    trlx_config.sft_config = sft_config
    trlx_config.train.seed = seeds.training
    trlx_config.tokenizer.tokenizer_path = sft_config.model_name
    trlx_config.model.model_path = sft_config.model_name
    trlx_config.method.ppo_epochs = 4
    trlx_config.method.gen_kwargs.update(
        {"max_new_tokens": 128, "top_k": 0, "top_p": 1.0, "do_sample": True, "temperature": 1.0}
    )
    trlx_config.method.init_kl_coef = args.kl_beta
    trlx_config.method.target = None

    if args.method == "ppo":
        trlx_config.train.trainer = "ExperimentAcceleratePPOTrainer"
        trlx_config.method.num_rollouts = 2 * args.prompts_per_rollout
        trlx_config.train.batch_size = 2 * args.pair_batch_size
        reward_fn = get_reward_fn(rank_config, training_conf)
        pair_config = None
        v2_config = None
        advpo_config = None
    elif args.method == "advpo":
        trlx_config.train.trainer = "ExperimentAcceleratePPOTrainer"
        trlx_config.method.num_rollouts = 2 * args.prompts_per_rollout
        trlx_config.train.batch_size = 2 * args.pair_batch_size
        pair_config = None
        v2_config = None
        advpo_config = AdvPOConfig(
            confidence_radius_squared=args.advpo_B,
            kl_beta=args.kl_beta,
            adversarial_batch_responses=args.pair_chunk_size,
        )
        reward_fn = AdvPORewardCallback(
            proxy_rm_path=rank_config.model_names[0],
            reference_policy_path=sft_config.model_name,
            reference_cache_path=args.reference_cache,
            confidence_path=args.advpo_confidence,
            config=advpo_config,
            device=torch.device("cuda", torch.cuda.device_count() - 1),
            batch_size=args.proxy_batch_size,
            data_manifest_path=str(data_manifest),
            prompt_schedule_path=str(schedule_path),
            allow_smoke_artifacts=args.allow_smoke_artifacts,
        )
    elif args.method == "cpdpo_v2":
        trlx_config.train.trainer = "ExperimentAcceleratePPOTrainer"
        trlx_config.method.num_rollouts = 2 * args.prompts_per_rollout
        trlx_config.train.batch_size = 2 * args.pair_batch_size
        pair_config = None
        v2_config = CPDPOV2Config(alpha=args.alpha, kl_beta=args.kl_beta)
        reward_fn = ReferenceAnchoredRewardCallback(
            proxy_rm_path=rank_config.model_names[0],
            reference_policy_path=sft_config.model_name,
            reference_cache_path=args.reference_cache,
            config=v2_config,
            device=torch.device("cuda", torch.cuda.device_count() - 1),
            batch_size=args.proxy_batch_size,
            geometry_path=args.geometry,
            calibration_path=args.calibration,
            data_manifest_path=str(data_manifest),
            prompt_schedule_path=str(schedule_path),
            allow_smoke_artifacts=args.allow_smoke_artifacts,
        )
        advpo_config = None
    else:
        trlx_config.train.trainer = "CustomAcceleratePairPPOTrainer"
        trlx_config.method.num_rollouts = args.prompts_per_rollout
        trlx_config.train.batch_size = args.pair_batch_size
        pair_config = CPDPOConfig(
            method=args.method,
            alpha=args.alpha,
            kl_beta=args.kl_beta,
            reward_variant=args.reward_variant,
            geometry_mode=args.geometry_mode,
        )
        reward_fn = PairRewardCallback(
            proxy_rm_path=rank_config.model_names[0],
            config=pair_config,
            device=torch.device("cuda", torch.cuda.device_count() - 1),
            batch_size=args.proxy_batch_size,
            geometry_path=args.geometry,
            calibration_path=args.calibration,
            data_manifest_path=str(data_manifest),
            allow_smoke_artifacts=args.allow_smoke_artifacts,
        )
        v2_config = None
        advpo_config = None
    trlx_config.method.chunk_size = args.pair_chunk_size
    if trlx_config.method.chunk_size % 2:
        raise ValueError("pair-chunk-size must be even")
    if (2 * args.prompts_per_rollout) % trlx_config.method.chunk_size:
        raise ValueError("2 * prompts-per-rollout must be divisible by pair-chunk-size")
    if trlx_config.method.num_rollouts % trlx_config.train.batch_size:
        raise ValueError("method rollout count must be divisible by its batch size")

    updates_per_rollout = (
        trlx_config.method.num_rollouts // trlx_config.train.batch_size
    ) * trlx_config.method.ppo_epochs
    total_optimizer_steps = args.rollout_steps * updates_per_rollout
    checkpoint_interval = args.checkpoint_every_rollouts * updates_per_rollout
    trlx_config.train.epochs = args.rollout_steps
    trlx_config.train.total_steps = total_optimizer_steps
    trlx_config.train.checkpoint_interval = checkpoint_interval
    trlx_config.train.eval_interval = checkpoint_interval
    trlx_config.scheduler.kwargs["T_max"] = total_optimizer_steps

    output_dir = Path(args.output_root).resolve() / f"seed_{args.base_seed}" / run_variant
    resume_checkpoint = Path(args.resume_from_checkpoint).resolve() if args.resume_from_checkpoint else None
    if resume_checkpoint is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    if resume_checkpoint is not None and not (output_dir / "run_metadata.json").is_file():
        raise FileNotFoundError("A resumed run requires its original run_metadata.json")
    (output_dir / "eval").mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is not None:
        try:
            resume_checkpoint.relative_to(checkpoint_dir.resolve())
        except ValueError as exc:
            raise ValueError("Resume checkpoint must belong to this method/seed run") from exc
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(resume_checkpoint)
        checkpoint_candidates = []
        for path in checkpoint_dir.glob("checkpoint_*"):
            try:
                checkpoint_candidates.append((int(path.name.rsplit("_", 1)[1]), path.resolve()))
            except ValueError:
                continue
        if not checkpoint_candidates or resume_checkpoint != max(checkpoint_candidates)[1]:
            raise ValueError("Resume is allowed only from the latest checkpoint in this run")
    trlx_config.train.output_dir = str(output_dir)
    trlx_config.train.checkpoint_dir = str(checkpoint_dir)
    trlx_config.train.run_name = f"reward_overoptimization/seed_{args.base_seed}/{run_variant}"

    eos_token = transformers.AutoTokenizer.from_pretrained(
        sft_config.model_name, cache_dir=sft_config.cache_dir
    ).eos_token
    duplicated = load_schedule_as_duplicated_prompts(schedule_path, rollout_steps=args.rollout_steps)
    for item in duplicated:
        item["prompt"] = "".join(format_pairs([item["prompt"]], eos_token, add_initial_reply_token=True))
    _train, eval_dict = get_manifest_dataset(training_conf, mode="rl")
    eval_dataset = eval_dict[next(iter(eval_dict))]
    eval_prompts = [
        "".join(format_pairs(eval_dataset[index], eos_token, add_initial_reply_token=True))
        for index in range(len(eval_dataset))
    ]
    if training_conf.num_eval_prompts:
        eval_prompts = eval_prompts[: training_conf.num_eval_prompts]

    initial_policy_fingerprint = model_fingerprint(sft_config.model_name)
    proxy_rm_fingerprint = model_fingerprint(rank_config.model_names[0])
    pair_artifacts = reward_fn.provenance() if isinstance(reward_fn, PairRewardCallback) else None
    reference_anchor = (
        reward_fn.provenance() if isinstance(reward_fn, ReferenceAnchoredRewardCallback) else None
    )
    advpo_artifacts = reward_fn.provenance() if isinstance(reward_fn, AdvPORewardCallback) else None
    code_revision = git_revision(ROOT)
    experiment_context = {
        "schema_version": "1.0.0",
        "method": args.method,
        "run_variant": run_variant,
        "experiment_track": experiment_track,
        "cpdpo_alpha": args.alpha if args.method in cpdpo_methods else None,
        "advpo_B": args.advpo_B if args.method == "advpo" else None,
        "base_seed": args.base_seed,
        "rollout_steps": args.rollout_steps,
        "responses_per_rollout": 2 * args.prompts_per_rollout,
        "prompt_chunk_size": int(trlx_config.method.chunk_size),
        "updates_per_rollout": updates_per_rollout,
        "prompt_schedule_sha256": sha256_file(schedule_path),
        "prompt_id_sequence_sha256": schedule_metadata["prompt_id_sequence_sha256"],
        "data_manifest_sha256": sha256_file(data_manifest),
        "initial_policy_fingerprint": initial_policy_fingerprint,
        "reference_policy_fingerprint": initial_policy_fingerprint,
        "proxy_rm_fingerprint": proxy_rm_fingerprint,
        "pair_artifacts": pair_artifacts,
        "code_revision": code_revision,
    }
    if reference_anchor is not None:
        experiment_context["reference_anchor"] = reference_anchor
    if advpo_artifacts is not None:
        experiment_context["advpo"] = advpo_artifacts
    trlx_config.train.trainer_kwargs = {
        "experiment_seeds": asdict(seeds),
        "experiment_context": experiment_context,
        "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "max_grad_norm": 1.0,
    }
    if pair_config is not None:
        trlx_config.train.trainer_kwargs["pair_method_config"] = pair_config.to_dict()

    metadata_trlx_config = copy.deepcopy(trlx_config.to_dict())
    metadata_trlx_config["train"]["trainer_kwargs"]["resume_from_checkpoint"] = None
    run_metadata = {
        "schema_version": "1.0.0",
        "experiment": "reward_overoptimization",
        "method": args.method,
        "run_variant": run_variant,
        "experiment_track": experiment_track,
        "cpdpo_alpha": args.alpha if args.method in cpdpo_methods else None,
        "advpo_B": args.advpo_B if args.method == "advpo" else None,
        "base_seed": args.base_seed,
        "resolved_seeds": asdict(seeds),
        "rollout_steps": args.rollout_steps,
        "prompts_per_rollout": args.prompts_per_rollout,
        "responses_per_prompt": 2,
        "generated_responses_budget": 2 * args.prompts_per_rollout * args.rollout_steps,
        "updates_per_rollout": updates_per_rollout,
        "total_optimizer_steps": total_optimizer_steps,
        "checkpoint_every_rollouts": args.checkpoint_every_rollouts,
        "checkpoint_interval_optimizer_steps": checkpoint_interval,
        "prompt_schedule": str(schedule_path),
        "prompt_schedule_sha256": sha256_file(schedule_path),
        "prompt_id_sequence_sha256": schedule_metadata["prompt_id_sequence_sha256"],
        "data_manifest_path": str(data_manifest),
        "data_manifest_sha256": sha256_file(data_manifest),
        "initial_policy_path": str(Path(sft_config.model_name).resolve()),
        "reference_policy_path": str(Path(sft_config.model_name).resolve()),
        "proxy_rm_path": str(Path(rank_config.model_names[0]).resolve()),
        "initial_policy_fingerprint": initial_policy_fingerprint,
        "reference_policy_fingerprint": initial_policy_fingerprint,
        "proxy_rm_fingerprint": proxy_rm_fingerprint,
        "code_revision": code_revision,
        "gold_training_access": False,
        "smoke_artifacts_allowed": args.allow_smoke_artifacts,
        "pair_method": pair_config.to_dict() if pair_config else None,
        "pair_artifacts": pair_artifacts,
        "trlx_config": metadata_trlx_config,
    }
    if v2_config is not None:
        run_metadata["cpdpo_v2"] = v2_config.to_dict()
        run_metadata["reference_anchor"] = reference_anchor
    if advpo_config is not None:
        run_metadata["advpo_config"] = advpo_config.to_dict()
        run_metadata["advpo"] = advpo_artifacts
    if resume_checkpoint is None:
        atomic_write_json(output_dir / "run_metadata.json", run_metadata)
    else:
        prior_metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if prior_metadata != run_metadata:
            raise ValueError("Resume arguments or immutable artifacts differ from the original run metadata")
    trainer = trlx.train(
        sft_config.model_name,
        reward_fn=reward_fn,
        prompts=duplicated,
        eval_prompts=eval_prompts,
        config=trlx_config,
        stop_sequences=[eos_token],
    )
    trainer.save_pretrained(str(output_dir / "model"))
    (output_dir / "rm_model_names.txt").write_text("\n".join(rank_config.model_names) + "\n", encoding="utf-8")
    print(f"PASS {args.method} training: {output_dir}")


if __name__ == "__main__":
    main()
