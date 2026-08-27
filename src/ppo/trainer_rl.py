# Taken and modified from Open-Assistant's model/model_training/trainer_rl.py

import argparse
import sys
from argparse import Namespace
from pathlib import Path


# ``accelerate launch path/to/script.py`` exposes the script directory, not the
# repository root, on sys.path. Add the root before importing the ``src``
# namespace so the documented path-based cluster command works from any cwd.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transformers
import trlx
from trlx.data.configs import TRLConfig
from model_training.custom_datasets.formatting import (
    format_pairs,
)
from model_training.utils.utils import (
    _strtobool,
    init_rng,
    read_yamls,
)
from src.data_utils.manifest_dataset_loader import get_manifest_dataset
from src.data_utils.oa_custom_datasets.get_dataset_patch import get_dataset
from src.ppo.custom_helpers import gold_score, get_reward_fn, process_configs
from src.ppo.runtime_config import apply_local_asset_overrides, resolve_ppo_config_path
from src.ppo.custom_trlx_trainers.custom_accelerate_ppo_trainer import (
    CustomAcceleratePPOTrainer,  # noqa: F401
)


def argument_parsing(notebook=False, notebook_args=None, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument("--rm_seed", type=int, help="RM seed", default=None)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("./configs")
    for name in args.configs:
        if "," in name:
            for n in name.split(","):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    conf["local_rank"] = args.local_rank
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["rm_seed"] = args.rm_seed

    # Override config from command-line
    parser = argparse.ArgumentParser()

    for key, value in kwargs.items():
        type_ = type(value) if value is not None else str
        parser.add_argument(f"--{key}", type=type_, default=value)

    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)

    return parser.parse_args(remaining)


def main():
    training_conf = argument_parsing()
    rank_config = Namespace(**training_conf.rank_config)
    sft_config = Namespace(**training_conf.sft_config)
    gold_config = Namespace(**training_conf.gold_config)
    apply_local_asset_overrides(training_conf, rank_config, sft_config)

    init_rng(training_conf)

    eos_token = transformers.AutoTokenizer.from_pretrained(
        sft_config.model_name, cache_dir=sft_config.cache_dir
    ).eos_token

    # Load pretrained SFT model

    # override model_name to be the same as sft_model
    ppo_config_path = resolve_ppo_config_path(training_conf.ppo_config_path)
    trlx_config = TRLConfig.load_yaml(str(ppo_config_path))
    trlx_config.sft_config = sft_config

    if getattr(training_conf, "data_split_manifest_path", ""):
        train, eval_dict = get_manifest_dataset(training_conf, mode="rl")
    else:
        train, eval_dict = get_dataset(training_conf, mode="rl")
    print(train, eval_dict)

    # take the dataset as the eval prompt generation dataset
    eval = eval_dict[next(iter(eval_dict))]

    # trlx requires training data to be a list of prompts
    # first element of each sample is the context and the prompt
    prompts, eval_prompts = tuple(
        map(
            lambda x: ["".join(format_pairs(x[i], eos_token, add_initial_reply_token=True)) for i in range(len(x))],
            (train, eval),
        )
    )

    if training_conf.num_eval_prompts is not None and training_conf.num_eval_prompts > 0:
        eval_prompts = eval_prompts[: training_conf.num_eval_prompts]

    # Sanity Check for prompts to make sure it's loading properly
    with open(r"output.txt", "w") as fp:
        for item in eval_prompts:
            # write each item on a new line
            fp.write("Prompt For RL: %s\n" % item)

    trlx_config.tokenizer.tokenizer_path = sft_config.model_name
    trlx_config.model.model_path = sft_config.model_name

    # Main changes ---------------------------------------------------------------------

    output_dir = process_configs(training_conf, rank_config, trlx_config)

    if training_conf.debug:
        print("Continuing in debug mode")
        prompts = prompts[:10]
        eval_prompts = eval_prompts[:10]
        trlx_config.method.num_rollouts = 1

    reward_fn = get_reward_fn(rank_config, training_conf)
    trainer = trlx.train(
        sft_config.model_name,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=trlx_config,
        stop_sequences=[eos_token],
    )

    trainer.save_pretrained(output_dir + "/model")

    # Save the list of model names used in /model as well
    with open(output_dir + "/rm_model_names.txt", "w") as f:
        f.write("\n".join(rank_config.model_names))

    # Gold scoring stays offline and remains enabled by default. The opt-in
    # smoke configuration disables it because the licensed gold checkpoint is
    # not required to validate the online PPO setup.
    if training_conf.run_gold_evaluation:
        gold_score(
            output_dir + "/eval",
            gold_config.model_name,
            gold_config.is_alpacafarm_rm,
            gold_config.batch_size,
        )


if __name__ == "__main__":
    main()
