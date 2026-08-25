"""Adapt verified logical split artifacts to the legacy Coste/OA trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data_utils.split_manifest import (
    SplitManifestError,
    load_split_records,
    verify_split_manifest,
)


def _manifest_path(conf: Any) -> Path:
    value = getattr(conf, "data_split_manifest_path", "")
    if not isinstance(value, str) or not value.strip():
        raise SplitManifestError("data_split_manifest_path must select a split manifest")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _limited_eval(dataset: Any, eval_size: int | None) -> Any:
    if eval_size is None or eval_size == 0:
        return dataset
    if eval_size < 0:
        raise SplitManifestError("eval_size cannot be negative")
    from torch.utils.data import Subset

    return Subset(dataset, list(range(min(len(dataset), eval_size))))


def _preference_dataset(conf: Any, manifest_path: Path):
    from torch.utils.data import ConcatDataset

    from src.data_utils.oa_custom_datasets.rank_datasets import CustomHFPref

    train_rows = load_split_records(manifest_path, "D_rm_train", expected_kind="preference")
    validation_rows = load_split_records(manifest_path, "D_rm_val", expected_kind="preference")
    train = CustomHFPref(train_rows, len(train_rows))
    validation = CustomHFPref(validation_rows, len(validation_rows), train=False)
    return ConcatDataset([train]), {"D_rm_val": _limited_eval(validation, conf.eval_size)}


def _format_ppo_prompt(row: dict[str, Any]) -> list[str]:
    instruction = row["instruction"]
    input_text = row["input"]
    return [f"{instruction}\n{input_text}" if input_text else instruction]


def _prompt_dataset(conf: Any, manifest_path: Path):
    from torch.utils.data import ConcatDataset
    from model_training.custom_datasets.utils import _filter_by_words

    override = getattr(conf, "rl_dataset_path_override", "")
    if override:
        raise SplitManifestError(
            "rl_dataset_path_override and data_split_manifest_path are mutually exclusive"
        )

    train_rows = load_split_records(manifest_path, "D_rl_train_prompts", expected_kind="prompt")
    validation_rows = load_split_records(
        manifest_path, "D_rl_val_prompts", expected_kind="prompt"
    )

    # The builder applies the pinned Coste/Open-Assistant filter before assigning
    # IDs. Recheck it here so a changed runtime filter cannot silently alter the
    # frozen split after the manifest was built.
    for role, rows in (
        ("D_rl_train_prompts", train_rows),
        ("D_rl_val_prompts", validation_rows),
    ):
        for row in rows:
            prompt = _format_ppo_prompt(row)[0]
            if _filter_by_words(prompt) is None or _filter_by_words(row["output"]) is None:
                raise SplitManifestError(
                    f"Runtime Open-Assistant filtering rejects a frozen record in {role}: "
                    f"{row['_split_record_id']}"
                )

    train = [_format_ppo_prompt(row) for row in train_rows]
    validation = [_format_ppo_prompt(row) for row in validation_rows]
    return ConcatDataset([train]), {
        "D_rl_val_prompts": _limited_eval(validation, conf.eval_size)
    }


def get_manifest_dataset(conf: Any, mode: str):
    """Return the legacy trainer contract from an immutable split manifest.

    Only training and validation roles are exposed. Calibration, test, and
    preserved external-validation roles remain inaccessible to online trainers.
    """

    manifest_path = _manifest_path(conf)
    verify_split_manifest(manifest_path)
    if mode == "rm":
        return _preference_dataset(conf, manifest_path)
    if mode == "rl":
        return _prompt_dataset(conf, manifest_path)
    raise SplitManifestError(f"Manifest-backed datasets do not support mode {mode!r}")
