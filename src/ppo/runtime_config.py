"""Small, dependency-free helpers for selecting legacy PPO configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_PPO_CONFIG_PATH = Path("configs/ppo_config.yaml")


def resolve_ppo_config_path(configured_path: str | Path | None, *, working_directory: Path | None = None) -> Path:
    """Return an existing PPO config path without changing the legacy default.

    Relative paths are interpreted from ``working_directory`` (or the current
    process directory), matching the existing launch commands.
    """

    path = Path(configured_path) if configured_path else DEFAULT_PPO_CONFIG_PATH
    if not path.is_absolute():
        path = (working_directory or Path.cwd()) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PPO configuration does not exist: {path}")
    return path


def apply_local_asset_overrides(training_conf: Any, rank_config: Any, sft_config: Any) -> None:
    """Apply explicit local snapshots without changing legacy defaults.

    The override is intentionally limited to the audited single-RM AlpacaFarm
    baseline. It lets the smoke run operate with Hub access disabled after the
    caller downloads revision-pinned snapshots.
    """

    if training_conf.policy_model_path_override:
        sft_config.model_name = str(
            _resolve_local_directory(training_conf.policy_model_path_override, "policy model")
        )

    if training_conf.proxy_rm_path_override:
        if len(rank_config.model_names) != 1:
            raise ValueError("proxy_rm_path_override supports only the single-RM baseline")
        rank_config.model_names = [
            str(_resolve_local_directory(training_conf.proxy_rm_path_override, "proxy reward model"))
        ]

    if training_conf.rl_dataset_path_override:
        if training_conf.datasets != ["alpaca_farm"] or training_conf.datasets_extra:
            raise ValueError("rl_dataset_path_override supports only the single AlpacaFarm RL dataset")
        training_conf.datasets = [
            {
                "alpaca_farm": {
                    "dataset_path": str(
                        _resolve_local_directory(training_conf.rl_dataset_path_override, "RL dataset")
                    )
                }
            }
        ]


def _resolve_local_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local {label} directory does not exist: {path}")
    return path
