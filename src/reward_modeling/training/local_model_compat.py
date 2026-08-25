"""Compatibility helpers for architecture-neutral local checkpoint paths.

The pinned Open-Assistant utilities infer both tokenizer behavior and the
reward-model class with substring checks against ``conf.model_name``.  A local
path such as ``assets/proxy_rm_sft_base`` therefore hides that the verified
checkpoint is Pythia/GPT-NeoX.  This module supplies an explicit, validated
family hint while leaving the path passed to ``from_pretrained`` unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_EXPECTED_MODEL_TYPES = {"pythia": frozenset({"gpt_neox"})}


class _ArchitectureAwareModelName(str):
    """A filesystem path that also answers the legacy family-name check."""

    def __new__(cls, value: str, family: str):
        instance = super().__new__(cls, value)
        instance.family = family
        return instance

    def __contains__(self, item: object) -> bool:
        return item == self.family or super().__contains__(item)

    def __getnewargs__(self) -> tuple[str, str]:
        return str(self), self.family


def apply_local_model_family(conf: Any, *, working_directory: Path | None = None) -> Any:
    """Validate and apply ``conf.model_family`` for a local model directory.

    Configurations without ``model_family`` retain the exact legacy behavior.
    The hint is intentionally limited to known family/model-type pairs.
    """

    family = getattr(conf, "model_family", "")
    if not family:
        return conf
    if family not in _EXPECTED_MODEL_TYPES:
        supported = ", ".join(sorted(_EXPECTED_MODEL_TYPES))
        raise ValueError(f"Unsupported model_family={family!r}; expected one of: {supported}")

    model_name = getattr(conf, "model_name", None)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_family requires a non-empty model_name")

    model_path = Path(model_name).expanduser()
    if not model_path.is_absolute():
        base = working_directory if working_directory is not None else Path.cwd()
        model_path = base / model_path
    config_path = model_path / "config.json"
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Local {family} checkpoint is missing its config: {config_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Local checkpoint config is not valid JSON: {config_path}: {exc}") from exc

    model_type = model_config.get("model_type")
    expected_types = _EXPECTED_MODEL_TYPES[family]
    if model_type not in expected_types:
        expected = ", ".join(sorted(expected_types))
        raise ValueError(
            f"model_family={family!r} expects model_type in {{{expected}}}, "
            f"but {config_path} declares {model_type!r}"
        )

    if family not in model_name:
        conf.model_name = _ArchitectureAwareModelName(model_name, family)
    return conf
