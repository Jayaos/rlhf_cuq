"""Named, validated initial/reference policy variants for the experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_POLICY_VARIANT = "1p4b"


@dataclass(frozen=True)
class PolicyVariantSpec:
    name: str
    default_path: str
    hidden_size: int
    num_hidden_layers: int
    output_tag: str

    def to_dict(self) -> dict:
        return asdict(self)


POLICY_VARIANTS = {
    "1p4b": PolicyVariantSpec(
        name="1p4b",
        default_path="assets/initial_sft_policy",
        hidden_size=2048,
        num_hidden_layers=24,
        output_tag="policy_1p4b",
    ),
    "70m": PolicyVariantSpec(
        name="70m",
        default_path="assets/proxy_rm_sft_base",
        hidden_size=512,
        num_hidden_layers=6,
        output_tag="policy_70m",
    ),
}


def get_policy_variant(name: str) -> PolicyVariantSpec:
    try:
        return POLICY_VARIANTS[name]
    except KeyError as exc:
        supported = ", ".join(POLICY_VARIANTS)
        raise ValueError(f"Unknown policy variant {name!r}; expected one of: {supported}") from exc


def validate_policy_checkpoint(path: str | Path, variant: str) -> dict:
    """Validate that ``path`` is the declared full GPT-NeoX causal LM.

    This intentionally rejects the trained scalar-head proxy RM, even though
    that reward model originates from the same 70M SFT asset.
    """

    spec = get_policy_variant(variant)
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory does not exist: {model_path}")
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Policy checkpoint is missing config.json: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Policy config is not valid JSON: {config_path}: {exc}") from exc

    if config.get("model_type") != "gpt_neox":
        raise ValueError(
            f"Policy variant {variant} requires model_type='gpt_neox'; "
            f"{config_path} declares {config.get('model_type')!r}. "
            "A scalar reward-model checkpoint cannot be used as a policy."
        )
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "GPTNeoXForCausalLM" not in architectures:
        raise ValueError(
            f"Policy variant {variant} requires architectures=['GPTNeoXForCausalLM']; "
            f"{config_path} declares {architectures!r}"
        )
    actual_hidden = config.get("hidden_size")
    actual_layers = config.get("num_hidden_layers")
    if actual_hidden != spec.hidden_size or actual_layers != spec.num_hidden_layers:
        raise ValueError(
            f"Policy variant {variant} expects hidden_size={spec.hidden_size} and "
            f"num_hidden_layers={spec.num_hidden_layers}; {config_path} declares "
            f"hidden_size={actual_hidden!r}, num_hidden_layers={actual_layers!r}"
        )
    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 2:
        raise ValueError(f"Policy checkpoint has an invalid vocab_size in {config_path}: {vocab_size!r}")

    return {
        "variant": spec.name,
        "architecture": "GPTNeoXForCausalLM",
        "model_type": "gpt_neox",
        "hidden_size": actual_hidden,
        "num_hidden_layers": actual_layers,
        "vocab_size": vocab_size,
    }
