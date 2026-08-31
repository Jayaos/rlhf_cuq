"""Immutable fixed-SFT reference cache used by the additive AdvPO branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.cpdpo.artifacts import canonical_json_hash, model_fingerprint, sha256_file, tokenizer_fingerprint


ADVPO_REFERENCE_SCHEMA = "1.0.0"
ADVPO_REFERENCE_GENERATION_SEED_OFFSET = 40_000
ADVPO_PROMPT_CANONICALIZATION = "policy_tokenizer_decode_skip_special_tokens_v1"
ADVPO_GENERATION_SETTINGS = {
    "do_sample": True,
    "top_k": 0,
    "top_p": 1.0,
    "temperature": 1.0,
    "max_new_tokens": 128,
}


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"AdvPO reference cache {name} contains non-finite values")


class AdvPOReferenceCache:
    """Load and fingerprint one fixed SFT response per scheduled prompt."""

    def __init__(
        self,
        path: str | Path,
        *,
        reference_policy_path: str | Path,
        proxy_rm_path: str | Path,
        data_manifest_path: str | Path,
        prompt_schedule_path: str | Path,
        allow_smoke_artifacts: bool = False,
    ) -> None:
        self.path = str(Path(path).resolve())
        cache_path = Path(self.path)
        metadata_path = cache_path.with_name("reference_cache_metadata.json")
        responses_path = cache_path.with_name("reference_responses.jsonl")
        schedule_path = Path(prompt_schedule_path).resolve()
        schedule_metadata_path = schedule_path.with_suffix(schedule_path.suffix + ".metadata.json")
        if not metadata_path.is_file() or not responses_path.is_file():
            raise FileNotFoundError("AdvPO reference cache metadata/response records are missing")
        if not schedule_metadata_path.is_file():
            raise FileNotFoundError(f"Prompt schedule metadata is missing: {schedule_metadata_path}")
        self.fingerprint = sha256_file(cache_path)
        self.metadata_path = str(metadata_path.resolve())
        self.metadata_fingerprint = sha256_file(metadata_path)
        self.responses_path = str(responses_path.resolve())
        self.responses_fingerprint = sha256_file(responses_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._validate_metadata(
            metadata,
            reference_policy_path=Path(reference_policy_path).resolve(),
            proxy_rm_path=Path(proxy_rm_path).resolve(),
            data_manifest_path=Path(data_manifest_path).resolve(),
            prompt_schedule_path=schedule_path,
            prompt_schedule_metadata_path=schedule_metadata_path,
            allow_smoke_artifacts=allow_smoke_artifacts,
        )
        state = torch.load(cache_path, map_location="cpu")
        self._validate_state(state, metadata)
        scheduled_prompt_ids = []
        seen_prompt_ids = set()
        with schedule_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                prompt_id = str(json.loads(line)["prompt_id"])
                if prompt_id not in seen_prompt_ids:
                    scheduled_prompt_ids.append(prompt_id)
                    seen_prompt_ids.add(prompt_id)
        if list(state["prompt_ids"]) != scheduled_prompt_ids:
            raise ValueError("AdvPO reference cache does not cover the exact prompt schedule")
        self.prompt_ids = list(state["prompt_ids"])
        self.prompts = list(state["prompts"])
        self.responses = list(state["responses"])
        self.rewards = state["reference_rewards"].detach().float().cpu()
        self.features = state["reference_features"].detach().float().cpu()
        self.index_by_prompt_id = {value: index for index, value in enumerate(self.prompt_ids)}
        self.artifact_scope = metadata["artifact_scope"]
        self.generation_seed = int(metadata["reference_generation_seed"])
        self.reference_policy_fingerprint = metadata["reference_policy_fingerprint"]

    def _validate_metadata(
        self,
        value: dict[str, Any],
        *,
        reference_policy_path: Path,
        proxy_rm_path: Path,
        data_manifest_path: Path,
        prompt_schedule_path: Path,
        prompt_schedule_metadata_path: Path,
        allow_smoke_artifacts: bool,
    ) -> None:
        if value.get("schema_version") != ADVPO_REFERENCE_SCHEMA:
            raise ValueError("Unsupported AdvPO reference cache schema")
        if value.get("method") != "advpo" or value.get("source_role") != "D_rl_train_prompts":
            raise ValueError("Invalid AdvPO reference cache identity")
        if value.get("prompt_canonicalization") != ADVPO_PROMPT_CANONICALIZATION:
            raise ValueError("Unsupported AdvPO reference prompt canonicalization")
        if value.get("generation_settings") != ADVPO_GENERATION_SETTINGS:
            raise ValueError("AdvPO reference generation settings mismatch")
        if value.get("reference_cache_fingerprint") != self.fingerprint:
            raise ValueError("AdvPO reference cache fingerprint mismatch")
        if value.get("reference_responses_fingerprint") != self.responses_fingerprint:
            raise ValueError("AdvPO reference response-record fingerprint mismatch")
        if value.get("reference_policy_fingerprint") != model_fingerprint(reference_policy_path):
            raise ValueError("AdvPO references were generated by a different SFT policy")
        if value.get("reference_policy_tokenizer_fingerprint") != tokenizer_fingerprint(reference_policy_path):
            raise ValueError("AdvPO references used a different SFT tokenizer")
        if value.get("proxy_rm_fingerprint") != model_fingerprint(proxy_rm_path):
            raise ValueError("AdvPO references were scored by a different proxy RM")
        if value.get("tokenizer_fingerprint") != tokenizer_fingerprint(proxy_rm_path):
            raise ValueError("AdvPO references used a different proxy tokenizer")
        if value.get("data_manifest_sha256") != sha256_file(data_manifest_path):
            raise ValueError("AdvPO reference manifest mismatch")
        if value.get("prompt_schedule_sha256") != sha256_file(prompt_schedule_path):
            raise ValueError("AdvPO reference prompt schedule mismatch")
        if value.get("prompt_schedule_metadata_sha256") != sha256_file(
            prompt_schedule_metadata_path
        ):
            raise ValueError("AdvPO reference prompt schedule metadata mismatch")
        schedule_metadata = json.loads(prompt_schedule_metadata_path.read_text(encoding="utf-8"))
        if value.get("base_seed") != schedule_metadata.get("base_seed"):
            raise ValueError("AdvPO reference base seed does not match the prompt schedule")
        expected_reference_seed = int(schedule_metadata["base_seed"]) + ADVPO_REFERENCE_GENERATION_SEED_OFFSET
        if value.get("reference_generation_seed") != expected_reference_seed:
            raise ValueError("AdvPO reference generation seed is not in its frozen namespace")
        scope = value.get("artifact_scope")
        if scope not in {"scientific", "smoke"}:
            raise ValueError(f"Unsupported AdvPO reference scope: {scope}")
        if scope == "smoke" and not allow_smoke_artifacts:
            raise ValueError("Smoke AdvPO references cannot be used for scientific training")
        if value.get("gold_access") is not False:
            raise ValueError("AdvPO reference metadata does not prove gold isolation")

    @staticmethod
    def _validate_state(state: Any, metadata: dict[str, Any]) -> None:
        required = {
            "schema_version",
            "prompt_ids",
            "prompts",
            "responses",
            "response_token_ids",
            "reference_rewards",
            "reference_features",
        }
        if not isinstance(state, dict) or required - set(state):
            raise ValueError("Malformed AdvPO reference cache")
        if state["schema_version"] != ADVPO_REFERENCE_SCHEMA:
            raise ValueError("AdvPO reference state schema mismatch")
        prompt_ids = state["prompt_ids"]
        prompts = state["prompts"]
        responses = state["responses"]
        token_ids = state["response_token_ids"]
        rewards = state["reference_rewards"]
        features = state["reference_features"]
        if not all(isinstance(values, list) for values in (prompt_ids, prompts, responses, token_ids)):
            raise ValueError("AdvPO reference cache arrays must be lists")
        count = len(prompt_ids)
        if count < 1 or len(set(prompt_ids)) != count or any(not isinstance(item, str) for item in prompt_ids):
            raise ValueError("AdvPO reference prompt IDs must be nonempty and unique")
        if len(prompts) != count or len(responses) != count or len(token_ids) != count:
            raise ValueError("AdvPO reference arrays have unequal lengths")
        if any(not isinstance(item, str) for item in prompts + responses):
            raise ValueError("AdvPO reference prompts/responses must be strings")
        if any(
            not isinstance(tokens, list)
            or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens)
            for tokens in token_ids
        ):
            raise ValueError("AdvPO reference token IDs are malformed")
        if not isinstance(rewards, torch.Tensor) or rewards.shape != (count,):
            raise ValueError("AdvPO reference rewards have an invalid shape")
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] != count:
            raise ValueError("AdvPO reference features have an invalid shape")
        _finite("rewards", rewards)
        _finite("features", features)
        if metadata.get("unique_prompt_count") != count:
            raise ValueError("AdvPO reference metadata count mismatch")
        if metadata.get("prompt_ids_sha256") != canonical_json_hash(prompt_ids):
            raise ValueError("AdvPO reference prompt-ID sequence mismatch")
        response_ids = [
            canonical_json_hash([prompt_id, tokens])
            for prompt_id, tokens in zip(prompt_ids, token_ids)
        ]
        if metadata.get("reference_response_ids_sha256") != canonical_json_hash(response_ids):
            raise ValueError("AdvPO reference response identity mismatch")

    def lookup(self, prompt_ids: list[str], prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            indices = [self.index_by_prompt_id[str(value)] for value in prompt_ids]
        except KeyError as exc:
            raise ValueError(f"AdvPO reference cache is missing prompt ID {exc.args[0]}") from exc
        expected_prompts = [self.prompts[index] for index in indices]
        if prompts != expected_prompts:
            mismatch = next(
                index
                for index, (observed, expected) in enumerate(zip(prompts, expected_prompts))
                if observed != expected
            )
            raise ValueError(
                "AdvPO tokenizer-canonical prompt mismatch for "
                f"prompt_id={prompt_ids[mismatch]}: observed={prompts[mismatch]!r}, "
                f"cached={expected_prompts[mismatch]!r}"
            )
        return self.rewards[indices], self.features[indices]
