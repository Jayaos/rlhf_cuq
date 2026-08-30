"""Immutable SFT-reference artifacts and the exploratory CPDPOv2 reward."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from src.cpdpo.artifacts import canonical_json_hash, model_fingerprint, sha256_file, tokenizer_fingerprint
from src.cpdpo.pair_reward import PairRewardCallback
from src.cpdpo.spec import CPDPOConfig, CPDPOV2Config, is_main_alpha


REFERENCE_CACHE_SCHEMA = "1.1.0"
REFERENCE_GENERATION_SEED_OFFSET = 40_000
REFERENCE_PROMPT_CANONICALIZATION = "policy_tokenizer_decode_skip_special_tokens_v1"


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"Reference cache {name} contains non-finite values")


class ReferenceAnchoredRewardCallback:
    """Score current trajectories against one immutable SFT response per prompt."""

    def __init__(
        self,
        *,
        proxy_rm_path: str,
        reference_policy_path: str,
        reference_cache_path: str,
        config: CPDPOV2Config,
        device: str | torch.device,
        batch_size: int,
        geometry_path: str,
        calibration_path: str,
        data_manifest_path: str,
        prompt_schedule_path: str,
        allow_smoke_artifacts: bool = False,
    ):
        self.config = config
        # Reuse the fully audited v1 artifact loader. This does not reuse the
        # v1 pair loss or its current/current rollout signal.
        self.artifacts = PairRewardCallback(
            proxy_rm_path=proxy_rm_path,
            config=CPDPOConfig(
                method="cpdpo",
                alpha=config.alpha,
                epsilon=config.epsilon,
                kl_beta=config.kl_beta,
            ),
            device=device,
            batch_size=batch_size,
            geometry_path=geometry_path,
            calibration_path=calibration_path,
            data_manifest_path=data_manifest_path,
            allow_smoke_artifacts=allow_smoke_artifacts,
        )
        if self.artifacts.geometry is None:
            raise ValueError("CPDPOv2 requires a full pair geometry")
        self.scorer = self.artifacts.scorer
        self.proxy_rm_fingerprint = self.artifacts.proxy_rm_fingerprint
        self.tokenizer_fingerprint = self.artifacts.tokenizer_fingerprint
        self.geometry_fingerprint = self.artifacts.geometry_fingerprint
        self.calibration_fingerprint = self.artifacts.calibration_fingerprint
        self.q_alpha = self.artifacts.q_alpha
        calibration_scores_path = Path(calibration_path).with_name("calibration_scores.pt")
        calibration_state = torch.load(calibration_scores_path, map_location="cpu")
        cal_margins = calibration_state.get("margins")
        cal_uncertainties = calibration_state.get("uncertainties")
        if (
            not isinstance(cal_margins, torch.Tensor)
            or not isinstance(cal_uncertainties, torch.Tensor)
            or cal_margins.shape != cal_uncertainties.shape
            or cal_margins.numel() < 1
        ):
            raise ValueError("Malformed calibration distribution diagnostics")
        cal_margins = cal_margins.detach().float().reshape(-1)
        cal_uncertainties = cal_uncertainties.detach().float().reshape(-1)
        _finite_tensor("calibration margins", cal_margins)
        _finite_tensor("calibration uncertainties", cal_uncertainties)
        if torch.any(cal_uncertainties < 0):
            raise ValueError("Calibration uncertainties cannot be negative")
        cal_normalized = cal_margins.abs() / (cal_uncertainties + config.epsilon)
        self.calibration_distribution_summary = self._distribution_summary(
            {
                "absolute_margin": cal_margins.abs(),
                "uncertainty": cal_uncertainties,
                "normalized_margin": cal_normalized,
            }
        )

        self.reference_policy_path = str(Path(reference_policy_path).resolve())
        self.reference_policy_fingerprint = model_fingerprint(self.reference_policy_path)
        self.reference_policy_tokenizer_fingerprint = tokenizer_fingerprint(self.reference_policy_path)
        self.reference_cache_path = str(Path(reference_cache_path).resolve())
        self.reference_cache_fingerprint = sha256_file(self.reference_cache_path)
        metadata_path = Path(self.reference_cache_path).with_name("reference_cache_metadata.json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Reference cache metadata is missing: {metadata_path}")
        self.reference_cache_metadata_path = str(metadata_path.resolve())
        self.reference_cache_metadata_fingerprint = sha256_file(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        responses_path = Path(self.reference_cache_path).with_name("reference_responses.jsonl")
        if not responses_path.is_file():
            raise FileNotFoundError(f"Reference response records are missing: {responses_path}")
        self.reference_responses_path = str(responses_path.resolve())
        self.reference_responses_fingerprint = sha256_file(responses_path)
        self._validate_metadata(
            metadata,
            data_manifest_path=Path(data_manifest_path).resolve(),
            prompt_schedule_path=Path(prompt_schedule_path).resolve(),
            allow_smoke_artifacts=allow_smoke_artifacts,
        )

        state = torch.load(self.reference_cache_path, map_location="cpu")
        self._validate_state(state, metadata)
        self.prompt_ids = list(state["prompt_ids"])
        self.prompts = list(state["prompts"])
        self.responses = list(state["responses"])
        self.reference_rewards = state["reference_rewards"].detach().float().cpu()
        self.reference_features = state["reference_features"].detach().float().cpu()
        if self.reference_features.shape[1] != self.artifacts.geometry.dimension:
            raise ValueError("Reference cache feature dimension does not match pair geometry")
        self.index_by_prompt_id = {prompt_id: index for index, prompt_id in enumerate(self.prompt_ids)}
        self.artifact_scope = metadata["artifact_scope"]
        self.reference_generation_seed = int(metadata["reference_generation_seed"])
        self._rollout_batches: list[dict[str, torch.Tensor]] = []

    def _validate_metadata(
        self,
        value: dict[str, Any],
        *,
        data_manifest_path: Path,
        prompt_schedule_path: Path,
        allow_smoke_artifacts: bool,
    ) -> None:
        if value.get("schema_version") != REFERENCE_CACHE_SCHEMA:
            raise ValueError("Unsupported CPDPOv2 reference cache schema")
        if value.get("prompt_canonicalization") != REFERENCE_PROMPT_CANONICALIZATION:
            raise ValueError("Unsupported CPDPOv2 reference prompt canonicalization")
        if value.get("method") != "cpdpo_v2" or value.get("source_role") != "D_rl_train_prompts":
            raise ValueError("Invalid CPDPOv2 reference cache identity")
        if value.get("reference_cache_fingerprint") != self.reference_cache_fingerprint:
            raise ValueError("CPDPOv2 reference cache fingerprint mismatch")
        if value.get("reference_policy_fingerprint") != self.reference_policy_fingerprint:
            raise ValueError("Reference cache was generated by a different SFT policy")
        if value.get("reference_policy_tokenizer_fingerprint") != self.reference_policy_tokenizer_fingerprint:
            raise ValueError("Reference cache used a different SFT tokenizer")
        if value.get("reference_responses_fingerprint") != self.reference_responses_fingerprint:
            raise ValueError("Reference response record fingerprint mismatch")
        if value.get("proxy_rm_fingerprint") != self.proxy_rm_fingerprint:
            raise ValueError("Reference cache was scored by a different proxy RM")
        if value.get("tokenizer_fingerprint") != self.tokenizer_fingerprint:
            raise ValueError("Reference cache proxy tokenizer fingerprint mismatch")
        if value.get("geometry_fingerprint") != self.geometry_fingerprint:
            raise ValueError("Reference cache was prepared with a different geometry")
        if value.get("calibration_fingerprint") != self.calibration_fingerprint:
            raise ValueError("Reference cache was prepared with a different calibration")
        if value.get("data_manifest_sha256") != sha256_file(data_manifest_path):
            raise ValueError("Reference cache was built from a different data manifest")
        if value.get("prompt_schedule_sha256") != sha256_file(prompt_schedule_path):
            raise ValueError("Reference cache was built from a different prompt schedule")
        if value.get("alpha") != self.config.alpha or value.get("q_alpha") != self.q_alpha:
            raise ValueError("Reference cache alpha/q does not match CPDPOv2 artifacts")
        expected_track = "main" if is_main_alpha(self.config.alpha) else "cpdpo_v2_alpha_ablation"
        if value.get("experiment_track") != expected_track:
            raise ValueError("Reference cache experiment track does not match alpha")
        scope = value.get("artifact_scope")
        if scope not in {"scientific", "smoke"}:
            raise ValueError(f"Unsupported CPDPOv2 reference artifact scope: {scope}")
        if scope == "smoke" and not allow_smoke_artifacts:
            raise ValueError("Smoke reference cache cannot be used for scientific training")
        if scope != self.artifacts.artifact_scope:
            raise ValueError("Reference cache and geometry/calibration artifact scopes differ")
        settings = value.get("generation_settings")
        expected_settings = {
            "do_sample": True,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
            "max_new_tokens": 128,
        }
        if settings != expected_settings:
            raise ValueError("Reference cache generation settings differ from CPDPOv2")

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
            raise ValueError("Malformed CPDPOv2 reference cache")
        if state["schema_version"] != REFERENCE_CACHE_SCHEMA:
            raise ValueError("Reference cache state schema mismatch")
        prompt_ids = state["prompt_ids"]
        prompts = state["prompts"]
        responses = state["responses"]
        response_token_ids = state["response_token_ids"]
        rewards = state["reference_rewards"]
        features = state["reference_features"]
        n = len(prompt_ids)
        if n < 1 or len(set(prompt_ids)) != n or any(not isinstance(value, str) for value in prompt_ids):
            raise ValueError("Reference cache prompt IDs must be nonempty and unique")
        if (
            len(prompts) != n
            or len(responses) != n
            or len(response_token_ids) != n
            or any(not isinstance(value, str) for value in prompts)
            or any(not isinstance(value, str) for value in responses)
            or any(
                not isinstance(tokens, list)
                or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens)
                for tokens in response_token_ids
            )
        ):
            raise ValueError("Reference cache text arrays have unequal lengths")
        if not isinstance(rewards, torch.Tensor) or rewards.shape != (n,):
            raise ValueError("Reference cache reward tensor has an invalid shape")
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] != n:
            raise ValueError("Reference cache feature tensor has an invalid shape")
        _finite_tensor("rewards", rewards)
        _finite_tensor("features", features)
        if metadata.get("unique_prompt_count") != n:
            raise ValueError("Reference cache metadata count mismatch")
        if metadata.get("prompt_ids_sha256") != canonical_json_hash(prompt_ids):
            raise ValueError("Reference cache prompt-ID sequence mismatch")
        response_ids = [
            canonical_json_hash([prompt_id, token_ids])
            for prompt_id, token_ids in zip(prompt_ids, response_token_ids)
        ]
        if metadata.get("reference_response_ids_sha256") != canonical_json_hash(response_ids):
            raise ValueError("Reference response identity mismatch")

    def start_rollout(self) -> None:
        self._rollout_batches = []

    @staticmethod
    def _distribution_summary(values_by_name: dict[str, torch.Tensor]) -> dict[str, float]:
        summary = {}
        for name, values in values_by_name.items():
            values = values.detach().float().reshape(-1)
            summary[f"{name}_mean"] = float(values.mean().item())
            for label, quantile in (("q10", 0.10), ("q50", 0.50), ("q90", 0.90)):
                summary[f"{name}_{label}"] = float(torch.quantile(values, quantile).item())
        return summary

    def __call__(
        self,
        samples,
        prompts,
        outputs,
        eval=False,
        prompt_id=None,
        orientation=None,
        **_kwargs,
    ):
        current_rewards, current_features = self.scorer.score(
            list(prompts), list(outputs), evaluation=bool(eval)
        )
        if eval:
            return current_rewards, torch.zeros_like(current_rewards)
        if prompt_id is None:
            raise ValueError("CPDPOv2 training requires prompt_id metadata")
        prompt_ids = list(prompt_id)
        if len(prompt_ids) != len(prompts):
            raise ValueError("CPDPOv2 prompt metadata cardinality mismatch")
        orientations = list(orientation) if orientation is not None else None
        if orientations is None or len(orientations) != len(prompt_ids):
            raise ValueError("CPDPOv2 training requires complete response-orientation metadata")
        for index in range(0, len(prompt_ids), 2):
            if (
                index + 1 >= len(prompt_ids)
                or prompt_ids[index] != prompt_ids[index + 1]
                or orientations[index : index + 2] != ["a", "b"]
            ):
                raise ValueError("CPDPOv2 requires adjacent a/b current responses for each prompt")
        try:
            indices = [self.index_by_prompt_id[str(value)] for value in prompt_ids]
        except KeyError as exc:
            raise ValueError(f"CPDPOv2 reference cache is missing prompt ID {exc.args[0]}") from exc
        expected_prompts = [self.prompts[index] for index in indices]
        if list(prompts) != expected_prompts:
            mismatch = next(
                index
                for index, (observed, expected) in enumerate(zip(prompts, expected_prompts))
                if observed != expected
            )
            raise ValueError(
                "CPDPOv2 tokenizer-canonical prompt mismatch for "
                f"prompt_id={prompt_ids[mismatch]}: "
                f"observed={prompts[mismatch]!r}, cached={expected_prompts[mismatch]!r}"
            )
        reference_rewards = self.reference_rewards[indices]
        reference_features = self.reference_features[indices]
        differences = current_features.float() - reference_features
        uncertainty = self.artifacts.geometry.uncertainty(differences).float()
        margins = current_rewards.float() - reference_rewards
        normalized = margins.abs() / (uncertainty + self.config.epsilon)
        robust_rewards = (margins - self.q_alpha * uncertainty) * self.config.reward_scale
        for name, tensor in {
            "current rewards": current_rewards,
            "margins": margins,
            "uncertainties": uncertainty,
            "robust rewards": robust_rewards,
        }.items():
            _finite_tensor(name, tensor)
        self._rollout_batches.append(
            {
                "current_reward": current_rewards.detach().float().cpu(),
                "reference_reward": reference_rewards.detach().float().cpu(),
                "margin": margins.detach().cpu(),
                "uncertainty": uncertainty.detach().cpu(),
                "normalized_margin": normalized.detach().cpu(),
                "robust_reward": robust_rewards.detach().cpu(),
                "certified_current_better": (
                    (margins > 0.0) & (normalized > self.q_alpha)
                ).detach().float().cpu(),
            }
        )
        return robust_rewards, torch.zeros_like(robust_rewards)

    def finish_rollout(self, *, expected_responses: int) -> dict[str, float | int]:
        if not self._rollout_batches:
            raise RuntimeError("CPDPOv2 collected no training reward batches")
        combined = {
            key: torch.cat([batch[key] for batch in self._rollout_batches])
            for key in self._rollout_batches[0]
        }
        if combined["robust_reward"].numel() != expected_responses:
            raise RuntimeError("CPDPOv2 reward count does not match the rollout budget")
        metrics: dict[str, float | int] = {
            "reference_cache_unique_prompts": len(self.prompt_ids),
            "reference_preparation_proxy_calls": len(self.prompt_ids),
            "cpdpo_v2_q_alpha": self.q_alpha,
        }
        for key in (
            "current_reward",
            "reference_reward",
            "margin",
            "uncertainty",
            "normalized_margin",
            "robust_reward",
        ):
            values = combined[key].float()
            metrics[f"cpdpo_v2_{key}_mean"] = float(values.mean().item())
            metrics[f"cpdpo_v2_{key}_std"] = float(values.std(unbiased=False).item())
            for label, quantile in (("q10", 0.10), ("q50", 0.50), ("q90", 0.90)):
                metrics[f"cpdpo_v2_{key}_{label}"] = float(torch.quantile(values, quantile).item())
        metrics["cpdpo_v2_certified_current_better_rate"] = float(
            combined["certified_current_better"].mean().item()
        )
        metrics.update(
            {
                f"cpdpo_v2_calibration_{key}": value
                for key, value in self.calibration_distribution_summary.items()
            }
        )
        calibration_u50 = self.calibration_distribution_summary["uncertainty_q50"]
        rollout_u50 = metrics["cpdpo_v2_uncertainty_q50"]
        metrics["cpdpo_v2_uncertainty_median_ratio_to_calibration"] = (
            float(rollout_u50) / max(calibration_u50, self.config.epsilon)
        )
        if not all(
            isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
            for value in metrics.values()
        ):
            raise RuntimeError("CPDPOv2 produced non-finite rollout metrics")
        return metrics

    def provenance(self) -> dict[str, Any]:
        value = dict(self.artifacts.provenance())
        value.update(
            {
                "method": "cpdpo_v2",
                "reference_policy_path": self.reference_policy_path,
                "reference_policy_fingerprint": self.reference_policy_fingerprint,
                "reference_cache_path": self.reference_cache_path,
                "reference_cache_fingerprint": self.reference_cache_fingerprint,
                "reference_cache_metadata_path": self.reference_cache_metadata_path,
                "reference_cache_metadata_fingerprint": self.reference_cache_metadata_fingerprint,
                "reference_responses_path": self.reference_responses_path,
                "reference_responses_fingerprint": self.reference_responses_fingerprint,
                "reference_generation_seed": self.reference_generation_seed,
                "exchangeability_assumption": "D_cal pair differences exchangeable with current/SFT differences",
                "gold_reward_bound": False,
                "calibration_distribution_summary": self.calibration_distribution_summary,
            }
        )
        return value
