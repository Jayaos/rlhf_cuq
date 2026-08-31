"""Batch-shared AdvPO reward callback for the existing scalar PPO trainer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from src.advpo.geometry import AdvPOConfidenceGeometry, advpo_batch_signal
from src.advpo.reference import AdvPOReferenceCache
from src.advpo.spec import AdvPOConfig
from src.cpdpo.artifacts import model_fingerprint, sha256_file, tokenizer_fingerprint
from src.cpdpo.reward_features import load_proxy_feature_scorer


class AdvPORewardCallback:
    """Apply one paper Eq. (7) adversarial reward head per PPO scoring batch."""

    def __init__(
        self,
        *,
        proxy_rm_path: str,
        reference_policy_path: str,
        reference_cache_path: str,
        confidence_path: str,
        config: AdvPOConfig,
        device: str | torch.device,
        batch_size: int,
        data_manifest_path: str,
        prompt_schedule_path: str,
        allow_smoke_artifacts: bool = False,
    ) -> None:
        self.config = config
        self.proxy_rm_path = str(Path(proxy_rm_path).resolve())
        self.proxy_rm_fingerprint = model_fingerprint(self.proxy_rm_path)
        self.tokenizer_fingerprint = tokenizer_fingerprint(self.proxy_rm_path)
        self.scorer = load_proxy_feature_scorer(
            self.proxy_rm_path, device=device, batch_size=batch_size
        )
        confidence_file = Path(confidence_path).resolve()
        metadata_file = confidence_file.with_name("confidence_matrix_metadata.json")
        if not metadata_file.is_file():
            raise FileNotFoundError(f"AdvPO confidence metadata is missing: {metadata_file}")
        self.confidence_path = str(confidence_file)
        self.confidence_fingerprint = sha256_file(confidence_file)
        self.confidence_metadata_path = str(metadata_file.resolve())
        self.confidence_metadata_fingerprint = sha256_file(metadata_file)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        self._validate_confidence_metadata(
            metadata,
            data_manifest_path=Path(data_manifest_path).resolve(),
            allow_smoke_artifacts=allow_smoke_artifacts,
        )
        self.geometry = AdvPOConfidenceGeometry.load(confidence_file)
        if self.geometry.dimension != int(metadata["dimension"]):
            raise ValueError("AdvPO confidence state/metadata feature dimensions differ")
        if self.geometry.n_responses != int(metadata["n_responses"]):
            raise ValueError("AdvPO confidence state/metadata response counts differ")
        if self.geometry.ridge_lambda != float(metadata["ridge_lambda"]):
            raise ValueError("AdvPO confidence state/metadata ridge values differ")
        self.artifact_scope = metadata["artifact_scope"]
        self.reference = AdvPOReferenceCache(
            reference_cache_path,
            reference_policy_path=reference_policy_path,
            proxy_rm_path=proxy_rm_path,
            data_manifest_path=data_manifest_path,
            prompt_schedule_path=prompt_schedule_path,
            allow_smoke_artifacts=allow_smoke_artifacts,
        )
        if self.reference.artifact_scope != self.artifact_scope:
            raise ValueError("AdvPO confidence and reference artifact scopes differ")
        if self.reference.features.shape[1] != self.geometry.dimension:
            raise ValueError("AdvPO reference feature dimension does not match confidence matrix")
        self._running_original_sum = 0.0
        self._running_adversarial_sum = 0.0
        self._running_count = 0
        self._rollout_batches: list[dict[str, torch.Tensor | float]] = []

    def _validate_confidence_metadata(
        self,
        value: dict[str, Any],
        *,
        data_manifest_path: Path,
        allow_smoke_artifacts: bool,
    ) -> None:
        if value.get("schema_version") != "1.0.0" or value.get("method") != "advpo":
            raise ValueError("Invalid AdvPO confidence metadata identity")
        if value.get("source_role") != "D_rm_train":
            raise ValueError("AdvPO confidence matrix must use D_rm_train")
        if value.get("normalization") != "unnormalized_sum":
            raise ValueError("AdvPO confidence matrix must use the paper's unnormalised sum")
        if value.get("feature_terms_per_preference") != 2:
            raise ValueError("AdvPO confidence matrix must include both preference responses")
        if value.get("confidence_fingerprint") != self.confidence_fingerprint:
            raise ValueError("AdvPO confidence matrix fingerprint mismatch")
        if value.get("proxy_rm_fingerprint") != self.proxy_rm_fingerprint:
            raise ValueError("AdvPO confidence matrix used a different proxy RM")
        if value.get("tokenizer_fingerprint") != self.tokenizer_fingerprint:
            raise ValueError("AdvPO confidence matrix used a different proxy tokenizer")
        if value.get("data_manifest_sha256") != sha256_file(data_manifest_path):
            raise ValueError("AdvPO confidence matrix used a different data manifest")
        scope = value.get("artifact_scope")
        if scope not in {"scientific", "smoke"}:
            raise ValueError(f"Unsupported AdvPO confidence scope: {scope}")
        if scope == "smoke" and not allow_smoke_artifacts:
            raise ValueError("Smoke AdvPO confidence cannot be used for scientific training")
        if value.get("gold_access") is not False:
            raise ValueError("AdvPO confidence metadata does not prove gold isolation")

    def start_rollout(self) -> None:
        self._rollout_batches = []

    def state_dict(self) -> dict[str, float | int]:
        """Persist the paper's running-mean reward scaler at rollout boundaries."""

        return {
            "schema_version": "1.0.0",
            "running_original_sum": self._running_original_sum,
            "running_adversarial_sum": self._running_adversarial_sum,
            "running_count": self._running_count,
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if value.get("schema_version") != "1.0.0":
            raise ValueError("Unsupported AdvPO running-scaler checkpoint schema")
        original = float(value["running_original_sum"])
        adversarial = float(value["running_adversarial_sum"])
        count = int(value["running_count"])
        if count < 1 or not math.isfinite(original) or not math.isfinite(adversarial):
            raise ValueError("Invalid AdvPO running-scaler checkpoint state")
        self._running_original_sum = original
        self._running_adversarial_sum = adversarial
        self._running_count = count
        self._rollout_batches = []

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
            raise ValueError("AdvPO training requires prompt_id metadata")
        prompt_ids = [str(value) for value in prompt_id]
        if len(prompt_ids) != len(prompts):
            raise ValueError("AdvPO prompt metadata cardinality mismatch")
        if len(prompt_ids) != self.config.adversarial_batch_responses:
            raise ValueError(
                "AdvPO callback batch differs from its frozen adversarial expectation scope: "
                f"expected {self.config.adversarial_batch_responses}, found {len(prompt_ids)}"
            )
        orientations = list(orientation) if orientation is not None else None
        if orientations is None or len(orientations) != len(prompt_ids):
            raise ValueError("AdvPO training requires complete response-orientation metadata")
        for index in range(0, len(prompt_ids), 2):
            if (
                index + 1 >= len(prompt_ids)
                or prompt_ids[index] != prompt_ids[index + 1]
                or orientations[index : index + 2] != ["a", "b"]
            ):
                raise ValueError("AdvPO requires adjacent a/b current responses for each prompt")
        reference_rewards, reference_features = self.reference.lookup(prompt_ids, list(prompts))
        signal = advpo_batch_signal(
            current_rewards=current_rewards.float(),
            reference_rewards=reference_rewards.float(),
            current_features=current_features.float(),
            reference_features=reference_features.float(),
            geometry=self.geometry,
            confidence_radius_squared=self.config.confidence_radius_squared,
            epsilon=self.config.epsilon,
        )
        unscaled = signal["current_adversarial_reward"].float().cpu()
        self._running_original_sum += float(current_rewards.double().sum().item())
        self._running_adversarial_sum += float(unscaled.double().sum().item())
        self._running_count += int(unscaled.numel())
        original_mean = self._running_original_sum / self._running_count
        adversarial_mean = self._running_adversarial_sum / self._running_count
        if abs(adversarial_mean) <= self.config.epsilon:
            raise RuntimeError("AdvPO dynamic scaling is undefined because its running adversarial mean is zero")
        scale = original_mean / adversarial_mean
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(
                "AdvPO dynamic scaling would be non-finite or reverse reward direction; "
                "the selected B is too pessimistic for this run"
            )
        scaled = unscaled * scale
        current_uncertainty = self.config.confidence_radius * self.geometry.quadratic_norm(
            current_features.float()
        ).float().cpu()
        reference_uncertainty = self.config.confidence_radius * self.geometry.quadratic_norm(
            reference_features.float()
        ).float().cpu()
        self._rollout_batches.append(
            {
                "current_reward": current_rewards.detach().float().cpu(),
                "reference_reward": reference_rewards.detach().float().cpu(),
                "adversarial_reward_unscaled": unscaled,
                "adversarial_reward_scaled": scaled,
                "current_penalty": signal["current_penalty"].float().cpu(),
                "reference_penalty": signal["reference_penalty"].float().cpu(),
                "current_uncertainty": current_uncertainty,
                "reference_uncertainty": reference_uncertainty,
                "batch_mahalanobis_mean_difference": float(
                    signal["mahalanobis_mean_difference"].item()
                ),
                "batch_lambda_star": float(signal["lambda_star"].item()),
                "batch_degenerate_direction": float(
                    signal["degenerate_direction"].float().item()
                ),
                "batch_robust_objective": float(signal["robust_objective"].item()),
                "dynamic_scale": float(scale),
            }
        )
        # The fixed-reference term determines the shared adversarial head via
        # g. Once that head is detached, the reference score is independent of
        # the trainable policy, so Eq. (7) supplies the adjusted current score
        # as the ordinary scalar PPO trajectory reward.
        return scaled, torch.zeros_like(scaled)

    def finish_rollout(self, *, expected_responses: int) -> dict[str, float | int]:
        if not self._rollout_batches:
            raise RuntimeError("AdvPO collected no training reward batches")
        vector_keys = (
            "current_reward",
            "reference_reward",
            "adversarial_reward_unscaled",
            "adversarial_reward_scaled",
            "current_penalty",
            "reference_penalty",
            "current_uncertainty",
            "reference_uncertainty",
        )
        combined = {
            key: torch.cat([batch[key] for batch in self._rollout_batches])
            for key in vector_keys
        }
        if combined["current_reward"].numel() != expected_responses:
            raise RuntimeError("AdvPO reward count does not match the rollout budget")
        metrics: dict[str, float | int] = {
            "advpo_B": self.config.confidence_radius_squared,
            "advpo_b": self.config.confidence_radius,
            "advpo_ridge_lambda": self.geometry.ridge_lambda,
            "advpo_reference_cache_unique_prompts": len(self.reference.prompt_ids),
            "advpo_reference_preparation_proxy_calls": len(self.reference.prompt_ids),
        }
        for key, values in combined.items():
            values = values.float()
            metrics[f"advpo_{key}_mean"] = float(values.mean().item())
            metrics[f"advpo_{key}_std"] = float(values.std(unbiased=False).item())
            for label, quantile in (("q10", 0.10), ("q50", 0.50), ("q90", 0.90)):
                metrics[f"advpo_{key}_{label}"] = float(torch.quantile(values, quantile).item())
        for key in (
            "batch_mahalanobis_mean_difference",
            "batch_lambda_star",
            "batch_degenerate_direction",
            "batch_robust_objective",
            "dynamic_scale",
        ):
            values = [float(batch[key]) for batch in self._rollout_batches]
            metrics[f"advpo_{key}_mean"] = sum(values) / len(values)
            metrics[f"advpo_{key}_min"] = min(values)
            metrics[f"advpo_{key}_max"] = max(values)
        if not all(
            isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
            for value in metrics.values()
        ):
            raise RuntimeError("AdvPO produced non-finite rollout metrics")
        return metrics

    def provenance(self) -> dict[str, Any]:
        return {
            "method": "advpo",
            "paper": "Zhang_et_al_NeurIPS_2024_arXiv_2403.05171v2",
            "proxy_rm_path": self.proxy_rm_path,
            "proxy_rm_fingerprint": self.proxy_rm_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "confidence_path": self.confidence_path,
            "confidence_fingerprint": self.confidence_fingerprint,
            "confidence_metadata_path": self.confidence_metadata_path,
            "confidence_metadata_fingerprint": self.confidence_metadata_fingerprint,
            "ridge_lambda": self.geometry.ridge_lambda,
            "confidence_response_count": self.geometry.n_responses,
            "confidence_dimension": self.geometry.dimension,
            "reference_cache_path": self.reference.path,
            "reference_cache_fingerprint": self.reference.fingerprint,
            "reference_cache_metadata_path": self.reference.metadata_path,
            "reference_cache_metadata_fingerprint": self.reference.metadata_fingerprint,
            "reference_responses_path": self.reference.responses_path,
            "reference_responses_fingerprint": self.reference.responses_fingerprint,
            "reference_policy_fingerprint": self.reference.reference_policy_fingerprint,
            "reference_generation_seed": self.reference.generation_seed,
            "reference_type": "fixed_sft_generation_per_prompt",
            "confidence_radius_squared_B": self.config.confidence_radius_squared,
            "confidence_radius_b": self.config.confidence_radius,
            "adversarial_direction_scope": "ppo_scoring_batch",
            "adversarial_batch_responses": self.config.adversarial_batch_responses,
            "dynamic_reward_scaling": "running_original_mean_over_running_adversarial_mean",
            "artifact_scope": self.artifact_scope,
            "gold_access": False,
        }
