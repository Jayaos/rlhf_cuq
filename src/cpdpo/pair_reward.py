"""Frozen proxy-RM callbacks for PairPPO and CPDPO."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.cpdpo.artifacts import model_fingerprint, sha256_file, tokenizer_fingerprint
from src.cpdpo.geometry import PairGeometry, pair_signals
from src.cpdpo.reward_features import ProxyRewardFeatureScorer, load_proxy_feature_scorer
from src.cpdpo.spec import CPDPOConfig


class PairRewardCallback:
    """One frozen proxy model shared by rollout-pair scoring and scalar evaluation."""

    def __init__(
        self,
        *,
        proxy_rm_path: str,
        config: CPDPOConfig,
        device: str | torch.device,
        batch_size: int,
        geometry_path: str | None = None,
        calibration_path: str | None = None,
        data_manifest_path: str | None = None,
    ):
        self.config = config
        self.proxy_rm_path = str(Path(proxy_rm_path).resolve())
        self.scorer: ProxyRewardFeatureScorer = load_proxy_feature_scorer(
            self.proxy_rm_path, device=device, batch_size=batch_size
        )
        self.proxy_rm_fingerprint = model_fingerprint(self.proxy_rm_path)
        self.tokenizer_fingerprint = tokenizer_fingerprint(self.proxy_rm_path)
        self.geometry: PairGeometry | None = None
        self.geometry_fingerprint: str | None = None
        self.geometry_metadata_fingerprint: str | None = None
        self.calibration_fingerprint: str | None = None
        self.calibration_scores_fingerprint: str | None = None
        self.q_alpha = 0.0
        self.data_manifest_path = str(Path(data_manifest_path).resolve()) if data_manifest_path else None
        self.data_manifest_fingerprint = (
            sha256_file(self.data_manifest_path) if self.data_manifest_path is not None else None
        )

        if config.method == "cpdpo":
            if not geometry_path or not calibration_path:
                raise ValueError("CPDPO requires geometry_path and calibration_path")
            if self.data_manifest_path is None:
                raise ValueError("CPDPO requires the expected data manifest for artifact validation")
            geometry_state = torch.load(geometry_path, map_location="cpu")
            if geometry_state.get("geometry_mode") != config.geometry_mode:
                raise ValueError("Geometry artifact mode does not match the CPDPO configuration")
            if config.geometry_mode == "full":
                self.geometry = PairGeometry.from_state_dict(geometry_state)
            self.geometry_fingerprint = sha256_file(geometry_path)
            geometry_metadata_file = Path(geometry_path).with_name("pair_geometry_metadata.json")
            if not geometry_metadata_file.is_file():
                raise FileNotFoundError(f"Pair geometry metadata is missing: {geometry_metadata_file}")
            geometry_metadata = json.loads(geometry_metadata_file.read_text(encoding="utf-8"))
            self.geometry_metadata_fingerprint = sha256_file(geometry_metadata_file)
            self._validate_geometry_metadata(geometry_metadata)
            calibration_file = Path(calibration_path)
            calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
            self.calibration_fingerprint = sha256_file(calibration_file)
            self._validate_calibration(calibration)
            calibration_scores_file = calibration_file.with_name("calibration_scores.pt")
            if not calibration_scores_file.is_file():
                raise FileNotFoundError(f"Calibration scores are missing: {calibration_scores_file}")
            self.calibration_scores_fingerprint = sha256_file(calibration_scores_file)
            if calibration.get("calibration_scores_fingerprint") != self.calibration_scores_fingerprint:
                raise ValueError("Calibration score artifact fingerprint mismatch")
            self.q_alpha = float(calibration["q_alpha"])
        elif geometry_path or calibration_path:
            raise ValueError("PairPPO must not load CPDPO geometry/calibration artifacts")

    def _validate_calibration(self, value: dict) -> None:
        if value.get("schema_version") != "1.0.0":
            raise ValueError("Unsupported conformal calibration schema")
        if value.get("alpha") != self.config.alpha or value.get("epsilon") != self.config.epsilon:
            raise ValueError("Calibration alpha/epsilon does not match the training configuration")
        if value.get("proxy_rm_fingerprint") != self.proxy_rm_fingerprint:
            raise ValueError("Calibration artifact was built from a different proxy RM")
        if value.get("tokenizer_fingerprint") != self.tokenizer_fingerprint:
            raise ValueError("Calibration artifact was built from a different tokenizer")
        if value.get("geometry_fingerprint") != self.geometry_fingerprint:
            raise ValueError("Calibration artifact does not match pair_geometry.pt")
        if value.get("geometry_mode") != self.config.geometry_mode:
            raise ValueError("Calibration geometry mode does not match the training configuration")
        if value.get("data_manifest_sha256") != self.data_manifest_fingerprint:
            raise ValueError("Calibration artifact was built from a different data manifest")

    def _validate_geometry_metadata(self, value: dict) -> None:
        if value.get("schema_version") != "1.0.0" or value.get("source_role") != "D_rm_train":
            raise ValueError("Invalid pair geometry metadata schema or source role")
        if value.get("geometry_fingerprint") != self.geometry_fingerprint:
            raise ValueError("Pair geometry metadata fingerprint mismatch")
        if value.get("proxy_rm_fingerprint") != self.proxy_rm_fingerprint:
            raise ValueError("Pair geometry was built from a different proxy RM")
        if value.get("tokenizer_fingerprint") != self.tokenizer_fingerprint:
            raise ValueError("Pair geometry was built from a different tokenizer")
        if value.get("data_manifest_sha256") != self.data_manifest_fingerprint:
            raise ValueError("Pair geometry was built from a different data manifest")
        if value.get("geometry") != self.config.geometry_mode:
            raise ValueError("Pair geometry metadata mode does not match the run")

    def __call__(self, samples, prompts, outputs, eval=False, **_kwargs):
        rewards, _features = self.scorer.score(list(prompts), list(outputs), evaluation=bool(eval))
        # The legacy Coste evaluator expects an ensemble-variance companion
        # even for a single RM. Pair methods have no ensemble, so return an
        # explicit zero rather than uninitialized tensor memory.
        return rewards, torch.zeros_like(rewards)

    def score_pairs(self, prompts: list[str], outputs: list[str]) -> dict[str, torch.Tensor]:
        if len(prompts) != len(outputs) or len(prompts) % 2:
            raise ValueError("Pair scoring requires adjacent a/b responses with even cardinality")
        if any(prompts[index] != prompts[index + 1] for index in range(0, len(prompts), 2)):
            raise ValueError("Pair responses do not share the same prompt")
        rewards, features = self.scorer.score(prompts, outputs)
        reward_a, reward_b = rewards[0::2], rewards[1::2]
        differences = features[0::2] - features[1::2]
        margins = reward_a - reward_b
        if self.config.method == "cpdpo" and self.config.geometry_mode == "unit":
            uncertainties = torch.ones_like(margins)
        else:
            uncertainties = self.geometry.uncertainty(differences) if self.geometry is not None else None
        signals = pair_signals(
            margins,
            uncertainties,
            method=self.config.method,
            q_alpha=self.q_alpha,
            epsilon=self.config.epsilon,
            reward_variant=self.config.reward_variant,
        )
        signals.update(
            {
                "reward_a": reward_a.detach(),
                "reward_b": reward_b.detach(),
                "feature_difference": differences.detach(),
            }
        )
        return signals

    def provenance(self) -> dict[str, str | float | None]:
        return {
            "proxy_rm_path": self.proxy_rm_path,
            "proxy_rm_fingerprint": self.proxy_rm_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "geometry_fingerprint": self.geometry_fingerprint,
            "geometry_metadata_fingerprint": self.geometry_metadata_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "calibration_scores_fingerprint": self.calibration_scores_fingerprint,
            "data_manifest_fingerprint": self.data_manifest_fingerprint,
            "q_alpha": self.q_alpha,
        }
