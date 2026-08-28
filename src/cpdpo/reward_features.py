"""Exact scalar-head feature extraction for the Coste proxy reward model."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F


MAX_PROXY_SEQUENCE_LENGTH = 776


def format_proxy_samples(prompts: Iterable[str], outputs: Iterable[str], *, evaluation: bool = False) -> list[str]:
    suffix = "<|endoftext|>" if evaluation else ""
    return [
        f"<|prompter|>{prompt}<|endoftext|><|assistant|>{output}{suffix}"
        for prompt, output in zip(prompts, outputs)
    ]


class RewardHeadFeatureExtractor:
    """Capture the exact input to `out_proj` while preserving model logits."""

    def __init__(self, model: torch.nn.Module, *, atol: float | None = None, rtol: float | None = None):
        head = getattr(model, "out_proj", None)
        if not isinstance(head, torch.nn.Linear) or head.out_features != 1:
            raise TypeError("CPDPO requires GPTNeoXRewardModel.out_proj = Linear(hidden_size, 1)")
        self.model = model
        self.head = head
        self.atol = atol
        self.rtol = rtol

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        captured: list[torch.Tensor] = []

        def capture(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            if len(args) != 1:
                raise RuntimeError("Unexpected reward-head input signature")
            captured.append(args[0])

        handle = self.head.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one reward-head invocation, observed {len(captured)}")
        feature = captured[0].detach()
        logits = output.logits[:, 0].detach()
        reconstructed = F.linear(feature, self.head.weight, self.head.bias)[:, 0]
        if self.atol is None:
            atol = 2e-3 if logits.dtype in {torch.float16, torch.bfloat16} else 1e-6
        else:
            atol = self.atol
        if self.rtol is None:
            rtol = 2e-3 if logits.dtype in {torch.float16, torch.bfloat16} else 1e-5
        else:
            rtol = self.rtol
        if not torch.allclose(logits, reconstructed, atol=atol, rtol=rtol):
            error = float((logits - reconstructed).abs().max().item())
            raise RuntimeError(f"Reward-head feature identity failed; max absolute error={error}")
        return logits, feature


class ProxyRewardFeatureScorer:
    """Frozen proxy scorer that returns scalar rewards and exact head features."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device, *, batch_size: int = 32):
        self.model = model.eval().requires_grad_(False).to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.extractor = RewardHeadFeatureExtractor(self.model)
        self.calls = 0

    def score(
        self,
        prompts: list[str],
        outputs: list[str],
        *,
        evaluation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(prompts) != len(outputs):
            raise ValueError("prompts and outputs must have equal length")
        samples = format_proxy_samples(prompts, outputs, evaluation=evaluation)
        reward_batches: list[torch.Tensor] = []
        feature_batches: list[torch.Tensor] = []
        for start in range(0, len(samples), self.batch_size):
            tokenized = self.tokenizer(
                samples[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_PROXY_SEQUENCE_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            rewards, features = self.extractor.forward(tokenized.input_ids, tokenized.attention_mask)
            reward_batches.append(rewards.cpu())
            feature_batches.append(features.cpu())
        self.calls += len(samples)
        if not reward_batches:
            raise ValueError("Cannot score an empty sample list")
        return torch.cat(reward_batches), torch.cat(feature_batches)


def load_proxy_feature_scorer(model_path: str, *, device: str | torch.device, batch_size: int = 32):
    # Importing this module registers the pinned custom GPT-NeoX RM class.
    from src.reward_modeling.scoring import score as _score_registration  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return ProxyRewardFeatureScorer(model, tokenizer, torch.device(device), batch_size=batch_size)
