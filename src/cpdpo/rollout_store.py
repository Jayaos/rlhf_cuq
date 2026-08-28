"""Atomic pair rollout storage for PairPPO and CPDPO."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.pipeline import BaseRolloutStore


@dataclass
class PairRolloutElement:
    prompt_id: str
    behavior_policy_step: int
    proxy_rm_fingerprint: str
    geometry_fingerprint: str | None
    calibration_fingerprint: str | None
    query_tensor: torch.Tensor
    response_tensor_a: torch.Tensor
    response_tensor_b: torch.Tensor
    old_logprobs_a: torch.Tensor
    old_logprobs_b: torch.Tensor
    ref_logprobs_a: torch.Tensor
    ref_logprobs_b: torch.Tensor
    pair_reward: torch.Tensor
    reward_a: torch.Tensor
    reward_b: torch.Tensor
    margin: torch.Tensor
    uncertainty: torch.Tensor
    normalized_margin: torch.Tensor
    certified: torch.Tensor
    gamma: torch.Tensor


@dataclass
class PairRolloutBatch:
    prompt_ids: list[str]
    behavior_policy_steps: list[int]
    proxy_rm_fingerprints: list[str]
    geometry_fingerprints: list[str | None]
    calibration_fingerprints: list[str | None]
    query_tensors: torch.Tensor
    response_tensors_a: torch.Tensor
    response_tensors_b: torch.Tensor
    old_logprobs_a: torch.Tensor
    old_logprobs_b: torch.Tensor
    ref_logprobs_a: torch.Tensor
    ref_logprobs_b: torch.Tensor
    pair_rewards: torch.Tensor
    rewards_a: torch.Tensor
    rewards_b: torch.Tensor
    margins: torch.Tensor
    uncertainties: torch.Tensor
    normalized_margins: torch.Tensor
    certified: torch.Tensor
    gammas: torch.Tensor


def _pad_queries(elems: list[PairRolloutElement], padding_side: str, pad_token_id: int) -> torch.Tensor:
    if padding_side == "left":
        return pad_sequence(
            [elem.query_tensor.flip(0) for elem in elems], padding_value=pad_token_id, batch_first=True
        ).flip(1)
    if padding_side == "right":
        return pad_sequence([elem.query_tensor for elem in elems], padding_value=pad_token_id, batch_first=True)
    raise ValueError(f"Invalid padding side: {padding_side}")


def pair_collate_fn(padding_side: str, pad_token_id: int, elems: Iterable[PairRolloutElement]) -> PairRolloutBatch:
    items = list(elems)
    if not items:
        raise ValueError("Cannot collate an empty pair batch")
    right_pad = lambda values, padding: pad_sequence(values, padding_value=padding, batch_first=True)
    scalars = lambda name: torch.stack([getattr(elem, name).reshape(()) for elem in items])
    return PairRolloutBatch(
        prompt_ids=[elem.prompt_id for elem in items],
        behavior_policy_steps=[elem.behavior_policy_step for elem in items],
        proxy_rm_fingerprints=[elem.proxy_rm_fingerprint for elem in items],
        geometry_fingerprints=[elem.geometry_fingerprint for elem in items],
        calibration_fingerprints=[elem.calibration_fingerprint for elem in items],
        query_tensors=_pad_queries(items, padding_side, pad_token_id),
        response_tensors_a=right_pad([elem.response_tensor_a for elem in items], pad_token_id),
        response_tensors_b=right_pad([elem.response_tensor_b for elem in items], pad_token_id),
        old_logprobs_a=right_pad([elem.old_logprobs_a for elem in items], 0.0),
        old_logprobs_b=right_pad([elem.old_logprobs_b for elem in items], 0.0),
        ref_logprobs_a=right_pad([elem.ref_logprobs_a for elem in items], 0.0),
        ref_logprobs_b=right_pad([elem.ref_logprobs_b for elem in items], 0.0),
        pair_rewards=scalars("pair_reward"),
        rewards_a=scalars("reward_a"),
        rewards_b=scalars("reward_b"),
        margins=scalars("margin"),
        uncertainties=scalars("uncertainty"),
        normalized_margins=scalars("normalized_margin"),
        certified=scalars("certified").bool(),
        gammas=scalars("gamma"),
    )


class PairRolloutStorage(BaseRolloutStore):
    """Rollout store whose sampling unit is a complete response pair."""

    def __init__(self, pad_token_id: int, padding_side: str):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.history: list[PairRolloutElement] = []

    def push(self, exps: Iterable[PairRolloutElement]) -> None:
        self.history.extend(exps)

    def clear_history(self) -> None:
        self.history = []

    def __getitem__(self, index: int) -> PairRolloutElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(self, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=partial(pair_collate_fn, self.padding_side, self.pad_token_id),
        )

    def export_history(self, location: str, only_text: bool = False) -> None:
        target = Path(location)
        if not target.is_dir():
            raise FileNotFoundError(target)
        rows = []
        for elem in self.history:
            row = {"prompt_id": elem.prompt_id}
            for key, value in elem.__dict__.items():
                if key != "prompt_id":
                    row[key] = value.detach().cpu().tolist()
            if only_text:
                row = {
                    key: row[key]
                    for key in ("prompt_id", "query_tensor", "response_tensor_a", "response_tensor_b")
                }
            rows.append(row)
        path = target / f"pair-epoch-{time.time()}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)

    def save_rollout_snapshot(self, output_dir: str | Path, rollout_step: int) -> Path:
        """Persist the complete atomic pair records for one rollout."""

        target_dir = Path(output_dir) / "pair_rollouts"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"rollout_{rollout_step:06d}.pt"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite pair rollout snapshot: {target}")
        records = []
        for elem in self.history:
            row = {
                "prompt_id": elem.prompt_id,
                "behavior_policy_step": elem.behavior_policy_step,
                "proxy_rm_fingerprint": elem.proxy_rm_fingerprint,
                "geometry_fingerprint": elem.geometry_fingerprint,
                "calibration_fingerprint": elem.calibration_fingerprint,
            }
            for key, value in elem.__dict__.items():
                if key not in row:
                    row[key] = value.detach().cpu()
            row["response_mask_a"] = torch.ones_like(elem.response_tensor_a, dtype=torch.bool)
            row["response_mask_b"] = torch.ones_like(elem.response_tensor_b, dtype=torch.bool)
            records.append(row)
        payload = {
            "schema_version": "1.0.0",
            "rollout_step": int(rollout_step),
            "pair_count": len(records),
            "records": records,
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target_dir)
        os.close(descriptor)
        try:
            with Path(temporary_name).open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return target
