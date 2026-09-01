"""PairPPO/CPDPO trainer layered on the smoke-tested Coste/trlx trainer."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.trainer import register_trainer
from trlx.utils import infinite_dataloader
from trlx.utils.modeling import logprobs_of_labels

from src.cpdpo.pair_loss import pairwise_clipped_loss
from src.cpdpo.experiment_checkpoint import ExperimentCheckpointMixin
from src.cpdpo.pair_reward import PairRewardCallback
from src.cpdpo.random_stream import GenerationSeedStream
from src.cpdpo.optimizer import install_gradient_clipping
from src.cpdpo.run_logging import append_rollout_record
from src.cpdpo.rollout_store import PairRolloutBatch, PairRolloutElement, PairRolloutStorage
from src.cpdpo.spec import CPDPOConfig
from src.ppo.custom_trlx_trainers.custom_accelerate_ppo_trainer import CustomAcceleratePPOTrainer


logger = logging.get_logger(__name__)


@register_trainer
class CustomAcceleratePairPPOTrainer(ExperimentCheckpointMixin, CustomAcceleratePPOTrainer):
    """Atomic pair trainer with no value loss or GAE."""

    def __init__(
        self,
        config,
        *,
        pair_method_config: dict[str, Any],
        experiment_seeds: dict[str, int],
        experiment_context: dict[str, Any],
        resume_from_checkpoint: str | None,
        max_grad_norm: float,
        **kwargs,
    ):
        self.pair_config = CPDPOConfig.from_mapping(pair_method_config)
        self.experiment_seeds = dict(experiment_seeds)
        super().__init__(config, **kwargs)
        if self.accelerator.num_processes != 1:
            raise RuntimeError(
                "The audited CPDPO v1 rollout collector currently supports one process; "
                "use one GPU rather than silently mis-pairing distributed generations"
            )
        if self.config.model.model_arch_type != "causal":
            raise RuntimeError("CPDPO v1 is implemented for the Coste causal Pythia policy")
        if not isinstance(self.reward_fn, PairRewardCallback):
            raise TypeError("Pair trainer requires PairRewardCallback")
        self.store = PairRolloutStorage(self.tokenizer.pad_token_id, self.tokenizer.padding_side)
        self.low_certification_rollouts = 0
        self.rollout_seed_stream = GenerationSeedStream(
            int(self.experiment_seeds["rollout_generation"]), self.accelerator.device
        )
        self.evaluation_seed_stream = GenerationSeedStream(
            int(self.experiment_seeds["evaluation_generation"]), self.accelerator.device
        )
        install_gradient_clipping(self, max_grad_norm)
        self.configure_experiment_checkpointing(
            resume_from_checkpoint=resume_from_checkpoint,
            experiment_context=experiment_context,
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        with self.rollout_seed_stream.activate():
            return super().generate(input_ids, attention_mask=attention_mask, **kwargs)

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        with self.evaluation_seed_stream.activate():
            return super().generate_eval(input_ids, attention_mask=attention_mask, **kwargs)

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=False)
        prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = infinite_dataloader(prompt_dataloader)

    def create_train_dataloader(self):
        return self.store.create_loader(self.config.train.batch_size, shuffle=True)

    def _new_logprobs(self, query_tensors, response_tensors):
        tokens = torch.cat((query_tensors, response_tensors), dim=1)
        attention_mask = tokens.ne(self.tokenizer.pad_token_id).long()
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        outputs = self.model(tokens, attention_mask, return_dict=True, position_ids=position_ids)
        logprobs = logprobs_of_labels(outputs.logits[:, :-1, :], tokens[:, 1:])
        start = query_tensors.shape[1] - 1
        end = start + response_tensors.shape[1]
        return logprobs[:, start:end], attention_mask[:, start + 1 : end + 1], outputs.logits[:, start:end]

    def loss(self, batch: PairRolloutBatch) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if len(set(batch.behavior_policy_steps)) != 1:
            raise RuntimeError("A pair minibatch mixed behavior-policy snapshots")
        if set(batch.proxy_rm_fingerprints) != {self.reward_fn.proxy_rm_fingerprint}:
            raise RuntimeError("A pair minibatch contains a mismatched proxy-RM fingerprint")
        if set(batch.geometry_fingerprints) != {self.reward_fn.geometry_fingerprint}:
            raise RuntimeError("A pair minibatch contains a mismatched geometry fingerprint")
        if set(batch.calibration_fingerprints) != {self.reward_fn.calibration_fingerprint}:
            raise RuntimeError("A pair minibatch contains a mismatched calibration fingerprint")
        device = self.accelerator.device
        query = batch.query_tensors.to(device)
        response_a = batch.response_tensors_a.to(device)
        response_b = batch.response_tensors_b.to(device)
        response_width = max(response_a.shape[1], response_b.shape[1])
        response_a = F.pad(response_a, (0, response_width - response_a.shape[1]), value=self.tokenizer.pad_token_id)
        response_b = F.pad(response_b, (0, response_width - response_b.shape[1]), value=self.tokenizer.pad_token_id)
        doubled_query = torch.cat((query, query), dim=0)
        doubled_response = torch.cat((response_a, response_b), dim=0)
        logprobs, masks, logits = self._new_logprobs(doubled_query, doubled_response)
        pair_count = query.shape[0]
        logprobs_a, logprobs_b = logprobs[:pair_count], logprobs[pair_count:]
        mask_a, mask_b = masks[:pair_count], masks[pair_count:]

        def padded(value):
            value = value.to(device)
            return F.pad(value, (0, response_width - value.shape[1]), value=0.0)

        loss, stats = pairwise_clipped_loss(
            logprobs_a=logprobs_a,
            logprobs_b=logprobs_b,
            old_logprobs_a=padded(batch.old_logprobs_a),
            old_logprobs_b=padded(batch.old_logprobs_b),
            ref_logprobs_a=padded(batch.ref_logprobs_a),
            ref_logprobs_b=padded(batch.ref_logprobs_b),
            mask_a=mask_a,
            mask_b=mask_b,
            pair_rewards=batch.pair_rewards.to(device),
            clip_epsilon=self.pair_config.clip_epsilon,
            kl_beta=self.pair_config.kl_beta,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite paired PPO loss before backward at optimizer step "
                f"{getattr(self, 'iter_count', 0) + 1}; no optimizer step was applied"
            )
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -(probabilities * torch.log_softmax(logits.float(), dim=-1)).sum(-1)
        stats.update(
            {
                "policy/entropy": (entropy * masks).sum().detach() / masks.sum().clamp_min(1),
                "pair/reward_abs_mean": batch.pair_rewards.abs().mean().to(device),
                "pair/reward_zero_fraction": (batch.pair_rewards == 0).float().mean().to(device),
            }
        )
        if self.pair_config.method == "cpdpo":
            stats["pair/certification_rate"] = batch.certified.float().mean().to(device)
        return loss, stats

    @staticmethod
    def _metadata_list(batch: PromptBatch, key: str) -> list:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return list(value)

    def _update_certification_monitor(self, certification_rate: float) -> None:
        if self.pair_config.method != "cpdpo":
            self.low_certification_rollouts = 0
            return
        if certification_rate < self.pair_config.certification_warning_threshold:
            self.low_certification_rollouts += 1
            if self.low_certification_rollouts >= self.pair_config.certification_warning_patience:
                logger.warning(
                    "CPDPO certification rate %.4f has stayed below %.4f for %d rollouts; no fallback is applied",
                    certification_rate,
                    self.pair_config.certification_warning_threshold,
                    self.low_certification_rollouts,
                )
        else:
            self.low_certification_rollouts = 0

    def make_experience(self, num_rollouts: int = 256, iter_count: int = 0):  # noqa: C901
        logger.info("Collecting atomic response pairs")
        pair_elements: list[PairRolloutElement] = []
        all_stats: list[dict[str, float]] = []
        while len(pair_elements) < num_rollouts:
            batch: PromptBatch = next(self.prompt_iterator)
            if batch.input_ids.shape[0] % 2:
                raise RuntimeError("Pair prompt chunk size must be even")
            prompt_ids = self._metadata_list(batch, "prompt_id")
            orientations = self._metadata_list(batch, "orientation")
            for index in range(0, len(prompt_ids), 2):
                if prompt_ids[index] != prompt_ids[index + 1] or orientations[index : index + 2] != ["a", "b"]:
                    raise RuntimeError("Shared prompt schedule lost adjacent a/b pair ordering")

            samples = self.generate(batch["input_ids"], batch["attention_mask"])
            prompt_tensors = batch.input_ids
            prompt_sizes = torch.full(
                (prompt_tensors.shape[0],), prompt_tensors.shape[1], device=samples.device, dtype=torch.long
            )
            _str_samples, str_prompts, str_outputs = self.decode(
                prompt_tensors, samples, prompt_sizes, append_eos_token=True
            )
            signals = self.reward_fn.score_pairs(str_prompts, str_outputs)

            output_ids = self.tokenizer(str_outputs).input_ids
            output_tensors = [torch.as_tensor(value, dtype=torch.long) for value in output_ids]
            max_output = max(value.shape[0] for value in output_tensors)
            sample_outputs = torch.stack(
                [F.pad(value, (0, max_output - value.shape[0]), value=self.tokenizer.pad_token_id) for value in output_tensors]
            ).to(samples.device)

            all_tokens = torch.cat((prompt_tensors.to(samples.device), sample_outputs), dim=1)
            attention_mask = all_tokens.ne(self.tokenizer.pad_token_id).long()
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            with torch.no_grad():
                logits, *_unused, _values = self.model(
                    all_tokens, attention_mask=attention_mask, position_ids=position_ids
                )
                if hasattr(self.model, "frozen_head") or self.model.peft_type:
                    ref_logits = self.model.forward_hydra(
                        all_tokens, attention_mask=attention_mask, position_ids=position_ids, return_dict=True
                    ).logits
                else:
                    ref_logits = self.ref_model(
                        all_tokens, attention_mask=attention_mask, position_ids=position_ids, return_dict=True
                    ).logits.to(samples.device)
            old_all = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:]).cpu()
            ref_all = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:]).cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            attention_mask = attention_mask.cpu()
            start = prompt_tensors.shape[1] - 1
            response_lengths = attention_mask[:, prompt_tensors.shape[1] :].sum(1)

            batch_kls = []
            for pair_index, sample_index in enumerate(range(0, len(prompt_ids), 2)):
                tensors = []
                for orientation_index in (sample_index, sample_index + 1):
                    length = int(response_lengths[orientation_index].item())
                    tensors.append(
                        (
                            sample_outputs[orientation_index, :length],
                            old_all[orientation_index, start : start + length],
                            ref_all[orientation_index, start : start + length],
                        )
                    )
                    log_ratio = tensors[-1][1] - tensors[-1][2]
                    batch_kls.append(float((torch.exp(log_ratio) - 1 - log_ratio).sum().item()))
                get = lambda name: signals[name][pair_index].detach().cpu().reshape(())
                pair_elements.append(
                    PairRolloutElement(
                        prompt_id=str(prompt_ids[sample_index]),
                        behavior_policy_step=int(iter_count),
                        proxy_rm_fingerprint=self.reward_fn.proxy_rm_fingerprint,
                        geometry_fingerprint=self.reward_fn.geometry_fingerprint,
                        calibration_fingerprint=self.reward_fn.calibration_fingerprint,
                        query_tensor=prompt_tensors[sample_index],
                        response_tensor_a=tensors[0][0],
                        response_tensor_b=tensors[1][0],
                        old_logprobs_a=tensors[0][1],
                        old_logprobs_b=tensors[1][1],
                        ref_logprobs_a=tensors[0][2],
                        ref_logprobs_b=tensors[1][2],
                        pair_reward=get("pair_reward"),
                        reward_a=get("reward_a"),
                        reward_b=get("reward_b"),
                        margin=get("margin"),
                        uncertainty=get("uncertainty"),
                        normalized_margin=get("normalized_margin"),
                        certified=get("certified"),
                        gamma=get("gamma"),
                    )
                )
            self.mean_kl = sum(batch_kls) / len(batch_kls)
            finite_normalized = signals["normalized_margin"][torch.isfinite(signals["normalized_margin"])]
            uncertainty = signals["uncertainty"].float()
            abs_margin = signals["margin"].abs().float()
            reward_a = signals["reward_a"].float()
            reward_b = signals["reward_b"].float()
            response_lengths_chunk = response_lengths.float()
            chunk_stats = {
                    "pair/identical_response_rate": sum(
                        a == b for a, b in zip(str_outputs[0::2], str_outputs[1::2])
                    )
                    / len(signals["margin"]),
                    "pair/margin_mean": float(signals["margin"].float().mean().item()),
                    "pair/margin_std": float(signals["margin"].float().std(unbiased=False).item()),
                    "pair/margin_abs_mean": float(abs_margin.mean().item()),
                    "pair/margin_abs_median": float(abs_margin.median().item()),
                    "pair/gamma_mean": float(signals["gamma"].mean().item()),
                    "pair/reward_abs_mean": float(signals["pair_reward"].abs().mean().item()),
                    "pair/reward_zero_fraction": float((signals["pair_reward"] == 0).float().mean().item()),
                    "proxy/reward_a_mean": float(reward_a.mean().item()),
                    "proxy/reward_b_mean": float(reward_b.mean().item()),
                    "proxy/winner_mean": float(torch.maximum(reward_a, reward_b).mean().item()),
                    "proxy/loser_mean": float(torch.minimum(reward_a, reward_b).mean().item()),
                    "rollout/response_length_a_mean": float(response_lengths_chunk[0::2].mean().item()),
                    "rollout/response_length_b_mean": float(response_lengths_chunk[1::2].mean().item()),
                    "rollout/generated_tokens": float(response_lengths_chunk.sum().item()),
                    "rollout/proxy_rm_calls": float(len(str_outputs)),
                    "rollout/response_count": float(len(str_outputs)),
                    "rollout/pair_count": float(len(str_outputs) // 2),
                    "rollout/kl": self.mean_kl,
            }
            if self.pair_config.method == "cpdpo":
                chunk_stats.update(
                    {
                        "pair/certification_rate": float(signals["certified"].float().mean().item()),
                        "pair/effective_pair_count": float(signals["certified"].sum().item()),
                        "pair/uncertainty_mean": float(uncertainty.mean().item()),
                        "pair/uncertainty_median": float(uncertainty.median().item()),
                        "pair/uncertainty_p90": float(torch.quantile(uncertainty, 0.90).item()),
                        "pair/uncertainty_p95": float(torch.quantile(uncertainty, 0.95).item()),
                        "pair/normalized_margin_mean": float(finite_normalized.mean().item()),
                        "pair/normalized_margin_p90": float(torch.quantile(finite_normalized, 0.90).item()),
                    }
                )
            all_stats.append(chunk_stats)
            if len(pair_elements) > num_rollouts:
                raise RuntimeError("Pair rollout chunk does not divide num_rollouts")

        certification_rate = (
            sum(item["pair/certification_rate"] for item in all_stats) / len(all_stats)
            if self.pair_config.method == "cpdpo"
            else 1.0
        )
        self._update_certification_monitor(certification_rate)
        count_keys = {
            "pair/effective_pair_count",
            "rollout/generated_tokens",
            "rollout/proxy_rm_calls",
            "rollout/response_count",
            "rollout/pair_count",
        }
        aggregated_stats = {
                key: (sum(item[key] for item in all_stats) if key in count_keys else sum(item[key] for item in all_stats) / len(all_stats))
                for key in all_stats[0]
        }
        self.accelerator.log(aggregated_stats, step=iter_count)
        self.push_to_store(pair_elements)
        if self.accelerator.is_main_process:
            updates_per_rollout = (
                self.config.method.num_rollouts // self.config.train.batch_size
            ) * self.config.method.ppo_epochs
            rollout_step = iter_count // updates_per_rollout + 1
            snapshot = self.store.save_rollout_snapshot(
                self.config.train.output_dir,
                rollout_step,
            )
            record = {
                "schema_version": "1.0.0",
                "method": self.pair_config.method,
                "rollout_step": rollout_step,
                "pair_rollout_snapshot": str(snapshot),
                "response_count": int(aggregated_stats["rollout/response_count"]),
                "generated_token_count": int(aggregated_stats["rollout/generated_tokens"]),
                "proxy_call_count": int(aggregated_stats["rollout/proxy_rm_calls"]),
                "q_alpha": self.reward_fn.q_alpha,
                **aggregated_stats,
            }
            append_rollout_record(self.config.train.output_dir, record)
        logger.info("Collected %d response pairs", len(pair_elements))
