"""Rollout-boundary checkpoint/resume for the additive experiment trainers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.cpdpo.artifacts import atomic_write_json, canonical_json_hash
from src.cpdpo.run_logging import archive_pair_rollouts_after_checkpoint, rewind_rollout_records


class ExperimentCheckpointMixin:
    """Add exact experiment counters around the pinned trlx state checkpoint."""

    _iter_count_value = 0
    _preserve_resume_iter_reset = False
    _nth_evaluation_value = 0
    _preserve_resume_eval_reset = False

    @property
    def iter_count(self) -> int:
        return int(self._iter_count_value)

    @iter_count.setter
    def iter_count(self, value: int) -> None:
        if self._preserve_resume_iter_reset and int(value) == 0:
            self._preserve_resume_iter_reset = False
            return
        self._iter_count_value = int(value)

    @property
    def nth_evaluation(self) -> int:
        return int(self._nth_evaluation_value)

    @nth_evaluation.setter
    def nth_evaluation(self, value: int) -> None:
        if self._preserve_resume_eval_reset and int(value) == 0:
            self._preserve_resume_eval_reset = False
            return
        self._nth_evaluation_value = int(value)

    def configure_experiment_checkpointing(
        self,
        *,
        resume_from_checkpoint: str | None,
        experiment_context: dict[str, Any],
    ) -> None:
        self.experiment_context = dict(experiment_context)
        self.experiment_context_hash = canonical_json_hash(self.experiment_context)
        self.resume_checkpoint = Path(resume_from_checkpoint).resolve() if resume_from_checkpoint else None
        self.resume_completed_rollouts = 0
        if self.resume_checkpoint is None:
            return
        state_path = self.resume_checkpoint / "experiment_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"Experiment checkpoint metadata is missing: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != "1.0.0":
            raise ValueError("Unsupported experiment checkpoint schema")
        if state.get("experiment_context_hash") != self.experiment_context_hash:
            raise ValueError("Resume checkpoint does not match this run's immutable context")
        if state.get("experiment_context") != self.experiment_context:
            raise ValueError("Resume checkpoint context hash matched but content did not")
        updates_per_rollout = int(self.experiment_context["updates_per_rollout"])
        resume_iter = int(state["optimizer_step"])
        if resume_iter <= 0 or resume_iter % updates_per_rollout:
            raise ValueError("Resume checkpoint is not on a positive rollout boundary")
        completed = resume_iter // updates_per_rollout
        if completed != int(state["completed_rollouts"]):
            raise ValueError("Resume checkpoint rollout and optimizer counters disagree")
        if completed >= int(self.experiment_context["rollout_steps"]):
            raise ValueError("The requested checkpoint already completed the configured run")
        chunks_per_rollout = int(self.experiment_context["responses_per_rollout"]) // int(
            self.experiment_context["prompt_chunk_size"]
        )
        expected_rollout_calls = completed * chunks_per_rollout
        if int(state["rollout_seed_stream_counter"]) != expected_rollout_calls:
            raise ValueError("Resume checkpoint generation counter disagrees with the prompt schedule")

        self.load(str(self.resume_checkpoint))
        self._iter_count_value = resume_iter
        self._preserve_resume_iter_reset = True
        self.resume_completed_rollouts = completed
        self.evaluation_seed_stream.counter = int(state["evaluation_seed_stream_counter"])
        self.mb_count = resume_iter * self.num_mb
        self.low_certification_rollouts = int(state.get("low_certification_rollouts", 0))
        reward_state = state.get("reward_callback_state")
        if reward_state is not None:
            if not hasattr(self.reward_fn, "load_state_dict"):
                raise ValueError("Checkpoint contains reward state but this callback cannot restore it")
            self.reward_fn.load_state_dict(reward_state)
        elif hasattr(self.reward_fn, "load_state_dict"):
            raise ValueError("Stateful reward callback is missing from the resume checkpoint")

        eval_dir = Path(self.config.train.output_dir) / "eval"
        prior_evaluations = []
        for path in eval_dir.glob("eval-*.json"):
            try:
                prior_evaluations.append(int(path.stem.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
        self._nth_evaluation_value = max(prior_evaluations, default=-1) + 1
        self._preserve_resume_eval_reset = True
        if self.accelerator.is_main_process:
            rewind_rollout_records(self.config.train.output_dir, completed)
            archive_pair_rollouts_after_checkpoint(self.config.train.output_dir, completed)
        self.accelerator.wait_for_everyone()

    def prepare_learning(self) -> None:
        eval_dataloader = self.eval_pipeline.create_loader(self.config.method.chunk_size)
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)

        chunks_per_rollout = int(self.experiment_context["responses_per_rollout"]) // int(
            self.config.method.chunk_size
        )
        if chunks_per_rollout * int(self.config.method.chunk_size) != int(
            self.experiment_context["responses_per_rollout"]
        ):
            raise ValueError("Response budget must be divisible by the prompt chunk size")
        skipped_chunks = self.resume_completed_rollouts * chunks_per_rollout
        for _ in range(skipped_chunks):
            next(self.prompt_iterator)
        self.rollout_seed_stream.counter = skipped_chunks
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

        self.train_dataloader = self.create_train_dataloader()
        self.n_inner_epochs = self.config.method.ppo_epochs
        self.total_steps = self.config.train.epochs * self.n_inner_epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)
        if self.iter_count >= self.total_steps:
            raise ValueError("Resume checkpoint has no remaining optimizer steps")

    def save(self, directory: str | None = None, **kwargs) -> None:
        super().save(directory, **kwargs)
        target = Path(directory or self.config.train.checkpoint_dir)
        updates_per_rollout = int(self.experiment_context["updates_per_rollout"])
        if self.iter_count % updates_per_rollout:
            raise RuntimeError("Experiment checkpoints may only be saved on rollout boundaries")
        if self.accelerator.is_main_process:
            reward_state = (
                self.reward_fn.state_dict() if hasattr(self.reward_fn, "state_dict") else None
            )
            atomic_write_json(
                target / "experiment_state.json",
                {
                    "schema_version": "1.0.0",
                    "optimizer_step": self.iter_count,
                    "completed_rollouts": self.iter_count // updates_per_rollout,
                    "rollout_seed_stream_counter": self.rollout_seed_stream.counter,
                    "evaluation_seed_stream_counter": self.evaluation_seed_stream.counter,
                    "low_certification_rollouts": int(getattr(self, "low_certification_rollouts", 0)),
                    "reward_callback_state": reward_state,
                    "experiment_context_hash": self.experiment_context_hash,
                    "experiment_context": self.experiment_context,
                },
            )
        self.accelerator.wait_for_everyone()
