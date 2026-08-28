"""Fair-budget scalar PPO control for the CPDPO experiment."""

from trlx.pipeline.offline_pipeline import PromptPipeline
from trlx.trainer import register_trainer
from trlx.utils import infinite_dataloader

from src.ppo.custom_trlx_trainers.custom_accelerate_ppo_trainer import CustomAcceleratePPOTrainer
from src.cpdpo.random_stream import GenerationSeedStream
from src.cpdpo.experiment_checkpoint import ExperimentCheckpointMixin
from src.cpdpo.optimizer import install_gradient_clipping
from src.cpdpo.run_logging import append_rollout_record


@register_trainer
class ExperimentAcceleratePPOTrainer(ExperimentCheckpointMixin, CustomAcceleratePPOTrainer):
    """Ordinary PPO/GAE with the exact shared, pre-materialized prompt order."""

    def __init__(
        self,
        config,
        *,
        experiment_seeds: dict,
        experiment_context: dict,
        resume_from_checkpoint: str | None,
        max_grad_norm: float,
        **kwargs,
    ):
        self.experiment_seeds = dict(experiment_seeds)
        super().__init__(config, **kwargs)
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

    def add_prompt_pipeline(self, pipeline: PromptPipeline):
        # The schedule is shuffled once when it is materialized. Shuffling its
        # duplicated a/b entries here would destroy the common method budget.
        prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=False)
        prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = infinite_dataloader(prompt_dataloader)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        with self.rollout_seed_stream.activate():
            return super().generate(input_ids, attention_mask=attention_mask, **kwargs)

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        with self.evaluation_seed_stream.activate():
            return super().generate_eval(input_ids, attention_mask=attention_mask, **kwargs)

    def make_experience(self, num_rollouts: int = 512, iter_count: int = 0):
        super().make_experience(num_rollouts, iter_count)
        if self.accelerator.is_main_process:
            updates_per_rollout = (
                self.config.method.num_rollouts // self.config.train.batch_size
            ) * self.config.method.ppo_epochs
            append_rollout_record(
                self.config.train.output_dir,
                {
                    "schema_version": "1.0.0",
                    "method": "ppo",
                    "rollout_step": iter_count // updates_per_rollout + 1,
                    "response_count": len(self.store.history),
                    "generated_token_count": sum(
                        int(element.response_tensor.ne(self.tokenizer.pad_token_id).sum().item())
                        for element in self.store.history
                    ),
                    "proxy_call_count": len(self.store.history),
                },
            )
