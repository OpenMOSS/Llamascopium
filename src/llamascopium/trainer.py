import json
import math
import os
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Literal, Tuple

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed.tensor
import torch.optim.lr_scheduler as lr_scheduler
from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
)
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.optim import Adam, Optimizer
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from llamascopium.config import BaseConfig
from llamascopium.metrics import (
    ExplainedVarianceMetric,
    FrequencyMetric,
    GradientNormMetric,
    L0Metric,
    L2NormErrorMetric,
    LossMetric,
    MeanFeatureActMetric,
    Metric,
    ModelSpecificMetric,
)
from llamascopium.models.sparse_dictionary import SparseDictionary
from llamascopium.optim import SparseAdam, clip_grad_norm, get_scheduler
from llamascopium.utils.distributed import is_primary_rank
from llamascopium.utils.distributed.ops import item
from llamascopium.utils.logging import get_distributed_logger, log_metrics
from llamascopium.utils.misc import (
    convert_str_to_torch_dtype,
    convert_torch_dtype_to_str,
)
from llamascopium.utils.tensor_specs import apply_token_mask
from llamascopium.utils.timer import timer

logger = get_distributed_logger("trainer")


class TrainerConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    l1_coefficient: float | None = 0.00008
    """Coefficient for the L1 sparsity loss. This loss is used to penalize the sparsity of the feature activations."""
    l1_coefficient_warmup_steps: int | float = 0.1
    """Steps (int) or fraction of total steps (float) to warm up the sparsity coefficient from 0."""
    lp_coefficient: float | None = None
    """Coefficient for the Lp sparsity loss. This loss is used to . To use the JumpReLU $L^p$ penalty, set lp_coefficient to a positive value."""
    auxk_coefficient: float | None = None
    """Coefficient for the Aux-K auxiliary loss. This loss is used to revive dead latents during training. To use the Aux-K loss, set auxk_coefficient to a positive value."""
    amp_dtype: Annotated[
        torch.dtype | None,
        BeforeValidator(lambda v: convert_str_to_torch_dtype(v) if isinstance(v, str) else v),
        PlainSerializer(convert_torch_dtype_to_str),
        WithJsonSchema(
            {
                "type": ["string", "null"],
            },
            mode="serialization",
        ),
    ] = Field(default=torch.bfloat16, exclude=True, validate_default=False)
    sparsity_loss_type: Literal["power", "tanh", "tanh-quad", "jumprelu-l0-quad", None] = None
    target_l0: float | None = None
    """Target average number of active latents for target-L0 sparsity losses."""
    tanh_stretch_coefficient: float = 4.0
    frequency_scale: float = 0.01
    p: int = 1
    initial_k: int | float | None = None
    k_warmup_steps: int | float = 0.1
    k_cold_booting_steps: int | float = 0
    k_schedule_type: Literal["linear", "exponential"] = "linear"
    k_exponential_factor: float = 3.0
    k_aux: int = 512
    dead_threshold: float = 10_000_000
    skip_metrics_calculation: bool = False
    gradient_accumulation_steps: int = 1

    lr: float | dict[str, float] = 0.0004
    betas: Tuple[float, float] = (0.9, 0.999)
    optimizer_class: Literal["adam", "sparseadam"] = "adam"
    optimizer_foreach: bool = True
    lr_scheduler_name: Literal[
        "constant",
        "constantwithwarmup",
        "linearwarmupdecay",
        "cosineannealing",
        "cosineannealingwarmup",
        "exponentialwarmup",
    ] = "constantwithwarmup"
    lr_end_ratio: float = 1 / 32
    lr_warm_up_steps: int | float = 5000
    lr_cool_down_steps: int | float = 0.2
    jumprelu_lr_factor: float = 1.0
    clip_grad_norm: float = 0.0
    feature_sampling_window: int = 1000
    total_training_tokens: int = 300_000_000

    log_frequency: int = 1000
    eval_frequency: int = 1000
    n_checkpoints: int = 10
    check_point_save_mode: Literal["log", "linear"] = "log"

    from_pretrained_path: str | None = None
    exp_result_path: str = "results"

    def model_post_init(self, __context):
        super().model_post_init(__context)
        Path(self.exp_result_path).mkdir(parents=True, exist_ok=True)
        Path(self.exp_result_path, "checkpoints").mkdir(parents=True, exist_ok=True)
        assert self.lr_end_ratio <= 1, "lr_end_ratio must be in 0 to 1 (inclusive)."

        if self.from_pretrained_path is not None:
            assert os.path.exists(self.from_pretrained_path), (
                f"from_pretrained_path {self.from_pretrained_path} does not exist"
            )


class WandbConfig(BaseConfig):
    wandb_project: str = "gpt2-sae-training"
    exp_name: str | None = None
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    wandb_resume: Literal["allow", "must", "never", "auto"] = "never"


class Trainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.checkpoint_thresholds: list[int] = []
        self.total_training_steps: int = 0
        self.lr_warm_up_steps: int = 0
        self.lr_cool_down_steps: int = 0
        self.k_warmup_steps: int = 0
        self.k_cold_booting_steps: int = 0
        self.l1_coefficient_warmup_steps: int = 0
        self.cur_step: int = 0
        self.cur_tokens: int = 0
        self.optimizer: Optimizer | None = None
        self.scheduler: lr_scheduler.LRScheduler | None = None
        self.wandb_logger: Run | None = None
        self.metrics: list[Metric] = []
        # Dead statistics for auxk loss
        self.tokens_since_last_activation: Tensor | None = None
        self.is_dead: Tensor | None = None

    def save_checkpoint(self, sae: SparseDictionary, checkpoint_path: Path | str) -> None:
        """
        Save a complete checkpoint including model, optimizer, scheduler, and
        trainer state.

        Args:
            sae: The sparse autoencoder model to save
            checkpoint_path: Path where to save the checkpoint (without extension)
        """

        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(checkpoint_path) / "checkpoints" / f"step_{self.cur_step}"

        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        sae.cfg.save_hyperparameters(checkpoint_dir)
        # Save model state
        if sae.device_mesh is None:
            sae.save_checkpoint(checkpoint_dir / "sae_weights.safetensors")
        else:
            sae.save_checkpoint(checkpoint_dir / "sae_weights.dcp")

        if is_primary_rank(sae.device_mesh):
            # Prepare trainer state
            trainer_state = {
                "cur_step": self.cur_step,
                "cur_tokens": self.cur_tokens,
                "total_training_steps": self.total_training_steps,
                "lr_warm_up_steps": self.lr_warm_up_steps,
                "lr_cool_down_steps": self.lr_cool_down_steps,
                "k_warmup_steps": self.k_warmup_steps,
                "k_cold_booting_steps": self.k_cold_booting_steps,
                "l1_coefficient_warmup_steps": self.l1_coefficient_warmup_steps,
                "checkpoint_thresholds": self.checkpoint_thresholds,
                "cfg": self.cfg,
            }
            # Save trainer state
            trainer_path = checkpoint_dir / "trainer.pt"
            torch.save(trainer_state, trainer_path)
            if self.wandb_logger is not None:
                with open(checkpoint_dir / "wandb_run_id.json", "w") as f:
                    json.dump({"wandb_run_id": self.wandb_logger.id}, f)
        # Save optimizer state - handle distributed tensors
        if self.optimizer is not None:
            if sae.device_mesh is None:
                if is_primary_rank(sae.device_mesh):
                    optimizer_path = checkpoint_dir / "optimizer.pt"
                    optimizer_state = self.optimizer.state_dict()
                    torch.save(optimizer_state, optimizer_path)
            else:
                optimizer_path = checkpoint_dir / "optimizer.dcp"
                optimizer_state = self.optimizer.state_dict()
                fs_writer = FileSystemWriter(optimizer_path)
                dcp.save(optimizer_state, storage_writer=fs_writer)

        # Save scheduler state - handle distributed tensors
        if self.scheduler is not None:
            if sae.device_mesh is None:
                if is_primary_rank(sae.device_mesh):
                    scheduler_path = checkpoint_dir / "scheduler.pt"
                    scheduler_state = self.scheduler.state_dict()
                    torch.save(scheduler_state, scheduler_path)
            else:
                scheduler_path = checkpoint_dir / "scheduler.dcp"
                scheduler_state = self.scheduler.state_dict()
                fs_writer = FileSystemWriter(scheduler_path)
                dcp.save(scheduler_state, storage_writer=fs_writer)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    @classmethod
    def from_checkpoint(
        cls,
        sae: SparseDictionary,
        checkpoint_path: str,
    ) -> "Trainer":
        """
        Load a complete checkpoint including model, optimizer, scheduler, and
        trainer state.

        Args:
            sae (SparseDictionary): The SAE model instance.
            checkpoint_path (str): Path where the checkpoint was saved (without extension).

        Returns:
            Trainer: A new trainer instance with loaded state.
        """
        # Load trainer state first to get the config
        checkpoint_dir = Path(checkpoint_path)
        trainer_path = checkpoint_dir / "trainer.pt"
        if os.path.exists(trainer_path):
            trainer_state = torch.load(trainer_path, map_location="cpu", weights_only=False)
            cfg = trainer_state.get("cfg")
            if cfg is None:
                raise ValueError("Checkpoint does not contain trainer config")

            # Create trainer instance with loaded config
            trainer = cls(cfg)
            trainer.cfg.from_pretrained_path = checkpoint_path

            # Restore trainer state variables
            trainer.cur_step = trainer_state["cur_step"]
            trainer.cur_tokens = trainer_state["cur_tokens"]
            trainer.total_training_steps = trainer_state["total_training_steps"]
            trainer.lr_warm_up_steps = trainer_state["lr_warm_up_steps"]
            trainer.lr_cool_down_steps = trainer_state["lr_cool_down_steps"]
            trainer.k_warmup_steps = trainer_state["k_warmup_steps"]
            trainer.k_cold_booting_steps = trainer_state["k_cold_booting_steps"]
            trainer.l1_coefficient_warmup_steps = trainer_state["l1_coefficient_warmup_steps"]
            trainer.checkpoint_thresholds = trainer_state["checkpoint_thresholds"]

            logger.info(f"Loaded trainer state from step {trainer.cur_step}")
        else:
            raise ValueError(f"Trainer checkpoint not found at {trainer_path}")

        trainer._initialize_optimizer(sae)
        assert trainer.optimizer is not None and trainer.scheduler is not None, (
            "Optimizer and scheduler should be already initialized"
        )

        # Load optimizer state
        if sae.device_mesh is None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            trainer.optimizer.load_state_dict(optimizer_state)
            logger.info("Loaded optimizer state")
        else:
            optimizer_path = checkpoint_dir / "optimizer.dcp"
            fs_reader = FileSystemReader(str(optimizer_path))
            optimizer_state = trainer.optimizer.state_dict()
            dcp.load(optimizer_state, storage_reader=fs_reader)
            trainer.optimizer.load_state_dict(optimizer_state)
            logger.info("Loaded optimizer state")
            logger.info(f"trainer.optimizer.state_dict(): {trainer.optimizer.state_dict()}")

        # Load scheduler state
        if sae.device_mesh is None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            trainer.scheduler.load_state_dict(scheduler_state)
            logger.info("Loaded scheduler state")
        else:
            scheduler_path = checkpoint_dir / "scheduler.dcp"
            fs_reader = FileSystemReader(str(scheduler_path))
            scheduler_state = trainer.scheduler.state_dict()
            dcp.load(scheduler_state, storage_reader=fs_reader)
            trainer.scheduler.load_state_dict(scheduler_state)
            logger.info("Loaded scheduler state")
            logger.info(f"trainer.scheduler.state_dict(): {trainer.scheduler.state_dict()}")

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return trainer

    @timer.time("initialize_trainer")
    def _initialize_trainer(
        self,
        sae: SparseDictionary,
        activation_stream: Iterable[dict[str, Tensor]],
        wandb_logger: Run | None = None,
    ):
        batch = next(iter(activation_stream))
        bs = batch["tokens"].numel()
        if batch["mask"].numel() != batch["mask"].sum():
            logger.warning(
                "We are training with batches of varying length. So we will not use as many as `self.cfg.total_training_tokens` for training as we assume each batch is full to estimate training steps. `details/n_training_tokens` is accurate in this case."
            )

        self.total_training_steps = self.cfg.total_training_tokens // bs

        def calculate_warmup_steps(warmup_steps: float | int) -> int:
            if isinstance(warmup_steps, float):
                assert 0 <= warmup_steps <= 1.0
                return int(warmup_steps * self.total_training_steps)
            return warmup_steps

        self.lr_warm_up_steps = calculate_warmup_steps(self.cfg.lr_warm_up_steps)
        self.lr_cool_down_steps = calculate_warmup_steps(self.cfg.lr_cool_down_steps)
        self.k_warmup_steps = calculate_warmup_steps(self.cfg.k_warmup_steps)
        self.k_cold_booting_steps = calculate_warmup_steps(self.cfg.k_cold_booting_steps)
        self.l1_coefficient_warmup_steps = calculate_warmup_steps(self.cfg.l1_coefficient_warmup_steps)
        if self.cfg.n_checkpoints > 0:
            if self.cfg.check_point_save_mode == "linear":
                self.checkpoint_thresholds = list(
                    range(0, self.total_training_steps, self.total_training_steps // self.cfg.n_checkpoints)
                )[1:]
            elif self.cfg.check_point_save_mode == "log":
                self.checkpoint_thresholds = [
                    math.ceil(2 ** (i / self.cfg.n_checkpoints * math.log2(self.total_training_steps))) * bs
                    for i in range(1, self.cfg.n_checkpoints)
                ]
        self.wandb_logger = wandb_logger

    @timer.time("initialize_optimizer")
    def _initialize_optimizer(self, sae: SparseDictionary):
        assert isinstance(self.cfg.lr, float)

        def _apply_lr(parameters: dict[str, Any]):
            assert isinstance(self.cfg.lr, float)
            if parameters["name"] == "jumprelu":
                return {**parameters, "lr": self.cfg.jumprelu_lr_factor * self.cfg.lr}
            return parameters

        params = [_apply_lr(parameters) for parameters in sae.get_parameters()]

        def _format_parameters(parameters: dict[str, Any]) -> str:
            param_info = f"{parameters['name']}:"
            for i, param in enumerate(parameters["params"]):
                param_info += f"\n    [{i}] shape={list(param.shape)}, dtype={param.dtype}"
                if param.requires_grad:
                    param_info += ", trainable"
                else:
                    param_info += ", frozen"
            if "lr" in parameters:
                param_info += f"\n    lr={parameters['lr']}"
            return param_info

        param_str = "\n".join([_format_parameters(p) for p in params])
        logger.info(f"\nParameter Groups: \n{param_str}\n")

        optim_cls = {
            "adam": Adam,
            "sparseadam": SparseAdam,
        }[self.cfg.optimizer_class]

        optimizer_kwargs = {
            "params": params,
            "lr": self.cfg.lr,
            "betas": self.cfg.betas,
        }

        # only adam optimizer supports foreach parameter
        if self.cfg.optimizer_class == "adam":
            optimizer_kwargs["foreach"] = self.cfg.optimizer_foreach

        optimizer = optim_cls(**optimizer_kwargs)

        scheduler = get_scheduler(
            scheduler_name=self.cfg.lr_scheduler_name,
            optimizer=optimizer,
            warm_up_steps=self.lr_warm_up_steps,
            cool_down_steps=self.lr_cool_down_steps,
            training_steps=self.total_training_steps,
            lr_end_ratio=self.cfg.lr_end_ratio,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _initialize_dead_statistics(self, sae: SparseDictionary) -> None:
        """Initialize the dead statistics tracking variables for auxk loss.

        Args:
            sae: The sparse autoencoder model to get the d_sae dimension from.
        """
        if sae.device_mesh is None:
            self.tokens_since_last_activation = torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=torch.long)
            self.is_dead = torch.zeros(sae.cfg.d_sae, device=sae.cfg.device, dtype=torch.bool)
        else:
            from llamascopium.utils.distributed import DimMap

            self.tokens_since_last_activation = torch.distributed.tensor.zeros(
                sae.cfg.d_sae,
                dtype=torch.long,
                device_mesh=sae.device_mesh,
                placements=DimMap({"model": 0}).placements(sae.device_mesh),
            )
            self.is_dead = torch.distributed.tensor.zeros(
                sae.cfg.d_sae,
                dtype=torch.bool,
                device_mesh=sae.device_mesh,
                placements=DimMap({"model": 0}).placements(sae.device_mesh),
            )

    @torch.no_grad()
    def update_dead_statistics(self, feature_acts: Tensor, mask: Tensor | None, specs: tuple[str, ...]) -> Tensor:
        """Update the dead latents tracking based on current feature activations.

        Args:
            feature_acts: Feature activations tensor of shape (batch, d_sae) or (batch, seq_len, d_sae)

        Returns:
            is_dead: Boolean tensor indicating which features are dead.
        """
        assert self.tokens_since_last_activation is not None, (
            "tokens_since_last_activation must be initialized before calling update_dead_statistics"
        )
        assert self.is_dead is not None, "is_dead must be initialized before calling update_dead_statistics"

        valid_tokens = mask.sum() if mask is not None else feature_acts[..., 0].numel()

        feature_acts_sum, _ = apply_token_mask(feature_acts, specs, mask, "sum")
        activated = feature_acts_sum.gt(0)

        self.tokens_since_last_activation = torch.where(
            activated,
            torch.zeros_like(self.tokens_since_last_activation),
            self.tokens_since_last_activation + valid_tokens,
        )
        self.is_dead = self.tokens_since_last_activation >= self.cfg.dead_threshold
        return self.is_dead

    @timer.time("training_step")
    def _training_step(
        self,
        sae: SparseDictionary,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if "topk" in sae.cfg.act_fn and self.k_warmup_steps > 0:
            assert self.cfg.initial_k is not None, "initial_k must be provided"
            assert self.cfg.initial_k >= sae.cfg.top_k, "initial_k must be greater than or equal to top_k"
            if self.cur_step < self.k_cold_booting_steps:
                sae.set_current_k(int(self.cfg.initial_k))
            else:
                warmup_progress = (self.cur_step - self.k_cold_booting_steps) / self.k_warmup_steps
                warmup_progress = min(1.0, warmup_progress)

                if self.cfg.k_schedule_type == "exponential":
                    exp_factor = self.cfg.k_exponential_factor
                    decay_factor = (1.0 - math.exp(-exp_factor * warmup_progress)) / (1.0 - math.exp(-exp_factor))
                    current_k = self.cfg.initial_k - (self.cfg.initial_k - sae.cfg.top_k) * decay_factor
                elif self.cfg.k_schedule_type == "linear":
                    current_k = self.cfg.initial_k + (sae.cfg.top_k - self.cfg.initial_k) * warmup_progress
                else:
                    current_k = self.cfg.initial_k + (sae.cfg.top_k - self.cfg.initial_k) * warmup_progress

                sae.set_current_k(
                    max(
                        sae.cfg.top_k,
                        math.ceil(current_k),
                    )
                )

        l1_coefficient = (
            min(1.0, self.cur_step / self.l1_coefficient_warmup_steps) * self.cfg.l1_coefficient
            if self.cfg.l1_coefficient is not None
            else 1.0
        )

        lp_coefficient = self.cfg.lp_coefficient if self.cfg.lp_coefficient is not None else 0.0

        auxk_coefficient = self.cfg.auxk_coefficient if self.cfg.auxk_coefficient is not None else 0.0

        ctx = sae.compute_loss(
            batch,
            sparsity_loss_type=self.cfg.sparsity_loss_type,
            tanh_stretch_coefficient=self.cfg.tanh_stretch_coefficient,
            p=self.cfg.p,
            target_l0=self.cfg.target_l0,
            return_aux_data=True,
            l1_coefficient=l1_coefficient,
            lp_coefficient=lp_coefficient,
            auxk_coefficient=auxk_coefficient,
            frequency_scale=self.cfg.frequency_scale,
            k_aux=self.cfg.k_aux,
            update_dead_statistics=self.update_dead_statistics if auxk_coefficient > 0.0 else None,
        )
        return ctx

    @torch.no_grad()
    @timer.time("log")
    def _log(self, sae: SparseDictionary, ctx: dict[str, Any]):
        """Log training metrics and sparsity statistics.

        Delegates model-specific logging to the model's methods.
        """
        assert self.optimizer is not None, "Optimizer must be initialized"

        # Initialize metrics on first call
        if not self.metrics:
            self.metrics = [
                FrequencyMetric(sae),
                LossMetric(sae),
                MeanFeatureActMetric(sae),
                ExplainedVarianceMetric(sae),
                L0Metric(sae),
                L2NormErrorMetric(sae),
                ModelSpecificMetric(sae),
                GradientNormMetric(sae),
            ]

        for metric in self.metrics:
            ctx = {**ctx, **metric.update(ctx)}

        if (self.cur_step + 1) % self.cfg.log_frequency == 0:
            metrics = {}

            for metric in self.metrics:
                metrics.update(metric.compute())

            metrics.update(
                {
                    "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
                    "details/n_training_tokens": self.cur_tokens,
                    "details/l1_coefficient": ctx.get("l1_coefficient"),
                    "details/lp_coefficient": ctx.get("lp_coefficient"),
                    "details/auxk_coefficient": ctx.get("auxk_coefficient"),
                }
            )

            metrics.update(sae.log_statistics())

            if is_primary_rank(sae.device_mesh):
                log_metrics(logger.logger, metrics, step=self.cur_step + 1, title="Training Metrics")

            if timer.enabled:
                logger.info(f"\nTimer Summary:\n{timer.summary()}\n")

            if self.wandb_logger is not None:
                self.wandb_logger.log(metrics, step=self.cur_step + 1)

    @timer.time("save_checkpoint")
    def _maybe_save_sae_checkpoint(self, sae: SparseDictionary):
        if len(self.checkpoint_thresholds) > 0 and self.cur_step >= self.checkpoint_thresholds[0]:
            suffix = "safetensors" if sae.device_mesh is None else "dcp"
            path = os.path.join(
                self.cfg.exp_result_path,
                "checkpoints",
                f"{self.cur_step}.{suffix}",
            )
            sae.save_checkpoint(path)
            self.checkpoint_thresholds.pop(0)

    def fit(
        self,
        sae: SparseDictionary,
        activation_stream: Iterable[dict[str, Tensor]],
        eval_fn: Callable[[SparseDictionary], None] | None = None,
        wandb_logger: Run | None = None,
    ) -> bool | None:
        # Reset timer at the start of training
        timer.reset()

        if self.cfg.from_pretrained_path is None:
            logger.info("Initializing trainer and optimizer")
            self._initialize_trainer(sae, activation_stream, wandb_logger)
            self._initialize_optimizer(sae)

        # Initialize dead statistics for auxk loss
        if self.cfg.auxk_coefficient is not None and self.cfg.auxk_coefficient > 0.0:
            self._initialize_dead_statistics(sae)

        assert self.optimizer is not None and self.scheduler is not None, (
            "Optimizer and scheduler should be already initialized"
        )

        proc_bar = tqdm(total=self.total_training_steps, smoothing=0.001, disable=not is_primary_rank(sae.device_mesh))
        proc_bar.update(self.cur_step)

        try:
            activation_stream = iter(activation_stream)
            batch = next(activation_stream)
            while True:
                with timer.time("training_iteration"):
                    proc_bar.update(1)

                    batch = sae.normalize_activations(batch)

                    sae.train()

                    with torch.autocast(device_type=sae.cfg.device, dtype=self.cfg.amp_dtype):
                        ctx = self._training_step(sae, batch)

                    # Get GPU memory usage if available
                    mem_info = ""
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                        mem_info = f", mem: {mem_allocated:.2f}/{mem_reserved:.2f}GB"

                    proc_bar.set_description(
                        f"loss: {item(ctx['loss']):.2f}, lr: {self.optimizer.param_groups[0]['lr']:.2e}{mem_info}"
                    )

                    with timer.time("backward"):
                        ctx["loss"].backward()

                    with timer.time("clip_grad_norm"):
                        # exclude the grad of the jumprelu threshold
                        ctx["grad_norm_before_clipping"] = clip_grad_norm(
                            [
                                param
                                for name, param in sae.named_parameters()
                                if param.grad is not None and "log_jumprelu_threshold" not in name
                            ],
                            max_norm=self.cfg.clip_grad_norm if self.cfg.clip_grad_norm > 0 else math.inf,
                        )

                    if not self.cfg.skip_metrics_calculation:
                        with torch.autocast(device_type=sae.cfg.device, dtype=self.cfg.amp_dtype):
                            self._log(sae, ctx)

                    with timer.time("optimizer_step"):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if eval_fn is not None and (self.cur_step + 1) % self.cfg.eval_frequency == 0:
                        with timer.time("evaluation"):
                            eval_fn(sae)

                    with timer.time("scheduler_step"):
                        self.scheduler.step()
                    self.cur_step += 1
                    self.cur_tokens += (
                        batch["tokens"].numel() if batch.get("mask") is None else int(item(batch["mask"].sum()))
                    )

                    self._maybe_save_sae_checkpoint(sae)
                    if self.cur_step >= self.total_training_steps:
                        break
                    with timer.time("refresh_batch"):
                        del batch
                        batch = next(activation_stream)
        except StopIteration:
            logger.info("the current stream has ended")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e
