"""Module for evaluating SAE models."""

import os
from typing import Optional

import torch
import wandb
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from llamascopium.activation.factory import ActivationFactory, ActivationFactoryConfig
from llamascopium.backend.language_model import LanguageModelConfig
from llamascopium.evaluator import EvalConfig, Evaluator
from llamascopium.models.crosscoder import Crosscoder
from llamascopium.models.sparse_dictionary import SparseDictionary
from llamascopium.trainer import WandbConfig
from llamascopium.utils.distributed import mesh_rank
from llamascopium.utils.logging import get_distributed_logger, setup_logging

from .utils import PretrainedSAE

logger = get_distributed_logger("runners.eval")


class EvaluateSAESettings(BaseSettings):
    """Settings for evaluating a Sparse Autoencoder."""

    sae: PretrainedSAE
    """Path to a pretrained SAE model"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    eval: EvalConfig
    """Configuration for evaluation"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    fold_activation_scale: bool = False
    """Whether to fold the activation scale."""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


def evaluate_sae(settings: EvaluateSAESettings) -> None:
    """Evaluate a SAE model.

    Args:
        settings: Configuration settings for SAE evaluation
    """
    # Set up logging
    setup_logging(level="INFO")

    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
            mesh_shape=(settings.model_parallel_size,),
            mesh_dim_names=("model",),
        )
        if settings.model_parallel_size > 1
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Loading SAE model")

    sae = SparseDictionary.from_pretrained(
        settings.sae.pretrained_name_or_path,
        device_mesh=device_mesh,
        fold_activation_scale=settings.fold_activation_scale,
        device=settings.sae.device,
        dtype=settings.sae.dtype,
        strict_loading=settings.sae.strict_loading,
    )

    logger.info(f"SAE model loaded: {type(sae).__name__}")

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
        )
        if settings.wandb is not None and (device_mesh is None or mesh_rank(device_mesh) == 0)
        else None
    )

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    logger.info("Processing activations for evaluation")
    activations = activation_factory.process()
    evaluator = Evaluator(settings.eval)
    evaluator.evaluate(sae, activations, wandb_logger)
    logger.info("Evaluation completed")


class EvaluateCrosscoderSettings(BaseSettings):
    """Settings for evaluating a Crosscoder model."""

    sae: PretrainedSAE
    """Path to a pretrained Crosscoder model"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    activation_factories: list[ActivationFactoryConfig]
    """Configuration for generating activations"""

    eval: EvalConfig
    """Configuration for evaluation"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""


@torch.no_grad()
def evaluate_crosscoder(settings: EvaluateCrosscoderSettings) -> None:
    """Evaluate a Crosscoder model. The key difference to evaluate_sae is that the activation factories are a list of ActivationFactoryConfig, one for each head; and the evaluating contains a device mesh transformation from head parallelism to model (feature) parallelism.

    Args:
        settings: Configuration settings for Crosscoder evaluation
    """
    # Set up logging
    setup_logging(level="INFO")

    parallel_size = len(settings.activation_factories)

    logger.info(f"Analyzing Crosscoder with {parallel_size} parallel size")

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
        mesh_shape=(parallel_size,),
        mesh_dim_names=("head",),
    )

    logger.info("Device meshes initialized for Crosscoder analysis")

    logger.info("Setting up activation factory for Crosscoder head")
    activation_factory = ActivationFactory(settings.activation_factories[device_mesh.get_local_rank("head")])

    logger.info("Loading Crosscoder model")
    sae = Crosscoder.from_pretrained(
        settings.sae.pretrained_name_or_path,
        device_mesh=device_mesh,
        device=settings.sae.device,
        dtype=settings.sae.dtype,
        fold_activation_scale=settings.sae.fold_activation_scale,
        strict_loading=settings.sae.strict_loading,
    )

    assert len(settings.activation_factories) * len(settings.activation_factories[0].hook_points) == sae.cfg.n_heads, (
        "Total number of hook points must match the number of heads in the Crosscoder"
    )

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
        )
        if settings.wandb is not None and (device_mesh is None or mesh_rank(device_mesh) == 0)
        else None
    )

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    logger.info("Processing activations for Crosscoder evaluation")
    activations = activation_factory.process()
    evaluator = Evaluator(settings.eval)
    evaluator.evaluate(sae, activations, wandb_logger)

    logger.info("Crosscoder evaluation completed successfully")
