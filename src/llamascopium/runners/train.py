"""Module for sweeping SAE experiments."""

import os
from pathlib import Path
from typing import Optional

import torch
import wandb
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from torch.distributed.device_mesh import init_device_mesh

from llamascopium.activation.factory import ActivationFactory, ActivationFactoryConfig
from llamascopium.backend.language_model import LanguageModelConfig
from llamascopium.config import DatasetConfig
from llamascopium.database import MongoClient, MongoDBConfig
from llamascopium.initializer import Initializer, InitializerConfig
from llamascopium.models.clt import CLTConfig
from llamascopium.models.crosscoder import CrosscoderConfig
from llamascopium.models.lorsa import LorsaConfig
from llamascopium.models.molt import MOLTConfig
from llamascopium.models.sparse_dictionary import SparseDictionary, SparseDictionaryConfig
from llamascopium.resource_loaders import load_dataset, load_model
from llamascopium.trainer import Trainer, TrainerConfig, WandbConfig
from llamascopium.utils.distributed import is_primary_rank, mesh_rank
from llamascopium.utils.logging import get_distributed_logger, setup_logging

from .utils import PretrainedSAE, load_config

logger = get_distributed_logger("runners.train")


class TrainSAESettings(BaseSettings):
    """Settings for training a Sparse Autoencoder (SAE)."""

    sae: SparseDictionaryConfig | PretrainedSAE
    """Configuration for the SAE model architecture and parameters, or the path to a pretrained SAE."""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    initializer: InitializerConfig | None = None
    """Configuration for model initialization. Should be None for a pretrained SAE."""

    trainer: TrainerConfig
    """Configuration for training process"""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    eval: bool = False
    """Whether to run in evaluation mode"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the model/tokenizer to load."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    fold_activation_scale: bool = True
    """Whether to fold the activation scale."""


def train_sae(settings: TrainSAESettings) -> None:
    """Train a SAE model.

    Args:
        settings: Configuration settings for SAE training
    """
    # Set up logging
    setup_logging(level="INFO")

    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
            mesh_shape=(settings.data_parallel_size, settings.model_parallel_size),
            mesh_dim_names=("data", "model"),
        )
        if settings.model_parallel_size > 1 or settings.data_parallel_size > 1
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

    # Load configurations
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    dataset_cfgs = (
        {
            dataset_name: load_config(
                config=dataset_cfg,
                name=dataset_name,
                mongo_client=mongo_client,
                config_type="dataset",
            )
            for dataset_name, dataset_cfg in settings.datasets.items()
        }
        if settings.datasets is not None
        else None
    )

    # Load model and datasets
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    logger.info("Initializing SAE")

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
            resume=settings.wandb.wandb_resume,
            id=settings.wandb.wandb_run_id,
        )
        if settings.wandb is not None and (device_mesh is None or mesh_rank(device_mesh) == 0)
        else None
    )

    assert settings.initializer is None or not isinstance(settings.initializer, str), (
        "Cannot use an initializer for a pretrained SAE"
    )
    if isinstance(settings.sae, PretrainedSAE):
        sae = SparseDictionary.from_pretrained(
            settings.sae.pretrained_name_or_path,
            device_mesh=device_mesh,
            fold_activation_scale=False,
            strict_loading=settings.sae.strict_loading,
            device=settings.sae.device,
            dtype=settings.sae.dtype,
        )
    elif settings.initializer is not None:
        initializer = Initializer(settings.initializer)
        sae = initializer.initialize_sae_from_config(
            settings.sae,
            activation_stream=activations_stream,
            device_mesh=device_mesh,
            wandb_logger=wandb_logger,
            model=model,
        )
    else:
        sae = SparseDictionary.from_config(settings.sae, device_mesh=device_mesh)

    if settings.trainer.from_pretrained_path is not None:
        trainer = Trainer.from_checkpoint(
            sae,
            settings.trainer.from_pretrained_path,
            total_training_tokens=settings.trainer.total_training_tokens,
            config_overrides=settings.trainer,
        )
        trainer.wandb_logger = wandb_logger
    else:
        trainer = Trainer(settings.trainer)

    logger.info(f"SAE initialized: {type(sae).__name__}")

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting training")

    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    end_of_stream = trainer.fit(
        sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger
    )
    logger.info("Training completed, saving model")
    if end_of_stream:
        trainer.save_checkpoint(
            sae=sae,
            checkpoint_path=settings.trainer.exp_result_path,
        )
    else:
        sae.save_pretrained(
            save_path=settings.trainer.exp_result_path,
            fold_activation_scale=settings.fold_activation_scale,
        )
        if getattr(settings.trainer, "save_full_checkpoint_on_finish", False):
            trainer.save_checkpoint(
                sae=sae,
                checkpoint_path=settings.trainer.exp_result_path,
            )
        if is_primary_rank(device_mesh) and mongo_client is not None:
            assert settings.sae_name is not None and settings.sae_series is not None, (
                "sae_name and sae_series must be provided when saving to MongoDB"
            )
            mongo_client.create_sae(
                name=settings.sae_name,
                series=settings.sae_series,
                path=str(Path(settings.trainer.exp_result_path).absolute()),
                cfg=sae.cfg,
            )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("SAE training completed successfully")


class TrainCrosscoderSettings(BaseSettings):
    """Settings for training a Crosscoder. The main difference to TrainSAESettings is that the activation factory is a list of ActivationFactoryConfig, one for each head."""

    sae: CrosscoderConfig | PretrainedSAE
    """Configuration for the Crosscoder model architecture and parameters, or the path to a pretrained Crosscoder."""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    initializer: InitializerConfig | None = None
    """Configuration for model initialization"""

    trainer: TrainerConfig
    """Configuration for training process"""

    activation_factories: list[ActivationFactoryConfig]
    """Configuration for generating activations"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    eval: bool = False
    """Whether to run in evaluation mode"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the model/tokenizer to load."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    fold_activation_scale: bool = True
    """Whether to fold the activation scale."""


def train_crosscoder(settings: TrainCrosscoderSettings) -> None:
    """Train a Crosscoder.

    Args:
        settings: Configuration settings for SAE training
    """
    # Set up logging
    setup_logging(level="INFO")

    assert isinstance(settings.sae, CrosscoderConfig), "CrosscoderConfig is required for training a Crosscoder"
    assert all(
        len(activation_factory.hook_points) == len(settings.activation_factories[0].hook_points)
        for activation_factory in settings.activation_factories
    ), "Number of hook points of activation factories must be the same"
    assert (
        len(settings.activation_factories) * len(settings.activation_factories[0].hook_points) == settings.sae.n_heads
    ), "Total number of hook points must match the number of heads in the Crosscoder"
    head_parallel_size = len(settings.activation_factories)

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
        mesh_shape=(settings.data_parallel_size, head_parallel_size, settings.model_parallel_size),
        mesh_dim_names=("data", "head", "model"),
    )

    logger.info(
        f"Device mesh initialized with {settings.sae.n_heads} heads, {head_parallel_size} head parallel size, {settings.data_parallel_size} data parallel size, {settings.model_parallel_size} model parallel size"
    )

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

    # Load configurations
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    dataset_cfgs = (
        {
            dataset_name: load_config(
                config=dataset_cfg,
                name=dataset_name,
                mongo_client=mongo_client,
                config_type="dataset",
            )
            for dataset_name, dataset_cfg in settings.datasets.items()
        }
        if settings.datasets is not None
        else None
    )

    # Load model and datasets
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory_mesh = device_mesh[
        "data", "model"
    ]  # Remove the head dimension, since each activation factory should only be responsible for a subset of the heads.

    logger.info("Setting up activation factory for Crosscoder")
    activation_factory = ActivationFactory(
        settings.activation_factories[device_mesh.get_local_rank("head")], device_mesh=activation_factory_mesh
    )

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
            resume=settings.wandb.wandb_resume,
            id=settings.wandb.wandb_run_id,
        )
        if settings.wandb is not None and (device_mesh is None or mesh_rank(device_mesh) == 0)
        else None
    )

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    logger.info("Initializing Crosscoder")
    assert settings.initializer is None or not isinstance(settings.initializer, str), (
        "Cannot use an initializer for a pretrained Crosscoder"
    )
    if isinstance(settings.sae, PretrainedSAE):
        sae = SparseDictionary.from_pretrained(
            settings.sae.pretrained_name_or_path,
            device_mesh=device_mesh,
            fold_activation_scale=False,
            strict_loading=settings.sae.strict_loading,
            device=settings.sae.device,
            dtype=settings.sae.dtype,
        )
    elif settings.initializer is not None:
        initializer = Initializer(settings.initializer)
        sae = initializer.initialize_sae_from_config(
            settings.sae,
            activation_stream=activations_stream,
            device_mesh=device_mesh,
            wandb_logger=wandb_logger,
            model=model,
        )
    else:
        sae = SparseDictionary.from_config(settings.sae, device_mesh=device_mesh)

    logger.info("Crosscoder initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting Crosscoder training")
    if settings.trainer.from_pretrained_path is not None:
        trainer = Trainer.from_checkpoint(
            sae,
            settings.trainer.from_pretrained_path,
            total_training_tokens=settings.trainer.total_training_tokens,
            config_overrides=settings.trainer,
        )
        trainer.wandb_logger = wandb_logger
    else:
        trainer = Trainer(settings.trainer)

    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    end_of_stream = trainer.fit(
        sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger
    )

    logger.info("Training completed, saving Crosscoder")
    if end_of_stream:
        trainer.save_checkpoint(
            sae=sae,
            checkpoint_path=settings.trainer.exp_result_path,
        )
    else:
        sae.save_pretrained(
            save_path=settings.trainer.exp_result_path,
            fold_activation_scale=settings.fold_activation_scale,
        )
        if getattr(settings.trainer, "save_full_checkpoint_on_finish", False):
            trainer.save_checkpoint(
                sae=sae,
                checkpoint_path=settings.trainer.exp_result_path,
            )
        if is_primary_rank(device_mesh) and mongo_client is not None:
            assert settings.sae_name is not None and settings.sae_series is not None, (
                "sae_name and sae_series must be provided when saving to MongoDB"
            )
            mongo_client.create_sae(
                name=settings.sae_name,
                series=settings.sae_series,
                path=str(Path(settings.trainer.exp_result_path).absolute()),
                cfg=settings.sae,
            )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("Crosscoder training completed successfully")


class TrainCLTSettings(BaseSettings):
    """Settings for training a Cross Layer Transcoder (CLT). CLT works with multiple layers and their corresponding hook points."""

    sae: CLTConfig | PretrainedSAE
    """Configuration for the CLT model architecture and parameters, or the path to a pretrained CLT."""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    initializer: InitializerConfig | None = None
    """Configuration for model initialization"""

    trainer: TrainerConfig
    """Configuration for training process"""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    eval: bool = False
    """Whether to run in evaluation mode"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the tokenizer to load. CLT requires a tokenizer to get the modality indices."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    fold_activation_scale: bool = True
    """Whether to fold the activation scale."""


def train_clt(settings: TrainCLTSettings) -> None:
    """Train a Cross Layer Transcoder (CLT) model.

    Args:
        settings: Configuration settings for CLT training
    """
    # Set up logging
    setup_logging(level="INFO")

    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
            mesh_shape=(settings.data_parallel_size, settings.model_parallel_size),
            mesh_dim_names=("data", "model"),
        )
        if settings.model_parallel_size > 1 or settings.data_parallel_size > 1
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

    # Load configurations
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    dataset_cfgs = (
        {
            dataset_name: load_config(
                config=dataset_cfg,
                name=dataset_name,
                mongo_client=mongo_client,
                config_type="dataset",
            )
            for dataset_name, dataset_cfg in settings.datasets.items()
        }
        if settings.datasets is not None
        else None
    )

    # Load model and datasets
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
            resume=settings.wandb.wandb_resume,
            id=settings.wandb.wandb_run_id,
        )
        if settings.wandb is not None and (device_mesh is None or mesh_rank(device_mesh) == 0)
        else None
    )

    logger.info("Initializing CLT")
    assert settings.initializer is None or not isinstance(settings.initializer, str), (
        "Cannot use an initializer for a pretrained CLT"
    )
    if isinstance(settings.sae, PretrainedSAE):
        sae = SparseDictionary.from_pretrained(
            settings.sae.pretrained_name_or_path,
            device_mesh=device_mesh,
            fold_activation_scale=False,
            strict_loading=settings.sae.strict_loading,
            device=settings.sae.device,
            dtype=settings.sae.dtype,
        )
    elif settings.initializer is not None:
        initializer = Initializer(settings.initializer)
        sae = initializer.initialize_sae_from_config(
            settings.sae,
            activation_stream=activations_stream,
            device_mesh=device_mesh,
            wandb_logger=wandb_logger,
            model=model,
        )
    else:
        sae = SparseDictionary.from_config(settings.sae, device_mesh=device_mesh)

    n_params = sum(p.numel() for p in sae.parameters())
    logger.info(f"CLT initialized with {n_params / 1e9:.2f}B parameters")

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting CLT training")
    if settings.trainer.from_pretrained_path is not None:
        trainer = Trainer.from_checkpoint(
            sae,
            settings.trainer.from_pretrained_path,
            total_training_tokens=settings.trainer.total_training_tokens,
            config_overrides=settings.trainer,
        )
        trainer.wandb_logger = wandb_logger
    else:
        trainer = Trainer(settings.trainer)
    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    end_of_stream = trainer.fit(
        sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger
    )

    logger.info("Training completed, saving CLT model")
    if end_of_stream:
        trainer.save_checkpoint(
            sae=sae,
            checkpoint_path=settings.trainer.exp_result_path,
        )
    else:
        sae.save_pretrained(
            save_path=settings.trainer.exp_result_path,
            fold_activation_scale=settings.fold_activation_scale,
        )
        if getattr(settings.trainer, "save_full_checkpoint_on_finish", False):
            trainer.save_checkpoint(
                sae=sae,
                checkpoint_path=settings.trainer.exp_result_path,
            )
        if is_primary_rank(device_mesh) and mongo_client is not None:
            assert settings.sae_name is not None and settings.sae_series is not None, (
                "sae_name and sae_series must be provided when saving to MongoDB"
            )
            mongo_client.create_sae(
                name=settings.sae_name,
                series=settings.sae_series,
                path=str(Path(settings.trainer.exp_result_path).absolute()),
                cfg=sae.cfg,
            )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("CLT training completed successfully")


class TrainLorsaSettings(BaseSettings):
    """Settings for training a Lorsa (Low-Rank Sparse Autoencoder) model."""

    sae: LorsaConfig | PretrainedSAE
    """Configuration for the Lorsa model architecture and parameters, or the path to a pretrained Lorsa."""

    sae_name: str
    """Name of the Lorsa model. Use as identifier for the Lorsa model in the database."""

    sae_series: str
    """Series of the Lorsa model. Use as identifier for the Lorsa model in the database."""

    initializer: InitializerConfig | None = None
    """Configuration for model initialization"""

    trainer: TrainerConfig
    """Configuration for training process"""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    eval: bool = False
    """Whether to run in evaluation mode"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the model/tokenizer to load."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    fold_activation_scale: bool = True
    """Whether to fold the activation scale."""


def train_lorsa(settings: TrainLorsaSettings) -> None:
    """Train a LORSA (Low-Rank Sparse Autoencoder) model.

    Args:
        settings: Configuration settings for LORSA training
    """
    # Set up logging
    setup_logging(level="INFO")

    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
            mesh_shape=(settings.data_parallel_size, settings.model_parallel_size),
            mesh_dim_names=("data", "model"),
        )
        if settings.model_parallel_size > 1 or settings.data_parallel_size > 1
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

    # Load configurations
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    dataset_cfgs = (
        {
            dataset_name: load_config(
                config=dataset_cfg,
                name=dataset_name,
                mongo_client=mongo_client,
                config_type="dataset",
            )
            for dataset_name, dataset_cfg in settings.datasets.items()
        }
        if settings.datasets is not None
        else None
    )

    # Load model and datasets
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    logger.info("Initializing lorsa")

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
            resume=settings.wandb.wandb_resume,
            id=settings.wandb.wandb_run_id,
        )
        if settings.wandb is not None and (device_mesh is None or device_mesh.get_rank() == 0)
        else None
    )

    assert settings.initializer is None or not isinstance(settings.initializer, str), (
        "Cannot use an initializer for a pretrained Lorsa"
    )
    if isinstance(settings.sae, PretrainedSAE):
        sae = SparseDictionary.from_pretrained(
            settings.sae.pretrained_name_or_path,
            device_mesh=device_mesh,
            fold_activation_scale=False,
            strict_loading=settings.sae.strict_loading,
            device=settings.sae.device,
            dtype=settings.sae.dtype,
        )
    elif settings.initializer is not None:
        initializer = Initializer(settings.initializer)
        sae = initializer.initialize_sae_from_config(
            settings.sae,
            activation_stream=activations_stream,
            device_mesh=device_mesh,
            wandb_logger=wandb_logger,
            model=model,
        )
    else:
        sae = SparseDictionary.from_config(settings.sae, device_mesh=device_mesh)

    n_params = sum(p.numel() for p in sae.parameters())
    logger.info(f"lorsa initialized with {n_params / 1e9:.2f}B parameters")

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting LORSA training")
    if settings.trainer.from_pretrained_path is not None:
        trainer = Trainer.from_checkpoint(
            sae,
            settings.trainer.from_pretrained_path,
            total_training_tokens=settings.trainer.total_training_tokens,
            config_overrides=settings.trainer,
        )
        trainer.wandb_logger = wandb_logger
    else:
        trainer = Trainer(settings.trainer)

    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    end_of_stream = trainer.fit(
        sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger
    )

    logger.info("Training completed, saving LORSA model")
    if end_of_stream:
        trainer.save_checkpoint(
            sae=sae,
            checkpoint_path=settings.trainer.exp_result_path,
        )
    else:
        sae.save_pretrained(
            save_path=settings.trainer.exp_result_path,
            fold_activation_scale=settings.fold_activation_scale,
        )
        if getattr(settings.trainer, "save_full_checkpoint_on_finish", False):
            trainer.save_checkpoint(
                sae=sae,
                checkpoint_path=settings.trainer.exp_result_path,
            )
        if is_primary_rank(device_mesh) and mongo_client is not None:
            assert settings.sae_name is not None and settings.sae_series is not None, (
                "sae_name and sae_series must be provided when saving to MongoDB"
            )
            mongo_client.create_sae(
                name=settings.sae_name,
                series=settings.sae_series,
                path=str(Path(settings.trainer.exp_result_path).absolute()),
                cfg=sae.cfg,
            )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("LORSA training completed successfully")


class TrainMOLTSettings(BaseSettings):
    """Settings for training a Mixture of Linear Transforms (MOLT). MOLT is a more efficient alternative to transcoders that sparsely replaces MLP computation in transformers."""

    sae: MOLTConfig | PretrainedSAE
    """Configuration for the MOLT model architecture and parameters"""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    initializer: InitializerConfig | None = None
    """Configuration for model initialization. Should be None for a pretrained MOLT."""

    trainer: TrainerConfig
    """Configuration for training process"""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""

    eval: bool = False
    """Whether to run in evaluation mode"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the tokenizer to load. MOLT requires a tokenizer to get the modality indices."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    fold_activation_scale: bool = True
    """Whether to fold the activation scale."""


def train_molt(settings: TrainMOLTSettings) -> None:
    """Train a Mixture of Linear Transforms (MOLT) model.

    Args:
        settings: Configuration settings for MOLT training
    """
    # Set up logging
    setup_logging(level="INFO")

    device_mesh = (
        init_device_mesh(
            device_type=settings.device_type,
            mesh_shape=(settings.model_parallel_size, settings.data_parallel_size),  # TODO: check the order
            mesh_dim_names=("model", "data"),
        )
        if settings.model_parallel_size > 1 or settings.data_parallel_size > 1
        else None
    )

    logger.info(f"Device mesh initialized: {device_mesh}")

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None
    if mongo_client:
        logger.info("MongoDB client initialized")

    # Load configurations
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    dataset_cfgs = (
        {
            dataset_name: load_config(
                config=dataset_cfg,
                name=dataset_name,
                mongo_client=mongo_client,
                config_type="dataset",
            )
            for dataset_name, dataset_cfg in settings.datasets.items()
        }
        if settings.datasets is not None
        else None
    )

    # Load model and datasets
    logger.info("Loading model and datasets")
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    wandb_logger = (
        wandb.init(
            project=settings.wandb.wandb_project,
            config=settings.model_dump(),
            name=settings.wandb.exp_name,
            entity=settings.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
            resume=settings.wandb.wandb_resume,
            id=settings.wandb.wandb_run_id,
        )
        if settings.wandb is not None and (device_mesh is None or mesh_rank(device_mesh) == 0)
        else None
    )

    logger.info("Initializing MOLT")

    assert settings.initializer is None or not isinstance(settings.initializer, str), (
        "Cannot use an initializer for a pretrained MOLT"
    )
    if isinstance(settings.sae, PretrainedSAE):
        sae = SparseDictionary.from_pretrained(
            settings.sae.pretrained_name_or_path,
            device_mesh=device_mesh,
            fold_activation_scale=False,
            strict_loading=settings.sae.strict_loading,
            device=settings.sae.device,
            dtype=settings.sae.dtype,
        )
    elif settings.initializer is not None:
        initializer = Initializer(settings.initializer)
        sae = initializer.initialize_sae_from_config(
            settings.sae,
            activation_stream=activations_stream,
            device_mesh=device_mesh,
            wandb_logger=wandb_logger,
            model=model,
        )
    else:
        sae = SparseDictionary.from_config(settings.sae, device_mesh=device_mesh)

    logger.info(f"MOLT initialized: {type(sae).__name__}")

    if wandb_logger is not None:
        logger.info("WandB logger initialized")

    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting MOLT training")
    if settings.trainer.from_pretrained_path is not None:
        trainer = Trainer.from_checkpoint(
            sae,
            settings.trainer.from_pretrained_path,
            total_training_tokens=settings.trainer.total_training_tokens,
            config_overrides=settings.trainer,
        )
        trainer.wandb_logger = wandb_logger
    else:
        trainer = Trainer(settings.trainer)

    sae.cfg.save_hyperparameters(settings.trainer.exp_result_path)
    end_of_stream = trainer.fit(
        sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger
    )

    logger.info("Training completed, saving MOLT model")
    if end_of_stream:
        trainer.save_checkpoint(
            sae=sae,
            checkpoint_path=settings.trainer.exp_result_path,
        )
    else:
        sae.save_pretrained(
            save_path=settings.trainer.exp_result_path,
            fold_activation_scale=settings.fold_activation_scale,
        )
        if getattr(settings.trainer, "save_full_checkpoint_on_finish", False):
            trainer.save_checkpoint(
                sae=sae,
                checkpoint_path=settings.trainer.exp_result_path,
            )
        if is_primary_rank(device_mesh) and mongo_client is not None:
            assert settings.sae_name is not None and settings.sae_series is not None, (
                "sae_name and sae_series must be provided when saving to MongoDB"
            )
            mongo_client.create_sae(
                name=settings.sae_name,
                series=settings.sae_series,
                path=str(Path(settings.trainer.exp_result_path).absolute()),
                cfg=sae.cfg,
            )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed")

    logger.info("MOLT training completed successfully")


class SweepingItem(BaseModel):
    """A single item in a sweeping configuration."""

    sae: SparseDictionaryConfig | PretrainedSAE
    """Configuration for the SAE model architecture and parameters, or the path to a pretrained SAE."""

    sae_name: str
    """Name of the SAE model. Use as identifier for the SAE model in the database."""

    sae_series: str
    """Series of the SAE model. Use as identifier for the SAE model in the database."""

    initializer: InitializerConfig | None = None
    """Configuration for model initialization. Should be None for a pretrained SAE."""

    trainer: TrainerConfig
    """Configuration for training process"""

    wandb: Optional[WandbConfig] = None
    """Configuration for Weights & Biases logging"""


class SweepSAESettings(BaseSettings):
    """Settings for sweeping a Sparse Autoencoder (SAE)."""

    items: list[SweepingItem]
    """List of sweeping items"""

    activation_factory: ActivationFactoryConfig
    """Configuration for generating activations"""

    eval: bool = False
    """Whether to run in evaluation mode"""

    data_parallel_size: int = 1
    """Size of data parallel mesh"""

    model_parallel_size: int = 1
    """Size of model parallel (tensor parallel) mesh"""

    mongo: Optional[MongoDBConfig] = None
    """Configuration for MongoDB"""

    model: Optional[LanguageModelConfig] = None
    """Configuration for the language model. Required if using dataset sources."""

    model_name: Optional[str] = None
    """Name of the tokenizer to load."""

    datasets: Optional[dict[str, Optional[DatasetConfig]]] = None
    """Name to dataset config mapping. Required if using dataset sources."""

    device_type: str = "cuda"
    """Device type to use for distributed training ('cuda' or 'cpu')"""

    fold_activation_scale: bool = True
    """Whether to fold the activation scale."""


def sweep_sae(settings: SweepSAESettings) -> None:
    """Sweep experiments for training SAE models.

    Args:
        settings: Configuration settings for SAE sweeping
    """
    # Set up logging
    setup_logging(level="INFO")

    n_sweeps = len(settings.items)

    device_mesh = init_device_mesh(
        device_type=settings.device_type,
        mesh_shape=(n_sweeps, settings.data_parallel_size, settings.model_parallel_size),
        mesh_dim_names=("sweep", "data", "model"),
    )

    logger.info(f"Device mesh initialized for sweep with {n_sweeps} configurations")

    mongo_client = MongoClient(settings.mongo) if settings.mongo is not None else None

    logger.info("Loading configurations on rank 0")
    # Load configurations
    model_cfg = load_config(
        config=settings.model,
        name=settings.model_name,
        mongo_client=mongo_client,
        config_type="model",
        required=False,
    )

    dataset_cfgs = (
        {
            dataset_name: load_config(
                config=dataset_cfg,
                name=dataset_name,
                mongo_client=mongo_client,
                config_type="dataset",
            )
            for dataset_name, dataset_cfg in settings.datasets.items()
        }
        if settings.datasets is not None
        else None
    )

    # Load model and datasets
    model = load_model(model_cfg) if model_cfg is not None else None
    datasets = (
        {
            dataset_name: load_dataset(dataset_cfg, device_mesh=device_mesh)
            for dataset_name, dataset_cfg in dataset_cfgs.items()
        }
        if dataset_cfgs is not None
        else None
    )

    activation_factory = ActivationFactory(settings.activation_factory, device_mesh=device_mesh)

    logger.info("Processing activations stream")
    activations_stream = activation_factory.process(
        model=model,
        model_name=settings.model_name,
        datasets=datasets,
    )

    sae_device_mesh = device_mesh["data", "model"]
    logger.info(f"Created 2D sub-mesh for SAE: {sae_device_mesh}")

    item = settings.items[device_mesh.get_local_rank("sweep")]
    logger.info(f"Processing sweep item: {item.sae_name}/{item.sae_series}")

    def convert_activations_to_2d_mesh(stream_3d, mesh_2d):
        from torch.distributed.tensor import DTensor

        for batch in stream_3d:
            converted_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    assert isinstance(value, DTensor), "value must be a DTensor"
                    local_tensor = value.to_local()
                    from llamascopium.utils.distributed import DimMap

                    converted_value = DTensor.from_local(
                        local_tensor,
                        device_mesh=mesh_2d,
                        placements=DimMap({"data": 0}).placements(mesh_2d),
                    )
                    converted_batch[key] = converted_value
                else:
                    converted_batch[key] = value
            yield converted_batch

    activations_stream = convert_activations_to_2d_mesh(activations_stream, sae_device_mesh)

    logger.info("Initializing SAE on 2D sub-mesh")

    assert item.initializer is None or not isinstance(item.initializer, str), (
        "Cannot use an initializer for a pretrained SAE"
    )
    if isinstance(item.sae, PretrainedSAE):
        sae = SparseDictionary.from_pretrained(
            item.sae.pretrained_name_or_path,
            device_mesh=sae_device_mesh,
            fold_activation_scale=False,
            strict_loading=item.sae.strict_loading,
            device=item.sae.device,
            dtype=item.sae.dtype,
        )
    elif item.initializer is not None:
        initializer = Initializer(item.initializer)
        sae = initializer.initialize_sae_from_config(
            item.sae,
            activation_stream=activations_stream,
            device_mesh=sae_device_mesh,
            model=model,
        )
    else:
        sae = SparseDictionary.from_config(item.sae, device_mesh=sae_device_mesh)

    wandb_logger = (
        wandb.init(
            project=item.wandb.wandb_project,
            config=item.model_dump(),
            name=item.wandb.exp_name,
            entity=item.wandb.wandb_entity,
            settings=wandb.Settings(x_disable_stats=True),
            mode=os.getenv("WANDB_MODE", "online"),  # type: ignore
        )
        if item.wandb is not None and is_primary_rank(device_mesh)
        else None
    )
    if wandb_logger is not None and isinstance(item.sae, SparseDictionaryConfig) and item.sae.sae_type == "lorsa":
        wandb_logger.config.update(
            {
                "lorsa_attn_type": getattr(item.sae, "attn_type", None),
                "lorsa_window_size": getattr(item.sae, "window_size", None),
                "lorsa_rotary_base": getattr(item.sae, "rotary_base", None),
                "lorsa_rotary_base_local": getattr(item.sae, "rotary_base_local", None),
            },
            allow_val_change=True,
        )
    # TODO: implement eval_fn
    eval_fn = (lambda x: None) if settings.eval else None

    logger.info("Starting training for sweep item")
    trainer = Trainer(item.trainer)
    sae.cfg.save_hyperparameters(item.trainer.exp_result_path)
    trainer.fit(sae=sae, activation_stream=activations_stream, eval_fn=eval_fn, wandb_logger=wandb_logger)

    logger.info("Training completed, saving sweep item")
    sae.save_pretrained(
        save_path=item.trainer.exp_result_path,
        fold_activation_scale=settings.fold_activation_scale,
    )
    if is_primary_rank(device_mesh) and mongo_client is not None:
        assert item.sae_name is not None and item.sae_series is not None, (
            "sae_name and sae_series must be provided when saving to MongoDB"
        )
        mongo_client.create_sae(
            name=item.sae_name,
            series=item.sae_series,
            path=str(Path(item.trainer.exp_result_path).absolute()),
            cfg=sae.cfg,
        )

    if wandb_logger is not None:
        wandb_logger.finish()
        logger.info("WandB session closed for sweep item")

    logger.info(f"Sweep item completed: {item.sae_name}/{item.sae_series}")
