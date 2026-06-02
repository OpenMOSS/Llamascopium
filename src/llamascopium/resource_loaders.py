from pathlib import Path
from typing import Any, Literal, Optional, cast

import datasets
from torch.distributed.device_mesh import DeviceMesh

from llamascopium.backend.language_model import (
    Evo2LanguageModel,
    HuggingFaceLanguageModel,
    LanguageModel,
    LanguageModelConfig,
    QwenVLLanguageModel,
    TokenizerOnlyLanguageModel,
    TransformerLensLanguageModel,
)
from llamascopium.config import DatasetConfig


def load_dataset_shard(
    cfg: DatasetConfig,
    shard_idx: int,
    n_shards: int,
) -> datasets.Dataset:
    if not cfg.is_dataset_on_disk:
        dataset = datasets.load_dataset(cfg.dataset_name_or_path, split="train", cache_dir=cfg.cache_dir)
    else:
        dataset = datasets.load_from_disk(cfg.dataset_name_or_path)
    dataset = cast(datasets.Dataset, dataset)
    dataset = dataset.shard(num_shards=n_shards, index=shard_idx, contiguous=True)
    return dataset


def load_dataset(
    cfg: DatasetConfig,
    device_mesh: Optional[DeviceMesh] = None,
    n_shards: Optional[int] = None,
    start_shard: int = 0,
) -> tuple[datasets.Dataset, Optional[dict[str, Any]]]:
    if not cfg.is_dataset_on_disk:
        dataset = datasets.load_dataset(
            cfg.dataset_name_or_path, split="train", cache_dir=cfg.cache_dir, trust_remote_code=True
        )
    else:
        dataset = datasets.load_from_disk(cfg.dataset_name_or_path)
    dataset = cast(datasets.Dataset, dataset)
    if device_mesh is not None:
        shard = dataset.shard(
            num_shards=n_shards or device_mesh.get_group("data").size(),
            index=start_shard + device_mesh.get_group("data").rank(),
            contiguous=True,
        )
        shard_metadata = {
            "shard_idx": start_shard + device_mesh.get_group("data").rank(),
            "n_shards": n_shards or device_mesh.get_group("data").size(),
        }
    else:
        shard = dataset
        shard_metadata = None
    return shard, shard_metadata


def infer_model_backend(model_name: str) -> Literal["huggingface", "transformer_lens", "evo2"]:
    if model_name.startswith("Qwen/Qwen2.5-VL"):
        return "huggingface"
    elif model_name.startswith("Qwen/Qwen2.5"):
        return "huggingface"
    elif model_name.startswith("arcinstitute/evo2_"):
        return "evo2"
    else:
        return "transformer_lens"


def load_model(cfg: LanguageModelConfig, device_mesh: DeviceMesh | None = None) -> LanguageModel:
    backend = infer_model_backend(cfg.model_name) if cfg.backend == "auto" else cfg.backend
    if backend == "tokenizer_only":
        return TokenizerOnlyLanguageModel(cfg)
    elif backend == "huggingface":
        if cfg.model_name.startswith("Qwen/Qwen2.5-VL"):
            return QwenVLLanguageModel(cfg)
        else:
            return HuggingFaceLanguageModel(cfg)
    elif backend == "evo2":
        return TransformerLensLanguageModel.from_pretrained_evo2(cfg, device_mesh)
    elif backend == "transformer_lens":
        return TransformerLensLanguageModel(cfg, device_mesh)
    # elif backend == "evo2":
    #     return Evo2LanguageModel(cfg)
    else:
        raise NotImplementedError(f"Backend {backend} not supported.")
