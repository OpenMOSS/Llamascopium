import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence, cast

import torch
from safetensors.torch import load_file
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from llamascopium.utils.distributed import DimMap, all_gather_dict, mesh_dim_rank, mesh_dim_size
from llamascopium.utils.misc import is_master
from llamascopium.utils.tensor_dict import move_dict_of_tensor_to_device

from .core import BaseActivationProcessor


class DistributedSampler(Sampler[Any]):
    def __init__(
        self,
        dataset: Dataset,
        chunk_to_index: dict[int, dict[str, int]],
        device_mesh: DeviceMesh,
    ) -> None:
        self.dataset = dataset
        self.chunk_to_index = chunk_to_index
        self.device_mesh = device_mesh
        self.n_samples = self._compute_n_samples()

    def _compute_n_samples(self) -> int:
        n_chunks = len(self.chunk_to_index)
        n_hook_points = len(next(iter(self.chunk_to_index.values())))

        dp_size = mesh_dim_size(self.device_mesh, "data")
        tp_size = mesh_dim_size(self.device_mesh, "model")
        sweep_size = mesh_dim_size(self.device_mesh, "sweep")
        non_dp_size = tp_size * sweep_size

        # Min chunk unit is decided by how many iterations are required to collect exact all hook points of some chunks in tensor parallel.
        min_chunk_unit = dp_size * non_dp_size // math.gcd(n_hook_points, non_dp_size)
        n_samples_total = n_chunks // min_chunk_unit * min_chunk_unit * n_hook_points
        n_samples_local = n_samples_total // dp_size // non_dp_size
        return n_samples_local

    def __iter__(self) -> Iterator[Any]:
        n_chunks = len(self.chunk_to_index)
        n_hook_points = len(next(iter(self.chunk_to_index.values())))

        dp_size = mesh_dim_size(self.device_mesh, "data")
        tp_size = mesh_dim_size(self.device_mesh, "model")
        sweep_size = mesh_dim_size(self.device_mesh, "sweep")

        non_dp_size = tp_size * sweep_size
        min_chunk_unit = dp_size * non_dp_size // math.gcd(n_hook_points, non_dp_size)
        n_chunks_effective = n_chunks // min_chunk_unit * min_chunk_unit

        # First assign chunks based on data groups to ensure each data group own self-contained chunks.
        chunks = list(self.chunk_to_index.keys())[
            mesh_dim_rank(self.device_mesh, "data") : n_chunks_effective : dp_size
        ]

        # In the local dp group: concat all chunks and flatten all hook point indices
        all_indices = list(itertools.chain.from_iterable([list(self.chunk_to_index[k].values()) for k in chunks]))
        model_rank = mesh_dim_rank(self.device_mesh, "model")
        sweep_rank = mesh_dim_rank(self.device_mesh, "sweep")
        combined_rank = model_rank * sweep_size + sweep_rank

        indices = all_indices[combined_rank::non_dp_size]

        assert len(indices) == self.n_samples, f"len(indices) {len(indices)} != self.n_samples {self.n_samples}"

        return iter(indices)

    def __len__(self) -> int:
        return self.n_samples


@dataclass
class ChunkInfo:
    """Information about a cached activation chunk file.

    Args:
        path: Path to the chunk file
        shard_id: Shard ID extracted from filename (0 if no shard)
        chunk_id: Chunk ID extracted from filename
    """

    path: Path
    shard_id: int
    chunk_id: int

    @classmethod
    def from_path(cls, path: Path) -> "ChunkInfo":
        """Create ChunkInfo from a file path.

        Supports formats:
        - shard-{shard_id}-chunk-{chunk_id}.(pt|safetensors)
        - chunk-{chunk_id}.(pt|safetensors)

        Args:
            path: Path to chunk file

        Returns:
            ChunkInfo object with parsed shard and chunk IDs

        Raises:
            ValueError: If filename doesn't match either expected pattern
        """
        # Try shard-chunk format first
        match = re.match(r"shard-(\d+)-chunk-(\d+)\.(pt|safetensors)", path.name)
        if match:
            return cls(path=path, shard_id=int(match.group(1)), chunk_id=int(match.group(2)))

        # Try chunk-only format
        match = re.match(r"chunk-(\d+)\.(pt|safetensors)", path.name)
        if match:
            return cls(
                path=path,
                shard_id=0,  # Default shard ID for non-sharded files
                chunk_id=int(match.group(1)),
            )

        raise ValueError(
            f"Invalid chunk filename format: {path.name}. "
            "Expected 'shard-{N}-chunk-{N}.(pt|safetensors)' or 'chunk-{N}.(pt|safetensors)'"
        )


def first_data_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return batch[0]


class CachedActivationLoader(BaseActivationProcessor[None, Iterable[dict[str, Any]]]):
    """Base class for cached activation loaders.

    Args:
        cache_dir: Root directory containing cached activations
        hook_points: List of hook point names to load
        device: Device to load tensors to
        num_workers: Number of worker processes for data loading. Default is 4
        prefetch_factor: Number of samples loaded in advance by each worker. Default is 8
        distributed: Whether to use distributed batch loading and broadcasting
    """

    def __init__(
        self,
        cache_dirs: Mapping[str, str | Path],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        self.cache_dirs = {k: Path(v) for k, v in cache_dirs.items()}
        self.device = device
        self.dtype = dtype
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.device_mesh = device_mesh

    def load_single_hook_chunk(
        self, chunk_idx: int, hook_point: str, hook_chunks: dict[str, list[ChunkInfo]]
    ) -> dict[str, Any]:
        """Load chunk data for a single hook point at given index.

        Args:
            chunk_idx: Index of the chunk to load
            hook_point: Name of the hook point to load
            hook_chunks: Dictionary mapping hook points to their chunk info lists

        Returns:
            dict[str, Any]: Chunk data for the specific hook point
        """
        chunk = hook_chunks[hook_point][chunk_idx]
        data: dict[str, Any] = self._load_chunk(chunk.path)

        # Validate data format
        assert isinstance(data, dict), f"Loading cached activation {chunk.path} error: returned {type(data)}"
        assert "activation" in data, f"Loading cached activation {chunk.path} error: missing 'activation' field"
        assert "tokens" in data, f"Loading cached activation {chunk.path} error: missing 'tokens' field"
        assert "mask" in data, f"Loading cached activation {chunk.path} error: missing 'mask' field"
        assert "attention_mask" in data, f"Loading cached activation {chunk.path} error: missing 'attention_mask' field"

        return {
            "hook_point": hook_point,
            "activation": data["activation"],
            "mask": data["mask"].bool() if isinstance(data["mask"], torch.Tensor) else data["mask"],
            "attention_mask": data["attention_mask"].long()
            if isinstance(data["attention_mask"], torch.Tensor)
            else data["attention_mask"],
            "tokens": data["tokens"].long() if isinstance(data["tokens"], torch.Tensor) else data["tokens"],
            "meta": data.get("meta"),
            "chunk_idx": chunk_idx,
        }

    def _get_sorted_chunks(self, hook_point: str) -> list[ChunkInfo]:
        """Get sorted list of chunk files for a hook point.

        Args:
            hook_point: Name of the hook point

        Returns:
            List of ChunkInfo objects sorted by shard ID then chunk ID

        Raises:
            FileNotFoundError: If hook point directory doesn't exist
        """
        hook_dir = self.cache_dirs[hook_point]
        if not hook_dir.exists():
            raise FileNotFoundError(f"Hook point directory not found: {hook_dir}")

        # Get both shard-chunk and chunk-only files, supporting both .pt and .safetensors
        chunks = [
            ChunkInfo.from_path(p)
            for pattern in ["shard-*-chunk-*.pt", "shard-*-chunk-*.safetensors", "chunk-*.pt", "chunk-*.safetensors"]
            for p in hook_dir.glob(pattern)
        ]
        return sorted(chunks, key=lambda x: (x.shard_id, x.chunk_id))

    def _load_chunk(self, chunk_path: Path) -> dict[str, Any]:
        """Load a chunk file, supporting both PyTorch and safetensors formats.

        Args:
            chunk_path: Path to the chunk file

        Returns:
            dict[str, Any]: Loaded data containing activations, tokens, and meta
        """
        if chunk_path.suffix == ".safetensors":
            chunk_data: dict[str, Any] = load_file(chunk_path, device="cpu")
            meta_path = chunk_path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                chunk_data = chunk_data | {"meta": meta}
            return chunk_data
        elif chunk_path.suffix == ".pt":
            return torch.load(chunk_path, map_location="cpu", weights_only=True)
        else:
            raise ValueError(f"Invalid chunk file format: {chunk_path}. Expected .safetensors or .pt")

    def _build_chunk_index_mappings(
        self, hook_chunks: dict[str, list[ChunkInfo]]
    ) -> tuple[dict[int, dict[str, int]], dict[int, tuple[int, str]]]:
        """Build index mappings between chunks and global indices."""
        num_chunks = len(next(iter(hook_chunks.values())))
        hook_points = list(hook_chunks.keys())
        num_hook_points = len(hook_points)

        index_to_chunk = {
            chunk_idx * num_hook_points + hook_idx: (chunk_idx, hook_points[hook_idx])
            for chunk_idx in range(num_chunks)
            for hook_idx in range(num_hook_points)
        }

        chunk_to_index = {
            chunk_idx: {
                hook_points[hook_idx]: chunk_idx * num_hook_points + hook_idx for hook_idx in range(num_hook_points)
            }
            for chunk_idx in range(num_chunks)
        }

        return chunk_to_index, index_to_chunk

    def _process_chunks(self, hook_chunks: dict[str, list[ChunkInfo]]) -> Iterator[dict[str, Any]]:
        """Process chunks using the appropriate method based on distributed setting.

        Args:
            hook_chunks: Dictionary mapping hook points to their chunk info lists
            num_chunks: Total number of chunks

        Returns:
            Iterator over chunk data
        """
        chunk_to_index, index_to_chunk = self._build_chunk_index_mappings(hook_chunks)
        cached_activation_dataset = CachedActivationDataset(
            self,
            hook_chunks,
            index_to_chunk,
        )
        dataloader = DataLoader(
            cached_activation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            collate_fn=first_data_collate_fn,
            sampler=DistributedSampler(
                cached_activation_dataset,
                chunk_to_index,
                self.device_mesh,
            )
            if self.device_mesh is not None
            else None,
        )

        for data in tqdm(
            dataloader,
            total=len(dataloader),
            desc="Processing activation chunks",
            disable=not is_master(),
        ):
            # Use all_gather_dict to gather chunk dicts from all ranks
            data = move_dict_of_tensor_to_device(data, device=self.device)
            if self.device_mesh is not None:
                gathered = all_gather_dict(data, group=self.device_mesh.get_group("model"))
                if mesh_dim_size(self.device_mesh, "sweep") > 1:
                    yield from (
                        sweep_item
                        for item in gathered
                        for sweep_item in all_gather_dict(item, group=self.device_mesh.get_group("sweep"))
                    )
                else:
                    yield from gathered
            else:
                yield data

    def process(self, data: None = None, **kwargs) -> Iterable[dict[str, Any]]:
        """Load cached activations in a streaming fashion.

        Yields dictionaries containing:
            - Activations for each hook point
            - Original tokens
            - Original info field

        The activations is of the shape (n_context, d_model).
        Files are loaded and yielded in order by shard ID then chunk ID.

        Args:
            data: Not used
            **kwargs: Additional keyword arguments (not used)

        Yields:
            dict[str, Any]: Dictionary containing activations, tokens, and info

        Raises:
            ValueError: If hook points have different numbers of chunks
            AssertionError: If loaded data doesn't match expected format
        """
        # Get sorted chunks for each hook point
        hook_chunks = {hook: self._get_sorted_chunks(hook) for hook in self.cache_dirs.keys()}

        # Verify all hook points have same number of chunks
        chunk_counts = {hook: len(chunks) for hook, chunks in hook_chunks.items()}
        if len(set(chunk_counts.values())) != 1:
            raise ValueError(
                f"Hook points have different numbers of chunks: {chunk_counts}. "
                "All hook points must have the same number of chunks."
            )

        # Group data by chunk_idx to maintain the same output structure
        chunk_buffer: dict[int, dict[str, Any]] = {}
        stream = self._process_chunks(hook_chunks)

        for single_hook_data in stream:
            chunk_idx = single_hook_data["chunk_idx"]
            hook_point = single_hook_data["hook_point"]

            # Initialize chunk buffer entry if not exists
            if chunk_idx not in chunk_buffer:
                chunk_buffer[chunk_idx] = {
                    "tokens": single_hook_data["tokens"],
                    "mask": single_hook_data["mask"],
                    "attention_mask": single_hook_data["attention_mask"],
                    "meta": single_hook_data["meta"],
                }

            # Add activation for this hook point
            chunk_buffer[chunk_idx][hook_point] = single_hook_data["activation"]

            # Verify tokens consistency across hook points for the same chunk
            if not torch.allclose(single_hook_data["tokens"], chunk_buffer[chunk_idx]["tokens"]):
                raise AssertionError(f"Loading cached activation error: tokens mismatch for chunk {chunk_idx}")

            # Check if we have all hook points for this chunk
            # -4 stands for initial tokens, meta, token_mask and attention_mask
            if len(chunk_buffer[chunk_idx]) - 4 == len(self.cache_dirs):
                activations = chunk_buffer.pop(chunk_idx)
                if self.dtype is not None:
                    for k, v in activations.items():
                        if k in self.cache_dirs.keys():
                            activations[k] = v.to(self.dtype)

                # Flatten tokens if needed
                while cast(torch.Tensor, activations["tokens"]).ndim >= 3:

                    def flatten(x: torch.Tensor | list[list[Any]]) -> torch.Tensor | list[Any]:
                        if isinstance(x, torch.Tensor):
                            return x.flatten(start_dim=0, end_dim=1)
                        else:
                            return [a for b in x for a in b]

                    activations = {k: flatten(v) for k, v in activations.items()}

                if self.device_mesh is not None:
                    activations = {
                        k: DTensor.from_local(
                            v,
                            device_mesh=self.device_mesh,
                            placements=DimMap({"data": 0}).placements(self.device_mesh),
                        )
                        if isinstance(v, torch.Tensor)
                        else v
                        for k, v in activations.items()
                    }

                yield activations


class CachedActivationDataset(Dataset):
    """Wrap the data loading process with torch dataset and loader for multiprocessing.

    This dataset uses (chunk_idx, hook_point) as the index to enable parallel loading
    of individual hook points instead of loading all hook points sequentially.
    """

    def __init__(
        self,
        activation_loader: CachedActivationLoader,
        hook_chunks: dict[str, list[ChunkInfo]],
        index_to_chunk: dict[int, tuple[int, str]],
    ):
        self.activation_loader = activation_loader
        self.hook_chunks = hook_chunks
        self.index_to_chunk = index_to_chunk

    def __len__(self):
        return len(self.index_to_chunk)

    def __getitem__(self, dataset_idx):
        chunk_idx, hook_point = self.index_to_chunk[dataset_idx]
        return self.activation_loader.load_single_hook_chunk(
            chunk_idx,
            hook_point,
            self.hook_chunks,
        )
