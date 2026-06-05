"""Post-analysis processor for Lorsa (Low-Rank Sparse Attention) SAE.

This module provides post-processing functionality specific to Lorsa SAEs,
including handling of z patterns (attention patterns) in sparse layout.

The z patterns represent the attention patterns for each active feature in the Lorsa model.
They are stored as sparse tensors with shape (batch, context, feature, n_ctx) where:
- batch: batch dimension
- context: sequence length dimension
- feature: feature dimension (d_sae)
- n_ctx: context length for attention patterns

Example:
    When analyzing Lorsa features, the post processor extracts z patterns for each feature
    and returns them in sparse layout format:

    {
        "top_z_patterns": {
            "indices": [[0, 0, 1], [0, 1, 2], [1, 0, 1]],  # batch, context, n_ctx
            "values": [0.1, 0.2, 0.3],  # attention pattern values
            "size": [2, 3, 5]  # batch, context, n_ctx dimensions
        }
    }
"""

from typing import Any

import numpy as np
import torch
from einops import repeat
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm

from llamascopium.activation.factory import ActivationFactory
from llamascopium.models.lorsa import LowRankSparseAttention
from llamascopium.models.sparse_dictionary import SparseDictionary
from llamascopium.utils.discrete import KeyedDiscreteMapper
from llamascopium.utils.distributed import is_primary_rank
from llamascopium.utils.logging import get_distributed_logger

from .base import PostAnalysisProcessor, register_post_analysis_processor

logger = get_distributed_logger("lorsa_post_analysis")
Z_PATTERN_HEAD_CHUNK_SIZE = 128


class LorsaPostAnalysisProcessor(PostAnalysisProcessor):
    """Post-analysis processor for Lorsa SAE.

    This processor handles Lorsa-specific analysis results including:
    - Z patterns (attention patterns) in sparse layout
    """

    def _process_tensors(
        self,
        sae: SparseDictionary,
        act_times: torch.Tensor,
        n_analyzed_tokens: int,
        max_feature_acts: torch.Tensor,
        sample_result: dict[str, dict[str, torch.Tensor]],
        mapper: KeyedDiscreteMapper,
        device_mesh: DeviceMesh | None = None,
        activation_factory: ActivationFactory | None = None,
        activation_factory_process_kwargs: dict[str, Any] = {},
    ) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]] | None]:
        """Process tensors and add Lorsa-specific data to sample_result.

        Args:
            sae: The sparse autoencoder model
            act_times: Tensor of activation times for each feature
            n_analyzed_tokens: Number of tokens analyzed
            max_feature_acts: Tensor of maximum activation values for each feature
            sample_result: Dictionary of sampling results
            mapper: KeyedDiscreteMapper for encoding/decoding metadata
            device_mesh: Device mesh for distributed tensors
            activation_factory: Factory for re-initializing activation stream
            activation_factory_process_kwargs: Keyword arguments for activation factory process
        Returns:
            Updated sample_result with any additional tensor data
        """
        assert isinstance(sae, LowRankSparseAttention)
        assert activation_factory is not None
        sample_device = next(
            sampling_data["feature_acts"].device for sampling_data in sample_result.values() if sampling_data is not None
        )

        # Extract interested (shard_idx, context_idx) pairs from sample_result
        interested_pairs = []
        head_indices = []
        sampling_slices = []

        _feature_acts = []
        global_row_start = 0
        for _, sampling_data in sample_result.items():
            if sampling_data is None:
                continue

            head_index = (
                torch.arange(sampling_data["feature_acts"].shape[1], device=sample_device, dtype=torch.long)[None, :]
                .expand(sampling_data["feature_acts"].shape[0], -1)
                .flatten()
            )
            head_indices.append(head_index)

            _feature_acts.append(sampling_data["feature_acts"])

            # Get shard_idx and context_idx from metadata
            # n_samples x d_sae
            shard_indices = sampling_data.get(
                "shard_idx", torch.zeros_like(sampling_data["feature_acts"][:, :, 0], dtype=torch.int64)
            ).flatten()
            context_indices = sampling_data["context_idx"].flatten()

            interested_pairs.append(torch.stack([shard_indices, context_indices], dim=1))
            n_sampling_rows = sampling_data["feature_acts"].shape[0] * sampling_data["feature_acts"].shape[1]
            sampling_slices.append((sampling_data, global_row_start, global_row_start + n_sampling_rows))
            global_row_start += n_sampling_rows
        interested_pairs = torch.cat(interested_pairs)
        head_indices = torch.cat(head_indices)
        _feature_acts = torch.cat(_feature_acts).flatten(0, 1)
        n_ctx = _feature_acts.size(-1)

        for sampling_data, _, _ in sampling_slices:
            d_sae = sampling_data["feature_acts"].shape[1]
            sampling_data["z_pattern_indices"] = [[] for _ in range(d_sae)]  # pyright: ignore[reportAssignmentType]
            sampling_data["z_pattern_values"] = [[] for _ in range(d_sae)]  # pyright: ignore[reportAssignmentType]

        # Re-initialize activation stream
        activation_stream = activation_factory.process(
            **activation_factory_process_kwargs,
        )

        visited = 0
        active_head_mask = act_times.ne(0).to(sample_device)
        pbar = tqdm(
            total=interested_pairs.shape[0],
            desc="Processing Lorsa z patterns",
            disable=not is_primary_rank(device_mesh),
        )
        # Iterate through activation stream
        for batch_data in activation_stream:
            # Extract metadata from batch
            meta = batch_data["meta"]

            for i, m in enumerate(meta):
                data_idx = torch.tensor(
                    [m.get("shard_idx", int(0)), m["context_idx"]], device=interested_pairs.device, dtype=torch.long
                )

                interested_pairs_idx = (data_idx == interested_pairs).all(dim=1)
                n_unfiltered_interested_pairs = interested_pairs_idx.sum()
                visited += n_unfiltered_interested_pairs

                interested_pairs_idx &= repeat(
                    tensor=active_head_mask,
                    pattern="d_sae -> (n_samples d_sae)",
                    n_samples=interested_pairs_idx.size(0) // active_head_mask.size(0),
                )

                if not interested_pairs_idx.any():
                    continue

                batch_device = batch_data[sae.cfg.hook_point_in].device
                interested_pair_positions = interested_pairs_idx.nonzero().squeeze(1)

                for chunk_start in range(0, interested_pair_positions.numel(), Z_PATTERN_HEAD_CHUNK_SIZE):
                    chunk_positions = interested_pair_positions[chunk_start : chunk_start + Z_PATTERN_HEAD_CHUNK_SIZE]
                    interested_heads = head_indices[chunk_positions].to(batch_device)
                    interested_feature_acts = _feature_acts[chunk_positions].to(device=batch_device, dtype=torch.float32)

                    z_pattern = sae.encode_z_pattern_for_head(
                        batch_data[sae.cfg.hook_point_in][i : i + 1],
                        interested_heads,
                    )
                    z_pattern *= interested_feature_acts.ne(0)[..., None]
                    small_zp_mask = z_pattern.abs() < 1e-2 * interested_feature_acts[..., None]
                    z_pattern.masked_fill_(small_zp_mask, 0.0)
                    z_pattern = z_pattern.to_sparse()
                    if z_pattern._nnz() == 0:
                        continue

                    z_pattern_indices = z_pattern.indices()
                    z_pattern_row_indices = z_pattern_indices[0].to(chunk_positions.device)
                    chunk_sample_feature_indices = chunk_positions[z_pattern_row_indices]
                    q_indices = z_pattern_indices[1].to(chunk_sample_feature_indices.device)
                    k_indices = z_pattern_indices[2].to(chunk_sample_feature_indices.device)
                    z_values = z_pattern.values().to(chunk_sample_feature_indices.device)

                    for sampling_data, slice_start, slice_end in sampling_slices:
                        sampling_mask = (chunk_sample_feature_indices >= slice_start) & (
                            chunk_sample_feature_indices < slice_end
                        )
                        if not sampling_mask.any():
                            continue

                        _, d_sae, _ = sampling_data["feature_acts"].shape
                        local_sample_feature_indices = chunk_sample_feature_indices[sampling_mask] - slice_start

                        for feature_idx in local_sample_feature_indices.remainder(d_sae).unique().tolist():
                            feature_mask = local_sample_feature_indices.remainder(d_sae).eq(feature_idx)
                            sample_indices = local_sample_feature_indices[feature_mask] // d_sae
                            z_pattern_indices = torch.stack(
                                [
                                    sample_indices,
                                    q_indices[sampling_mask][feature_mask],
                                    k_indices[sampling_mask][feature_mask],
                                ]
                            )
                            sampling_data["z_pattern_indices"][feature_idx].append(  # pyright: ignore[reportOptionalSubscript]
                                z_pattern_indices.cpu().numpy()
                            )
                            sampling_data["z_pattern_values"][feature_idx].append(  # pyright: ignore[reportOptionalSubscript]
                                z_values[sampling_mask][feature_mask].cpu().float().numpy()
                            )

                pbar.update(n_unfiltered_interested_pairs.item())

            if visited == interested_pairs.shape[0]:
                break

        for sampling_data, _, _ in sampling_slices:
            for feature_idx, indices_chunks in enumerate(sampling_data["z_pattern_indices"]):
                values_chunks = sampling_data["z_pattern_values"][feature_idx]
                if indices_chunks:
                    z_pattern_indices = np.concatenate(indices_chunks, axis=1)
                    z_pattern_values = np.concatenate(values_chunks)
                    unique_indices, inverse_indices = np.unique(z_pattern_indices, axis=1, return_inverse=True)
                    unique_values = np.zeros(unique_indices.shape[1], dtype=np.float32)
                    np.add.at(unique_values, inverse_indices, z_pattern_values)
                    nonzero_mask = unique_values != 0
                    sampling_data["z_pattern_indices"][feature_idx] = unique_indices[:, nonzero_mask]
                    sampling_data["z_pattern_values"][feature_idx] = unique_values[nonzero_mask]
                else:
                    sampling_data["z_pattern_indices"][feature_idx] = np.empty((3, 0), dtype=np.int64)
                    sampling_data["z_pattern_values"][feature_idx] = np.empty((0,), dtype=np.float32)

        return sample_result, None

    def _extra_info(self, sampling_data: dict[str, Any], i: int) -> dict[str, Any]:
        """Extra information to add to the feature result."""
        base_extra_info = super()._extra_info(sampling_data, i)
        z_pattern_indices = sampling_data["z_pattern_indices"][i]
        z_pattern_values = sampling_data["z_pattern_values"][i]

        return {
            **base_extra_info,
            "z_pattern_indices": z_pattern_indices,
            "z_pattern_values": z_pattern_values,
        }


# Register the processor for Lorsa SAE type
register_post_analysis_processor("lorsa", LorsaPostAnalysisProcessor)
