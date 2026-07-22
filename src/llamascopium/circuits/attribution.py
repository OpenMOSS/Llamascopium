from __future__ import annotations

import functools
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from itertools import groupby
from typing import (
    Any,
    Iterator,
    Literal,
    Self,
    Sequence,
    cast,
)

import einops
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from llamascopium.backend.language_model import TransformerLensLanguageModel
from llamascopium.circuits.hooks import replace_model_biases_with_leaves, replace_sae_biases_with_leaves
from llamascopium.circuits.indexed_tensor import (
    NodeDimension,
    NodeIndexed,
    NodeIndexedMatrix,
    NodeIndexedVector,
    NodeInfo,
)
from llamascopium.core.pytree import PyTree
from llamascopium.core.serialize import migrate
from llamascopium.models.lorsa import LowRankSparseAttention
from llamascopium.models.matryoshka_sae import MatryoshkaSparseAutoEncoder
from llamascopium.models.molt import MixtureOfLinearTransform
from llamascopium.models.sae import SparseAutoEncoder
from llamascopium.models.sparse_dictionary import SparseDictionary
from llamascopium.utils.distributed import DimMap, full_tensor
from llamascopium.utils.distributed.ops import maybe_local_map, multi_batch_index, nonzero, searchsorted
from llamascopium.utils.misc import ensure_tokenized
from llamascopium.utils.timer import timer

logger = logging.getLogger(__name__)
# Debugging
def debug_stage(stage: str, **values):
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
    else:
        allocated = reserved = 0

    details = " ".join(f"{key}={value}" for key, value in values.items())
    print(
        f"[CircuitDebug][rank={rank}] {stage} "
        f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB "
        f"{details}",
        flush=True,
    )
    

@dataclass
class NodeRefs:
    """Collection of named tensor references indexed by a :class:`NodeDimension`."""

    mapping: dict[str, torch.Tensor]
    dimension: NodeDimension

    @classmethod
    def from_nodes_and_refs(cls, entries: Sequence[tuple[str, torch.Tensor, torch.Tensor]]) -> Self:
        """Construct from ``(key, indices, ref)`` triples."""
        mapping: dict[str, torch.Tensor] = {}
        node_infos: list[NodeInfo] = []
        for key, indices, ref in entries:
            mapping[key] = ref
            node_infos.append(NodeInfo(key=key, indices=indices))
        if len(node_infos) == 0:
            raise ValueError("Cannot build NodeRefs from empty entries without an explicit device.")
        first_ref = next(iter(mapping.values()))
        device_mesh = first_ref.device_mesh if isinstance(first_ref, DTensor) else None
        return cls(mapping=mapping, dimension=NodeDimension.from_node_infos(node_infos, device_mesh=device_mesh))

    @property
    def device(self) -> torch.device:
        return next(iter(self.mapping.values())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.mapping.values())).dtype

    @property
    def device_mesh(self) -> DeviceMesh | None:
        first = next(iter(self.mapping.values()))
        return first.device_mesh if isinstance(first, DTensor) else None

    @property
    def batch_size(self) -> int:
        return next(iter(self.mapping.values())).shape[0]

    def refs(self) -> list[torch.Tensor]:
        """Unique ref tensors in insertion order (for ``torch.autograd.grad``)."""
        return list(self.mapping.values())

    @functools.cached_property
    def _entries(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        return [(ni.key, ni.indices, self.mapping[ni.key]) for ni in self.dimension.node_infos]

    @functools.cached_property
    @timer.time("values")
    def values(self) -> list[torch.Tensor]:  # type: ignore[override]
        """Index each ref by its node indices: ``mapping[key][:, *indices]``."""
        return multi_batch_index([(ref, indices) for _, indices, ref in self], n_batch_dims=1)

    @functools.cached_property
    def _slot_runs(self) -> list[tuple[bool, list[int]]]:
        """Contiguous source-order runs of slots with the same indices property and shape.
        Used for efficient computation of attribution scores.
        """

        def kind(s: int) -> tuple[bool, tuple[int, ...] | None]:
            is_full = self.dimension.node_infos[s].is_identity
            return (is_full, tuple(self.values[s].shape[1:]) if is_full else None)

        return [(key[0], list(group)) for key, group in groupby(range(len(self.dimension.node_infos)), key=kind)]

    def iter_batches(self, batch_size: int) -> Iterator[NodeRefs]:
        """Yield sub-:class:`NodeRefs` of at most `batch_size` elements."""
        queue = list(self.dimension.node_infos)
        while queue:
            accumulated = 0
            batch: list[NodeInfo] = []
            while accumulated < batch_size and queue:
                ni = queue[0]
                remaining = batch_size - accumulated
                if len(ni) > remaining:
                    batch.append(ni[:remaining])
                    queue[0] = ni[remaining:]
                    accumulated = batch_size
                else:
                    batch.append(queue.pop(0))
                    accumulated += len(batch[-1])
            yield self[NodeDimension.from_node_infos(batch, device=self.dimension.device)]

    def __add__(self, other: NodeRefs) -> NodeRefs:
        """Combine two :class:`NodeRefs` (merge mappings, concatenate dimensions)."""
        return NodeRefs(mapping={**self.mapping, **other.mapping}, dimension=self.dimension + other.dimension)

    def __getitem__(self, index: NodeDimension) -> NodeRefs:
        """Subset to a sub-dimension (filter mapping to relevant keys)."""
        relevant_keys = {node.key for node in index.node_mappings.values()}
        return NodeRefs(
            mapping={k: v for k, v in self.mapping.items() if k in relevant_keys},
            dimension=index,
        )

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor, torch.Tensor]]:
        return iter(self._entries)


@dataclass
class IntermediateRefs:
    """Paired upstream/downstream refs for intermediate nodes."""

    upstream: NodeRefs
    downstream: NodeRefs


@dataclass
class QKTracingResult(PyTree):
    """Result of QK tracing.

    It contains the marginal attributions for Q and K roles, and the second-order pairwise attributions for QK pairs.
    """

    q_marginal: NodeIndexed[list[NodeIndexed[torch.Tensor]]]
    k_marginal: NodeIndexed[list[NodeIndexed[torch.Tensor]]]
    pairs: NodeIndexed[list[NodeIndexed[torch.Tensor]]]

    @migrate(before="2")
    @staticmethod
    def _v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
        def _dim_size(dim_raw: dict[str, Any]) -> int:
            return sum(int(node["indices"].shape[0]) for node in dim_raw.values()) if dim_raw else 0

        q_marginal_slots: list[dict[str, Any]] = []
        k_marginal_slots: list[dict[str, Any]] = []
        for slot in data["value"]:
            q_dim_raw = slot["dimensions"][0]
            k_dim_raw = slot["dimensions"][1]
            q_marginal_slots.append({"value": torch.zeros(_dim_size(q_dim_raw)), "dimensions": [q_dim_raw]})
            k_marginal_slots.append({"value": torch.zeros(_dim_size(k_dim_raw)), "dimensions": [k_dim_raw]})

        return {
            "q_marginal": {"value": q_marginal_slots, "dimensions": data["dimensions"]},
            "k_marginal": {"value": k_marginal_slots, "dimensions": data["dimensions"]},
            "pairs": data,
        }


@dataclass
class AttributionResult(PyTree):
    """Result of attribution computation."""

    activations: NodeIndexedVector
    attribution: NodeIndexedMatrix
    targets: NodeDimension | None = None
    logits: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_tokens: list[str] = field(default_factory=list)
    logit_token_ids: list[int] | None = None
    logit_tokens: list[str] | None = None
    qk_trace_results: QKTracingResult | None = None


def get_normalized_matrix(matrix: NodeIndexedMatrix) -> NodeIndexedMatrix:
    return NodeIndexedMatrix.from_data(
        data=torch.abs(matrix.data) / torch.abs(matrix.data).sum(dim=1, keepdim=True).clamp(min=1e-8),
        dimensions=matrix.dimensions,
    )


@timer.time("compute_intermediates_attribution")
def compute_intermediates_attribution(
    attribution: NodeIndexedMatrix,
    targets: NodeDimension,
    intermediates: NodeDimension,
    max_iter: int,
) -> NodeIndexedMatrix:
    attribution = get_normalized_matrix(attribution)
    influence = attribution[targets, None]
    if len(intermediates) == 0:
        return influence
    t2i: NodeIndexedMatrix = attribution[targets, intermediates]
    i2all: NodeIndexedMatrix = attribution[intermediates, None]
    i2i: NodeIndexedMatrix = attribution[intermediates, intermediates]
    for _ in range(max_iter):
        cur_influence = t2i @ i2all
        if not torch.any(cur_influence.data):
            break
        influence += cur_influence
        t2i = t2i @ i2i
    return influence


@timer.time("_full_slot_attribution")
def _full_slot_attribution(
    values: list[torch.Tensor],
    grad_refs: Sequence[torch.Tensor | None],
    slots: list[int],
) -> torch.Tensor:
    """Input×gradient for identity-indexed slots."""
    grads = [grad_refs[s] if grad_refs[s] is not None else torch.zeros_like(values[s]) for s in slots]
    stacked_v = torch.stack([values[s].detach() for s in slots], dim=1)
    stacked_g = torch.stack([cast(torch.Tensor, g) for g in grads], dim=1)
    product = einops.reduce(stacked_v * stacked_g, "batch sources n ... -> batch sources n", "sum")
    return product.flatten(start_dim=1)


@timer.time("_sparse_slot_attribution")
def _sparse_slot_attribution(
    values: list[torch.Tensor],
    grad_refs: Sequence[torch.Tensor | None],
    slots: list[int],
    indices: list[torch.Tensor],
) -> torch.Tensor:
    """Input×gradient for sparse-indexed slots."""
    nonzero = [(cast(torch.Tensor, grad_refs[s]), idx) for s, idx in zip(slots, indices) if grad_refs[s] is not None]
    indexed = multi_batch_index(nonzero, n_batch_dims=1) if nonzero else []
    grad_iter = iter(indexed)
    parts = [
        einops.reduce(values[s].detach() * next(grad_iter), "batch n ... -> batch n", "sum")
        if grad_refs[s] is not None
        else values[s].detach().new_zeros(values[s].shape[:2])
        for s in slots
    ]
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


@timer.time("attribution_scores")
def attribution_scores(
    grad_refs: Sequence[torch.Tensor | None],
    sources: NodeRefs,
) -> torch.Tensor:
    """Input×gradient attribution scores of shape ``(fwd_batch, n_sources)``."""
    blocks = [
        _full_slot_attribution(sources.values, grad_refs, slots)
        if is_full
        else _sparse_slot_attribution(
            sources.values, grad_refs, slots, [sources.dimension.node_infos[s].indices for s in slots]
        )
        for is_full, slots in sources._slot_runs  # pyright: ignore[reportPrivateUsage]
    ]
    return torch.cat(blocks, dim=1) if len(blocks) > 1 else blocks[0]


@timer.time("greedily_collect_attribution")
def greedily_collect_attribution(
    targets: NodeRefs,
    sources: NodeRefs,
    intermediates: IntermediateRefs,
    max_intermediates: int,
    reduction_weight: torch.Tensor,
    required_intermediates: NodeDimension | None = None,
    max_iter: int = 100,
) -> tuple[NodeIndexedMatrix, NodeDimension]:
    """Greedily collect attribution from targets to sources through intermediates."""

    all_sources = sources + intermediates.downstream

    attribution = NodeIndexedMatrix.from_dimensions(
        dimensions=(targets.dimension, all_sources.dimension),
        device=targets.device,
        dtype=targets.dtype,
        device_mesh=targets.device_mesh,
    )

    batch_size = targets.batch_size

    @timer.time("per_target_attribution")
    def per_target_attribution(targets: NodeRefs) -> torch.Tensor:
        
        debug_stage("backward.root.begin", targets=len(targets.dimension))
        
        root = maybe_local_map(torch.diag)(torch.cat(targets.values, dim=1))
        
        debug_stage("backward.root.end", root_shape=tuple(root.shape))
        debug_stage("autograd.grad.begin")
        grad_refs = torch.autograd.grad(
            root.sum(),
            all_sources.refs(),
            retain_graph=True,
            materialize_grads=True,
        )
        debug_stage("autograd.grad.end")

        debug_stage("attribution_scores.begin")
        scores = attribution_scores(grad_refs, all_sources)[:root.shape[0]]
        debug_stage("attribution_scores.end", shape=tuple(scores.shape))
        return scores

    for target_batch in targets.iter_batches(batch_size):
        attribution[target_batch.dimension, None] = per_target_attribution(target_batch).to(attribution.data.dtype)

    collected = (
        required_intermediates.unique()
        if required_intermediates is not None
        else NodeDimension.empty(device=targets.device, device_mesh=targets.device_mesh)
    )
    collected = collected - targets.dimension
    if len(collected) > 0:
        required_refs = intermediates.upstream[collected]
        for required_batch in required_refs.iter_batches(batch_size):
            attribution.add_targets(
                required_batch.dimension,
                per_target_attribution(required_batch).to(attribution.data.dtype),
            )

    max_intermediates = min(
        max(max_intermediates, len(collected)),
        len(intermediates.downstream.dimension),
    )
    remaining_budget = max_intermediates - len(collected)
    reduction_weight_vec: NodeIndexedVector = NodeIndexedVector.from_data(
        reduction_weight, dimensions=(targets.dimension,)
    )
    

    
    for i in tqdm(range(0, remaining_budget, batch_size), desc="OV Attribution"):
        
        #Debugging
        step = i // batch_size
        debug_stage("iteration.begin", step=step, collected=len(collected))
        debug_stage("available.begin", step=step)
        # End of debugging
        
        available = (intermediates.downstream.dimension - collected).unique()
        
        # Debugging 
        debug_stage("available.end", step=step, available=len(available))
        
        cur_batch_size = min(batch_size, remaining_budget - i, len(available))
        
        # Debugging
        if cur_batch_size == 0:
            debug_stage("iteration.no_candidates", step=step)
            break
        debug_stage("compute_intermediates.begin", step=step)

        intermediates_attribution = compute_intermediates_attribution(
            attribution, targets.dimension, collected, max_iter
        )
        
        # Debugging
        debug_stage(
            "compute_intermediates.end",
            step=step,
            shape=tuple(intermediates_attribution.data.shape),
        )

        influence = reduction_weight_vec @ intermediates_attribution[None, intermediates.downstream.dimension]


        # Debugging
        debug_stage("influence.end", step=step)

        debug_stage("topk.begin", step=step, k=cur_batch_size)


        # Select from the remaining logical nodes directly. Masking the full
        # replicated DTensor with ``ignore_dimension`` can reselect collected
        # nodes when the dimension contains duplicate logical indices.
        available_influence = influence[available]
        _, selected_nodes = available_influence.topk(k=cur_batch_size)
        selected_nodes = selected_nodes.unique()
        debug_stage("topk.end", step=step, selected=len(selected_nodes))

        collected = collected + selected_nodes

        selected_refs = intermediates.upstream[selected_nodes]
        
        
        debug_stage("backward.begin", step=step)
        selected_attribution = per_target_attribution(selected_refs)
        debug_stage(
            "backward.end",
            step=step,
            shape=tuple(selected_attribution.shape),
        )

        debug_stage("add_targets.begin", step=step)
        attribution.add_targets(
            selected_nodes,
            selected_attribution.to(attribution.data.dtype),
        )
        debug_stage("iteration.end", step=step, collected=len(collected))
            
        
        
        
        # attribution.add_targets(
        #     selected_nodes,
        #     per_target_attribution(selected_refs).to(attribution.data.dtype),
        # )

    return attribution, collected


def _validate_active_matryoshka_nodes(
    replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform],
    intermediate_indices: dict[str, torch.Tensor],
    final_position: int,
    feature_range: tuple[int, int] | None,
) -> None:
    """Reject a trace whose Matryoshka range is inactive at the final token."""
    for module in replacement_modules:
        if not isinstance(module, MatryoshkaSparseAutoEncoder):
            continue
        key = module.cfg.hook_point_out + ".sae.hook_feature_acts"
        indices = intermediate_indices[key]
        final_indices = indices[indices[:, 0] == final_position]
        if len(final_indices) == 0:
            range_label = "full range" if feature_range is None else f"[{feature_range[0]}, {feature_range[1]})"
            raise ValueError(
                f"Matryoshka SAE at {module.cfg.hook_point_out} has no active features "
                f"at the final token position {final_position} in {range_label}. "
                "Choose a range containing at least one final-token activation."
            )


def ln_detach_hooks(model: TransformerLensLanguageModel) -> list[str]:
    detach_hooks = []
    for i, block in enumerate(model.blocks):
        for module_name in ["ln1", "ln2", "ln1_post", "ln2_post"]:
            if hasattr(block, module_name) and isinstance(getattr(block, module_name), torch.nn.Module):
                detach_hooks.append(f"blocks.{i}.{module_name}.hook_scale")

    detach_hooks.append("ln_final.hook_scale")
    return detach_hooks


def attn_detach_hooks(model: TransformerLensLanguageModel) -> list[str]:
    detach_hooks = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, "attn") and isinstance(block.attn, torch.nn.Module):
            detach_hooks.append(f"blocks.{i}.attn.hook_pattern")
            if model.cfg.use_qk_norm:
                detach_hooks.append(f"blocks.{i}.attn.q_norm.hook_scale")
                detach_hooks.append(f"blocks.{i}.attn.k_norm.hook_scale")
    return detach_hooks


@timer.time("_find_influence_threshold")
def _find_influence_threshold(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Find score threshold that keeps the desired fraction of total influence."""
    if scores.numel() == 0:
        return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    sorted_scores = torch.sort(scores.view(-1), descending=True).values
    cumulative_score = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores).clamp(min=1e-8)
    threshold_index = searchsorted(cumulative_score, threshold)
    threshold_index = min(int(threshold_index.item()), len(cumulative_score) - 1)
    return sorted_scores[threshold_index]


@timer.time("prune_attribution")
def prune_attribution(
    attribution: NodeIndexedMatrix,
    reduction_weight: torch.Tensor,
    node_threshold: float = 0.6,
    edge_threshold: float = 0.8,
    targets: NodeDimension | None = None,
    required_nodes: NodeDimension | None = None,
) -> NodeIndexedMatrix:
    """Prune an attribution NodeIndexedMatrix by removing low-influence nodes and edges.

    The attribution matrix is expected to have:
    - dim 0 (targets/rows): target nodes (logits or user-specified features) + collected feature nodes
    - dim 1 (sources/cols): embed nodes (key="hook_embed") + error nodes (key ends with ".error")
                             + all (possibly uncollected) feature nodes

    Target nodes and embed/error source nodes are always kept. Intermediate
    feature nodes are pruned based on their cumulative contribution to the
    weighted target output.

    Args:
        attribution: NodeIndexedMatrix from the attribution computation.
        reduction_weight: Per-target scalar weights (shape ``[n_targets]``) used to
            collapse the target dimension into a single influence score per
            intermediate / source. For logit targets this is typically the
            target-token probabilities; for feature targets, uniform ones.
        node_threshold: Retain feature nodes accounting for this fraction of total influence.
        edge_threshold: Retain edges accounting for this fraction of total edge influence.
        targets: Target NodeDimension identifying which rows of ``attribution``
            are targets vs intermediates. When ``None``, defaults to rows with
            key ``"logits"`` for backward compatibility with logit-target
            attributions.
        required_nodes: Intermediate nodes that must survive influence pruning
            when they have non-zero target and upstream attribution edges.

    Returns:
        Pruned NodeIndexedMatrix containing only kept nodes and edges.
    """
    if node_threshold > 1.0 or node_threshold < 0.0:
        raise ValueError("node_threshold must be between 0.0 and 1.0")
    if edge_threshold > 1.0 or edge_threshold < 0.0:
        raise ValueError("edge_threshold must be between 0.0 and 1.0")

    if targets is None:
        targets = attribution.dimensions[0].filter_keys(lambda key: key == "logits")
    intermediates = attribution.dimensions[0] - targets
    optional_sources = attribution.dimensions[1].filter_keys(lambda key: key.endswith(".error")) + intermediates

    node_scores = NodeIndexedVector.from_data(
        reduction_weight, dimensions=(targets,)
    ) @ compute_intermediates_attribution(attribution, targets, intermediates, max_iter=100)
    influence = node_scores[intermediates]
    influence.add_nodes(targets, reduction_weight)
    edge_scores = NodeIndexedMatrix.from_data(
        get_normalized_matrix(attribution).data * influence[attribution.dimensions[0]].data[:, None],
        dimensions=attribution.dimensions,
    )

    node_mask = node_scores.map(lambda x: x >= _find_influence_threshold(x, node_threshold))
    edge_mask = edge_scores.map(lambda x: x >= _find_influence_threshold(x, edge_threshold))

    if required_nodes is not None and len(required_nodes) > 0:
        required_source_offsets = attribution.dimensions[1].nodes_to_offsets(required_nodes)
        required_row_offsets = attribution.dimensions[0].nodes_to_offsets(required_nodes)
        required_source_offsets = required_source_offsets[required_source_offsets >= 0]
        required_row_offsets = required_row_offsets[required_row_offsets >= 0]
        target_offsets = attribution.dimensions[0].nodes_to_offsets(targets)

        node_mask.data[required_source_offsets] = True

        if len(required_source_offsets) > 0 and len(target_offsets) > 0:
            target_grid = target_offsets[:, None]
            source_grid = required_source_offsets[None, :]
            nonzero_target_edges = attribution.data[target_grid, source_grid] != 0
            edge_mask.data[target_grid, source_grid] |= nonzero_target_edges

        if len(required_row_offsets) > 0:
            required_rows = attribution.data[required_row_offsets].abs()
            strongest_values, strongest_sources = required_rows.max(dim=1)
            nonzero_rows = strongest_values > 0
            edge_mask.data[
                required_row_offsets[nonzero_rows],
                strongest_sources[nonzero_rows],
            ] = True

    old_node_mask = node_mask.clone()
    node_mask[optional_sources] = node_mask[optional_sources] & edge_mask[None, optional_sources].any(0)
    node_mask[intermediates] = node_mask[intermediates] & edge_mask[intermediates, None].any(1)

    while not torch.equal(node_mask.data, old_node_mask.data):
        old_node_mask = node_mask.clone()
        edge_mask.masked_fill_dim_(1, ~node_mask[optional_sources], False)
        edge_mask.masked_fill_dim_(0, ~node_mask[intermediates], False)
        node_mask[optional_sources] = node_mask[optional_sources] & edge_mask[None, optional_sources].any(0)
        node_mask[intermediates] = node_mask[intermediates] & edge_mask[intermediates, None].any(1)

    attribution = attribution.clone()
    attribution.masked_fill_dim_(1, ~node_mask[optional_sources], 0)
    attribution.masked_fill_dim_(0, ~node_mask[intermediates], 0)
    attribution.masked_fill_(~edge_mask, 0)
    return attribution


@timer.time("collect_cache")
def collect_cache(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform],
    with_bias_leaves: bool = False,
    matryoshka_feature_range: tuple[int, int] | None = None,
):
    """Run model forward pass and collect useful cache for circuit tracing.

    This function internally replaces model/SAE bias with batched leaves, for batch-simulation of Jacobian computation. It also uses `run_with_ref_cache` to directly collect the tensors that are in the computational graph, instead of cloning them in normal `run_with_cache`."""

    tokens = ensure_tokenized(inputs, model.tokenizer, device=model.device)
    batch_size, seq_len = tokens.shape[0], tokens.shape[1]

    model_bias_ctx = (
        replace_model_biases_with_leaves(model, batch_size, seq_len, device_mesh=model.device_mesh)
        if with_bias_leaves
        else nullcontext([])
    )
    sae_bias_ctx = (
        replace_sae_biases_with_leaves(
            model,
            cast(list[SparseDictionary], replacement_modules),
            batch_size,
            seq_len,
            device_mesh=model.device_mesh,
        )
        if with_bias_leaves
        else nullcontext([])
    )

    detach_hook_points = (
        ["hook_embed"]
        + [rm.cfg.hook_point_out + ".error" for rm in replacement_modules]
        + [rm.cfg.hook_point_out + ".sae.hook_feature_acts" for rm in replacement_modules]
        + [
            rm.cfg.hook_point_out + ".sae.hook_attn_pattern"
            for rm in replacement_modules
            if isinstance(rm, LowRankSparseAttention)
        ]
        + [
            rm.cfg.hook_point_out + item
            for rm in replacement_modules
            if isinstance(rm, LowRankSparseAttention) and rm.cfg.use_post_qk_ln
            for item in (".sae.ln_q.hook_scale", ".sae.ln_k.hook_scale")
        ]
        + ln_detach_hooks(model)
        + attn_detach_hooks(model)
    )

    with (
        model_bias_ctx as model_bias_names,
        model.apply_saes(
            cast(list[SparseDictionary], replacement_modules),
            matryoshka_feature_range=matryoshka_feature_range,
        ),
        sae_bias_ctx as sae_bias_names,
        model.detach_at(detach_hook_points),
    ):
        logits, cache = model.run_with_ref_cache(
            tokens,
            names_filter=["hook_embed.post"]
            + [rm.cfg.hook_point_out + ".error.post" for rm in replacement_modules]
            + [rm.cfg.hook_point_out + ".sae.hook_feature_acts.pre" for rm in replacement_modules]
            + [rm.cfg.hook_point_out + ".sae.hook_feature_acts.post" for rm in replacement_modules]
            + [
                rm.cfg.hook_point_out + ".sae.hook_attn_score"
                for rm in replacement_modules
                if isinstance(rm, LowRankSparseAttention)
            ]
            + [
                rm.cfg.hook_point_out + suffix
                for rm in replacement_modules
                if isinstance(rm, LowRankSparseAttention)
                for suffix in (".sae.hook_q_post_rot", ".sae.hook_k_post_rot")
            ]
            + ln_detach_hooks(model)
            + model_bias_names
            + sae_bias_names,
        )

    return logits, cache


@timer.time("attribute")
def attribute(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseDictionary],
    target_type: Literal["logits", "features"] = "logits",
    features: list[tuple[str, int, int]] | NodeDimension = [],
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_features: int | None = None,
    enable_qk_tracing: bool = False,
    qk_top_fraction: float = 0.6,
    qk_topk: int = 10,
    matryoshka_feature_range: tuple[int, int] | None = None,
):
    """Run end-to-end circuit attribution for a single prompt.

    Args:
        model: The language model to attribute through.
        inputs: Prompt as a token tensor or raw string (tokenized internally).
        replacement_modules: SAE / Lorsa / MoLT modules spliced into the
            residual stream as the upstream feature basis.
        target_type: The type of interested targets to attribute from. Can be "logits" or "features".
        features: Feature nodes to attribute from. Can be a NodeDimension directly representing the feature nodes, or a list of tuples (key, pos, feature_index) representing the feature nodes. Required if target_type is "features".
        max_n_logits: Maximum number of top output logits to treat as targets.
        desired_logit_prob: Cumulative probability mass that the selected top
            logits must cover; the actual target count is the smallest prefix
            of the top logits whose probabilities sum to at least this value.
        batch_size: Forward replication factor — the prompt is repeated this
            many times so that each target can be backwarded through a
            distinct forward-batch row to simulate the Jacobian computation.
            Affects the memory usage strongly.
        max_features: The number of upstream features collected by the
            greedy attribution loop; ``None`` means collecting all features.
        enable_qk_tracing: If ``True``, also run Q/K bilinear pair attribution
            for Lorsa attention targets. Note this will increase the memory usage,
            and takes more time to compute (typically 2x-10x slower).
        qk_top_fraction: Fraction of top Lorsa features to include
            as QK tracing targets.
        qk_topk: Number of QK tracing results to keep. This applies to both Q/K marginal attributions and pairwise attributions.
        matryoshka_feature_range: Optional ``(start, end)`` latent range for
            every Matryoshka SAE in ``replacement_modules``. ``None`` exposes
            its full feature space.
    """
    assert not enable_qk_tracing or batch_size >= qk_topk, (
        f"attribute(batch_size={batch_size}) must be >= qk_topk={qk_topk} when enable_qk_tracing=True"
    )
    tokens = ensure_tokenized(inputs, model.tokenizer, device=model.device)
    replacement_modules: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
    )
    batch_logits, cache = collect_cache(
        model,
        einops.repeat(tokens, "n -> b n", b=batch_size),
        replacement_modules,
        with_bias_leaves=enable_qk_tracing,
        matryoshka_feature_range=matryoshka_feature_range,
    )

    seq_len = cache["hook_embed.post"].shape[1]

    if target_type == "features":
        feat_dim = _coerce_feature_dim(features, model.device, model.device_mesh)
        # Use ``.pre`` (pre-detach, still connected to the encoder) as the target
        # ref so gradients flow back through the SAE encoder → residual stream →
        # upstream sources. ``.post`` is a detached leaf with no upstream path,
        # which would make attribution collapse to the target's self-entry.
        targets = NodeRefs.from_nodes_and_refs(
            [(ni.key, ni.indices, cache[ni.key + ".pre"]) for ni in feat_dim.node_infos]
        )
        reduction_weight = torch.ones(len(feat_dim), device=model.device, dtype=targets.dtype)
    elif target_type == "logits":
        with torch.no_grad():
            probs = torch.softmax(batch_logits[0, -1], dim=-1)
            top_p, top_idx = torch.topk(probs, max_n_logits)
            cutoff = int(searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
            top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]
        targets = NodeRefs.from_nodes_and_refs(
            [
                (
                    "logits",
                    top_idx.unsqueeze(-1),
                    batch_logits[:, -1, :] - batch_logits[:, -1, :].mean(dim=-1, keepdim=True),
                ),
            ]
        )
        reduction_weight = top_p
    else:
        raise ValueError(f"invalid target_type={target_type!r}")

    seq_indices = (
        torch.arange(seq_len, device=model.device).unsqueeze(-1)
        if model.device_mesh is None
        else DTensor.from_local(
            torch.arange(seq_len, device=model.device).unsqueeze(-1),
            device_mesh=model.device_mesh,
            placements=DimMap({}).placements(model.device_mesh),
        )
    )
    sources = NodeRefs.from_nodes_and_refs(
        [("hook_embed", seq_indices, cache["hook_embed.post"])]
        + [
            (rm.cfg.hook_point_out + ".error", seq_indices, cache[rm.cfg.hook_point_out + ".error.post"])
            for rm in replacement_modules
        ]
    )

    intermediate_entries = []
    intermediate_indices: dict[str, torch.Tensor] = {}
    for rm in replacement_modules:
        key = rm.cfg.hook_point_out + ".sae.hook_feature_acts"
        indices = nonzero(cache[key + ".pre"][0])
        full_indices = full_tensor(indices)
        intermediate_indices[key] = full_indices
        intermediate_entries.append(
            (
                key,
                indices,
                cache[key + ".pre"],
                cache[key + ".post"],
            )
        )

    _validate_active_matryoshka_nodes(
        replacement_modules,
        intermediate_indices,
        final_position=seq_len - 1,
        feature_range=matryoshka_feature_range,
    )
    intermediates = IntermediateRefs(
        upstream=NodeRefs.from_nodes_and_refs([(key, idx, pre) for key, idx, pre, _ in intermediate_entries]),
        downstream=NodeRefs.from_nodes_and_refs([(key, idx, post) for key, idx, _, post in intermediate_entries]),
    )

    max_intermediates = max_features if max_features is not None else len(replacement_modules)
    max_iter = len(replacement_modules) + 10

    attribution, collected_intermediates = greedily_collect_attribution(
        targets=targets,
        sources=sources,
        intermediates=intermediates,
        max_intermediates=max_intermediates,
        reduction_weight=reduction_weight,
        max_iter=max_iter,
    )

    attribution = attribution[None, sources.dimension + collected_intermediates]

    activations = torch.cat(
        multi_batch_index(
            [(ref[0].detach(), indices) for _, indices, ref in targets],
        )
        + multi_batch_index(
            [(ref[0].detach(), indices) for key, indices, ref in intermediates.upstream[collected_intermediates]],
        )
        + [torch.ones_like(indices[:, 0], dtype=sources.dtype) for _, indices, _ in sources],
        dim=0,
    )

    activations_vec = NodeIndexedVector.from_data(
        data=activations,
        dimensions=(targets.dimension + collected_intermediates + sources.dimension,),
    )

    prompt_token_ids = full_tensor(tokens).detach().cpu().tolist()
    logit_token_ids = (
        full_tensor(top_idx).detach().cpu().tolist() if target_type == "logits" and top_idx.numel() > 0 else None  # pyright: ignore[reportPossiblyUnboundVariable]
    )

    qk_trace_results = None
    if enable_qk_tracing:
        lorsa_features = collected_intermediates.filter_keys(
            lambda key: any(
                isinstance(rm, LowRankSparseAttention) and key == rm.cfg.hook_point_out + ".sae.hook_feature_acts"
                for rm in replacement_modules
            )
        )

        if len(lorsa_features) > 0:
            lorsa_influence = (
                NodeIndexedVector.from_data(reduction_weight, dimensions=(targets.dimension,))
                @ compute_intermediates_attribution(attribution, targets.dimension, collected_intermediates, max_iter)[
                    None, lorsa_features
                ]
            )

            _, top_lorsa_features = lorsa_influence.topk(k=max(1, int(len(lorsa_features) * qk_top_fraction)))

            q_targets, k_targets, slot_to_unique = _retrieve_qk_vector_targets(
                top_lorsa_features,
                cache,
                {
                    rm.cfg.hook_point_out: rm.cfg.ov_group_size
                    for rm in replacement_modules
                    if isinstance(rm, LowRankSparseAttention)
                },
                {
                    rm.cfg.hook_point_out: rm.attn_scale
                    for rm in replacement_modules
                    if isinstance(rm, LowRankSparseAttention)
                },
            )

            bias_keys = [
                f"blocks.{i}.{kind}" for i in range(len(model.blocks)) for kind in ("attn.hook_b_O", "mlp.hook_b_out")
            ] + [
                rm.cfg.hook_point_out + suffix
                for rm in replacement_modules
                for suffix in (".sae.hook_b_Q", ".sae.hook_b_K", ".sae.hook_b_D")
            ]
            qk_sources = sources + NodeRefs.from_nodes_and_refs(
                [
                    (
                        rm.cfg.hook_point_out + ".sae.hook_feature_acts",
                        nonzero(cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.post"][0]),
                        cache[rm.cfg.hook_point_out + ".sae.hook_feature_acts.post"],
                    )
                    for rm in replacement_modules
                ]
                + [(k, seq_indices, cache[k]) for k in bias_keys if k in cache]
            )

            tracing_results = compute_qk_tracing(q_targets, k_targets, qk_sources, topk=qk_topk)
            qk_trace_results = QKTracingResult(
                q_marginal=NodeIndexed(
                    value=[tracing_results.q_marginal.value[unique_idx] for unique_idx in slot_to_unique],
                    dimensions=(top_lorsa_features,),
                ),
                k_marginal=NodeIndexed(
                    value=[tracing_results.k_marginal.value[unique_idx] for unique_idx in slot_to_unique],
                    dimensions=(top_lorsa_features,),
                ),
                pairs=NodeIndexed(
                    value=[tracing_results.pairs.value[unique_idx] for unique_idx in slot_to_unique],
                    dimensions=(top_lorsa_features,),
                ),
            )

    last_logits = (
        batch_logits[:, -1, top_idx].detach()  # pyright: ignore[reportPossiblyUnboundVariable]
        if target_type == "logits" and top_idx.numel() > 0 and top_idx.dim() == 1  # pyright: ignore[reportPossiblyUnboundVariable]
        else None
    )
    logit_tokens = (
        [cast(Any, model.tokenizer).decode([token_id]) for token_id in logit_token_ids]
        if logit_token_ids is not None
        else None
    )

    return AttributionResult(
        activations=activations_vec,
        attribution=attribution,
        targets=targets.dimension,
        logits=last_logits,
        probs=top_p if target_type == "logits" else None,  # pyright: ignore[reportPossiblyUnboundVariable]
        prompt_token_ids=prompt_token_ids,
        prompt_tokens=[cast(Any, model.tokenizer).decode([token_id]) for token_id in prompt_token_ids],
        logit_token_ids=logit_token_ids,
        logit_tokens=logit_tokens,
        qk_trace_results=qk_trace_results,
    )


def _coerce_feature_dim(
    features: list[tuple[str, int, int]] | NodeDimension,
    device: torch.device | str,
    device_mesh: DeviceMesh | None,
) -> NodeDimension:
    if isinstance(features, NodeDimension):
        if len(features) == 0:
            raise ValueError("target_type='features' requires a non-empty NodeDimension")
        return features
    if len(features) == 0:
        raise ValueError("target_type='features' requires a non-empty features list")
    by_key: dict[str, list[tuple[int, int]]] = {}
    for key, pos, fidx in features:
        by_key.setdefault(key, []).append((int(pos), int(fidx)))
    return NodeDimension.from_node_infos(
        [NodeInfo(key=k, indices=torch.tensor(v, device=device, dtype=torch.long)) for k, v in by_key.items()],
        device=device,
        device_mesh=device_mesh,
    )


def _retrieve_qk_vector_targets(
    features: NodeDimension,
    cache: dict[str, torch.Tensor],
    ov_group_sizes: dict[str, int],
    attn_scales: dict[str, float],
) -> tuple[NodeRefs, NodeRefs, list[int]]:
    """Given Lorsa feature nodes, retrieve the QK targets (QK vectors which multiply to the attention score). Note that the QK pairs may be duplicated for different Lorsa features for OV groups larger than 1, since different OV heads share the same QK heads. This function deduplicates the QK pairs and returns the mapping from the original feature offsets to the deduplicated slot indices.

    Returns:
        A tuple of (`q_targets`, `k_targets`, `slot_to_unique`)
        - `q_targets` and `k_targets` are `NodeRefs` containing the Q and K targets respectively.
        - `slot_to_unique` is a list of ints mapping the original feature offsets to the deduplicated slot indices.
    """
    slot_triples: list[tuple[str, int, int, int]] = []
    for feature_node in features:
        sae_path = feature_node.key.replace(".sae.hook_feature_acts", "")
        attn_score_ref = cache[f"{sae_path}.sae.hook_attn_score"]
        q_pos, ov_head_idx = full_tensor(feature_node.indices)[0].tolist()
        qk_head_idx = ov_head_idx // ov_group_sizes[sae_path]
        k_pos = int(attn_score_ref[0, qk_head_idx, q_pos].argmax().item())
        slot_triples.append((sae_path, int(qk_head_idx), int(q_pos), k_pos))

    seen: dict[tuple[str, int, int, int], int] = {}
    deduped: list[tuple[str, int, int, int]] = []
    slot_to_unique: list[int] = []
    for triple in slot_triples:
        unique_idx = seen.get(triple)
        if unique_idx is None:
            unique_idx = len(deduped)
            seen[triple] = unique_idx
            deduped.append(triple)
        slot_to_unique.append(unique_idx)

    @functools.cache
    def _scaled_refs(sae_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        inv_sqrt_scale = attn_scales[sae_path] ** -0.5
        return cache[f"{sae_path}.sae.hook_q_post_rot"] * inv_sqrt_scale, cache[
            f"{sae_path}.sae.hook_k_post_rot"
        ] * inv_sqrt_scale

    q_entries: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    k_entries: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for sae_path, qk_head_idx, q_pos, k_pos in deduped:
        q_ref, k_ref = _scaled_refs(sae_path)
        device = q_ref.device
        q_entries.append(
            (
                f"{sae_path}.sae.hook_q_post_rot",
                torch.tensor([[q_pos, qk_head_idx]], device=device),
                q_ref,
            )
        )
        k_entries.append(
            (
                f"{sae_path}.sae.hook_k_post_rot",
                torch.tensor([[k_pos, qk_head_idx]], device=device),
                k_ref,
            )
        )
    q_targets = NodeRefs.from_nodes_and_refs(q_entries)
    k_targets = NodeRefs.from_nodes_and_refs(k_entries)
    return q_targets, k_targets, slot_to_unique


def qk_trace(
    model: TransformerLensLanguageModel,
    inputs: torch.Tensor | str,
    replacement_modules: list[SparseDictionary],
    lorsa_features: NodeDimension,
    topk: int = 10,
    batch_size: int = 1,
    matryoshka_feature_range: tuple[int, int] | None = None,
) -> QKTracingResult:
    tokens = ensure_tokenized(inputs, model.tokenizer, device=model.device)
    replacement_modules_cast: list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform] = cast(
        list[SparseAutoEncoder | LowRankSparseAttention | MixtureOfLinearTransform], replacement_modules
    )
    _, cache = collect_cache(
        model,
        einops.repeat(tokens, "n -> b n", b=batch_size * topk),
        replacement_modules_cast,
        with_bias_leaves=True,
        matryoshka_feature_range=matryoshka_feature_range,
    )

    seq_len = cache["hook_embed.post"].shape[1]
    pos_indices = torch.arange(seq_len, device=model.device).unsqueeze(-1)

    # sources: embed + errors + feature_acts + bias leaves
    source_entries: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("hook_embed", pos_indices, cache["hook_embed.post"]),
    ]
    for rm in replacement_modules_cast:
        hp = rm.cfg.hook_point_out
        source_entries.append((hp + ".error", pos_indices, cache[hp + ".error.post"]))
        source_entries.append(
            (
                hp + ".sae.hook_feature_acts",
                nonzero(cache[hp + ".sae.hook_feature_acts.post"][0]),
                cache[hp + ".sae.hook_feature_acts.post"],
            )
        )
    bias_keys = [
        f"blocks.{i}.{kind}" for i in range(len(model.blocks)) for kind in ("attn.hook_b_O", "mlp.hook_b_out")
    ] + [
        rm.cfg.hook_point_out + suffix
        for rm in replacement_modules_cast
        for suffix in (".sae.hook_b_Q", ".sae.hook_b_K", ".sae.hook_b_D")
    ]
    source_entries.extend((k, pos_indices, cache[k]) for k in bias_keys if k in cache)
    qk_sources = NodeRefs.from_nodes_and_refs(source_entries)

    q_targets, k_targets, slot_to_unique = _retrieve_qk_vector_targets(
        lorsa_features,
        cache,
        {
            rm.cfg.hook_point_out: rm.cfg.ov_group_size
            for rm in replacement_modules_cast
            if isinstance(rm, LowRankSparseAttention)
        },
        {
            rm.cfg.hook_point_out: rm.attn_scale
            for rm in replacement_modules_cast
            if isinstance(rm, LowRankSparseAttention)
        },
    )
    tracing_results = compute_qk_tracing(q_targets, k_targets, qk_sources, topk=topk)
    return QKTracingResult(
        q_marginal=NodeIndexed(
            value=[tracing_results.q_marginal.value[unique_idx] for unique_idx in slot_to_unique],
            dimensions=(lorsa_features,),
        ),
        k_marginal=NodeIndexed(
            value=[tracing_results.k_marginal.value[unique_idx] for unique_idx in slot_to_unique],
            dimensions=(lorsa_features,),
        ),
        pairs=NodeIndexed(
            value=[tracing_results.pairs.value[unique_idx] for unique_idx in slot_to_unique],
            dimensions=(lorsa_features,),
        ),
    )


def compute_hessian_matrix(
    targets: NodeRefs,
    sources: NodeRefs,
    topk: int,
) -> NodeIndexed[list["NodeIndexed[torch.Tensor]"]]:
    fwd_batch_size = sources.batch_size
    bwd_batch_size = fwd_batch_size // topk

    per_slot_results: list[NodeIndexed[torch.Tensor]] = []

    for target_batch in targets.iter_batches(bwd_batch_size):
        stacked = torch.cat(target_batch.values, dim=1)  # (fwd_batch, n_slots)
        bwd_scores = torch.stack(
            [stacked[i * topk : (i + 1) * topk, i] for i in range(stacked.shape[1])],
            dim=0,
        )
        first_grad_refs = torch.autograd.grad(
            bwd_scores.sum(),
            sources.refs(),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        first_attributions = attribution_scores(first_grad_refs, sources)

        second_targets = NodeDimension.empty(device=bwd_scores.device)
        all_topk_indices = []
        for slot_idx in range(stacked.shape[1]):
            _, topk_indices = first_attributions[slot_idx * topk].topk(topk)
            second_targets = second_targets + sources.dimension.offsets_to_nodes(offsets=topk_indices)
            all_topk_indices.append(topk_indices)

        second_bwd_values = first_attributions[
            torch.arange(stacked.shape[1], device=bwd_scores.device).unsqueeze(1) * topk
            + torch.arange(topk, device=bwd_scores.device).unsqueeze(0),
            torch.stack(all_topk_indices, dim=0),
        ].reshape(-1)

        second_grads_refs = torch.autograd.grad(
            second_bwd_values.sum(),
            sources.refs(),
            retain_graph=True,
            materialize_grads=True,
        )
        second_attributions = attribution_scores(second_grads_refs, sources)
        second_targets_list = list(second_targets)
        diag = torch.arange(topk, device=second_attributions.device)
        for slot_idx in range(stacked.shape[1]):
            topk_values, topk_indices = second_attributions[slot_idx * topk : (slot_idx + 1) * topk].topk(topk, dim=-1)
            pair_attrs = topk_values[diag, diag]
            second_nodes = list(sources.dimension.offsets_to_nodes(topk_indices[diag, diag]))
            first_nodes = second_targets_list[slot_idx * topk : (slot_idx + 1) * topk]

            order = torch.argsort(pair_attrs, descending=True).tolist()
            order = [i for i in order if pair_attrs[i].item() != 0.0]
            per_slot_results.append(
                NodeIndexed(
                    value=pair_attrs[order] if len(order) > 0 else pair_attrs[:0],
                    dimensions=(
                        NodeDimension.from_node_infos([first_nodes[i] for i in order], device=pair_attrs.device),
                        NodeDimension.from_node_infos([second_nodes[i] for i in order], device=pair_attrs.device),
                    ),
                )
            )

    return NodeIndexed(value=per_slot_results, dimensions=(targets.dimension,))


@timer.time("_extract_topk_pairwise_attributions")
def _extract_topk_pairwise_attributions(
    q_side: NodeIndexedMatrix,  # (q_picks, sources)
    k_side: NodeIndexedMatrix,  # (k_picks, sources)
    topk: int,
) -> NodeIndexed[torch.Tensor]:
    """Extract topk, deduped, non-zero pairwise attributions."""
    # Dedup: zero out K-side columns whose source is also a Q-pick.
    if len(k_side.dimensions[0]) > 0 and len(q_side.dimensions[0]) > 0:
        k_side = k_side.clone()
        k_side[None, q_side.dimensions[0]] = 0

    q_top = q_side.flat_topk(topk)
    k_top = k_side.flat_topk(topk)

    merged_values = torch.cat([q_top.value, k_top.value])
    merged_q_role = q_top.dimensions[0] + k_top.dimensions[1]
    merged_k_role = q_top.dimensions[1] + k_top.dimensions[0]

    nonzero_count = int((merged_values != 0.0).sum().item())
    n_final = min(topk, nonzero_count)
    top_values, top_offsets = torch.topk(merged_values, n_final)
    return NodeIndexed(
        value=top_values,
        dimensions=(
            merged_q_role.offsets_to_nodes(top_offsets),
            merged_k_role.offsets_to_nodes(top_offsets),
        ),
    )


@timer.time("compute_qk_tracing")
def compute_qk_tracing(
    q_targets: NodeRefs,
    k_targets: NodeRefs,
    sources: NodeRefs,
    topk: int,
) -> QKTracingResult:
    """Role-labeled QK tracing for attention-score targets.

    `q_targets` and `k_targets` are pre-attention-score vectors. They must hold the following properties:
    - Having the same length;
    - Multiplying to directly produce attention scores.
        - This means if extra steps exist in model forward between Q/K and attention score (e.g. rotary embedding, layernorms, attention score scaling), these steps must be applied and pass the final version to `q_targets` and `k_targets`.
    """
    assert len(q_targets.dimension) == len(k_targets.dimension)
    bwd_batch_size = sources.batch_size // topk
    n_slots_total = len(q_targets.dimension)
    device_mesh = sources.dimension.device_mesh
    tp_pair_skipped = False

    @timer.time("vjp_attribution")
    def vjp_attribution(outputs: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        with timer.time("vjp_grad"):
            grads = torch.autograd.grad(
                outputs,
                sources.refs(),
                grad_outputs=cotangent,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
        return attribution_scores(grads, sources)

    def replicate(t: torch.Tensor) -> torch.Tensor:
        return t if device_mesh is None or isinstance(t, DTensor) else DimMap({}).from_local(t, device_mesh)

    def topk_nonzero(scores: torch.Tensor) -> NodeIndexed[torch.Tensor]:
        values, indices = scores.detach().topk(topk)
        keep = values != 0.0
        return NodeIndexed(
            value=values[keep],
            dimensions=(sources.dimension.offsets_to_nodes(replicate(indices[keep])),),
        )

    q_marginal_slots: list[NodeIndexed[torch.Tensor]] = []
    k_marginal_slots: list[NodeIndexed[torch.Tensor]] = []
    pair_slots: list[NodeIndexed[torch.Tensor]] = []

    n_batches = (n_slots_total + bwd_batch_size - 1) // bwd_batch_size
    batch_iter = zip(q_targets.iter_batches(bwd_batch_size), k_targets.iter_batches(bwd_batch_size))
    for q_sub, k_sub in tqdm(batch_iter, total=n_batches, desc="QK Tracing", disable=n_batches <= 1):
        # `topk`-sized slots for second-order attributions: first-order grads
        # are replicated `topk` times for the same reason `collect_cache`
        # replicates `batch_size` times in forward.
        with timer.time("qk_prep"):
            q_stacked = full_tensor(torch.cat(q_sub.values, dim=1))
            k_stacked = full_tensor(torch.cat(k_sub.values, dim=1))
            n_slots = q_stacked.shape[1]
            q_vecs = torch.stack([q_stacked[s * topk : (s + 1) * topk, s] for s in range(n_slots)], dim=0)
            k_vecs = torch.stack([k_stacked[s * topk : (s + 1) * topk, s] for s in range(n_slots)], dim=0)

        a_Q = vjp_attribution(q_vecs, k_vecs)
        a_K = vjp_attribution(k_vecs, q_vecs)
        n_sources = a_Q.shape[1]

        with timer.time("qk_pick"):
            slot_first_rows = torch.arange(n_slots, device=a_Q.device) * topk
            bwd_rows = slot_first_rows.unsqueeze(1) + torch.arange(topk, device=a_Q.device).unsqueeze(0)
            merged = torch.cat(
                [full_tensor(a_Q[slot_first_rows].detach()), full_tensor(a_K[slot_first_rows].detach())], dim=-1
            )
            _, pick_flat = merged.topk(topk, dim=-1)
            is_q_pick = pick_flat < n_sources
            pick_src = torch.where(is_q_pick, pick_flat, pick_flat - n_sources)

        # Pair extraction requires double-backward through the SAE forward.
        # DTensor autograd doesn't support `create_graph=True`.

        if device_mesh is not None:
            if not tp_pair_skipped:
                logger.warning(
                    "compute_qk_tracing: pair extraction is disabled under "
                    "tensor-parallel execution because DTensor does not "
                    "support double-backward through SAE forward ops. Q/K "
                    "marginals remain valid."
                )
                tp_pair_skipped = True
            empty_dim = NodeDimension.empty(device=a_Q.device)
            pair_slots.extend(
                NodeIndexed(
                    value=torch.empty(0, device=a_Q.device, dtype=a_Q.dtype),
                    dimensions=(empty_dim, empty_dim),
                )
                for _ in range(n_slots)
            )
            a_Q = full_tensor(a_Q.detach())
            a_K = full_tensor(a_K.detach())
            q_marginal_slots.extend(topk_nonzero(a_Q[s * topk]) for s in range(n_slots))
            k_marginal_slots.extend(topk_nonzero(a_K[s * topk]) for s in range(n_slots))
        else:
            with timer.time("second_grad"):
                second_bwd_values = torch.where(is_q_pick, a_Q[bwd_rows, pick_src], a_K[bwd_rows, pick_src]).reshape(-1)
                with timer.time("grad"):
                    grads = torch.autograd.grad(
                        second_bwd_values.sum(),
                        sources.refs(),
                        retain_graph=True,
                        materialize_grads=True,
                    )
                second_attributions = attribution_scores(grads, sources).detach()
            a_Q = a_Q.detach()
            a_K = a_K.detach()
            q_marginal_slots.extend(topk_nonzero(a_Q[s * topk]) for s in range(n_slots))
            k_marginal_slots.extend(topk_nonzero(a_K[s * topk]) for s in range(n_slots))

            def slot_pair_attribution(s: int) -> NodeIndexed[torch.Tensor]:
                slot_rows = second_attributions[s * topk : (s + 1) * topk]
                q_mask = is_q_pick[s]
                k_mask = ~q_mask
                q_side = NodeIndexedMatrix.from_data(
                    data=slot_rows[q_mask],
                    dimensions=(sources.dimension.offsets_to_nodes(pick_src[s][q_mask]), sources.dimension),
                )
                k_side = NodeIndexedMatrix.from_data(
                    data=slot_rows[k_mask],
                    dimensions=(sources.dimension.offsets_to_nodes(pick_src[s][k_mask]), sources.dimension),
                )
                return _extract_topk_pairwise_attributions(q_side, k_side, topk=topk)

            pair_slots.extend(slot_pair_attribution(s) for s in range(n_slots))
            del second_attributions

        del q_stacked, k_stacked, q_vecs, k_vecs, a_Q, a_K, merged, pick_flat, bwd_rows
        torch.cuda.empty_cache()

    outer = (q_targets.dimension,)
    return QKTracingResult(
        q_marginal=NodeIndexed(value=q_marginal_slots, dimensions=outer),
        k_marginal=NodeIndexed(value=k_marginal_slots, dimensions=outer),
        pairs=NodeIndexed(value=pair_slots, dimensions=outer),
    )
