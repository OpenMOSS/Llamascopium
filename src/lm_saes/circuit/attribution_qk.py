"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import contextlib
import logging
import os
import time
import weakref
from collections import deque
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union
from lm_saes.sae import SparseAutoEncoder
from lm_saes.lorsa import LowRankSparseAttention

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from .graph_lc0 import Graph, compute_graph_scores
from .replacement_lc0_model import ReplacementModel
from .utils.disk_offload import offload_modules
from .utils.create_graph_files import create_graph_files

from ..utils.logging import get_distributed_logger

from .leela_board import *

logger = get_distributed_logger("attribution")


class FeatureTraceSpec(TypedDict, total=False):
    """Minimum metadata required to trace from a specific feature."""

    layer: int
    feature_idx: int
    position: int
    type: Literal["tc", "lorsa"]

class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
    """

    def __init__(
        self,
        lorsa_activation_matrix: torch.sparse.Tensor,
        tc_activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor, 
        lorsa_decoder_vecs: torch.Tensor,
        tc_decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        mlp_output_hook: str,
    ) -> None:
        # assert lorsa_activation_matrix.shape[:-1] == tc_activation_matrix.shape[:-1], "LORSAs and TCs must have the same shape"
        n_layers, n_pos, _ = tc_activation_matrix.shape # tc_activation_matrix.shape = torch.Size([15, 64, 12288])
        # Forward-pass cache
        # # L0Ainput, L0Minput, ... L-1Ainput, L-1Minput, pre_unembed
        # L0Minput, L1Minput, ... L-1Minput, policy_head_input for transcoder only tracing
        self._resid_activations: List[torch.Tensor | None] = [None] * (2 * n_layers + 1)
        # add policy head's q and k activations cache
        self._policy_q_activations: torch.Tensor | None = None
        self._policy_k_activations: torch.Tensor | None = None
        self._embed_activation: torch.Tensor | None = None
        self._attn_output_activations: List[torch.Tensor | None] = [None] * n_layers
        self._mlp_output_activations: List[torch.Tensor | None] = [None] * n_layers
        # (row_size, batch_size, 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers
        self.n_pos: int = n_pos
        self._lorsa_activation_matrix = lorsa_activation_matrix
        self._tc_activation_matrix = tc_activation_matrix
        self._error_vectors = error_vectors
        self._token_vectors = token_vectors
        self._lorsa_decoder_vecs = lorsa_decoder_vecs
        self._tc_decoder_vecs = tc_decoder_vecs
        self._attn_output_hook = attn_output_hook
        self._mlp_output_hook = mlp_output_hook

        # The fast path uses live tensor references with ``autograd.grad``.
        # Legacy hook closures are created only if ``install_hooks`` is called.
        self._attribution_hooks: List[Tuple[str, Callable]] | None = None
        
        total_active_feats = lorsa_activation_matrix._nnz() + tc_activation_matrix._nnz()
        # total_active_feats + error_vectors + token_vectors
        self._row_size: int = total_active_feats + 2 * n_layers * n_pos + n_pos  # + logits later

    # def _caching_hooks(self, attn_input_hook: str, mlp_input_hook: str) -> List[Tuple[str, Callable]]:
    #     """Return hooks that store residual activations layer-by-layer."""

    #     proxy = weakref.proxy(self)

    #     def _cache(acts: torch.Tensor, hook: HookPoint, *, index: int) -> torch.Tensor:
    #         proxy._resid_activations[index] = acts
    #         return acts

    #     hooks = []
    #     for layer in range(self.n_layers):
    #         hooks.append((f"blocks.{layer}.{attn_input_hook}", partial(_cache, index=layer * 2)))
    #         hooks.append((f"blocks.{layer}.{mlp_input_hook}", partial(_cache, index=layer * 2 + 1)))
    #     hooks.append(("unembed.hook_pre", partial(_cache, index=2 * self.n_layers)))

    #     return hooks

    def _caching_hooks(self, attn_input_hook: str, mlp_input_hook: str) -> List[Tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, index: int) -> torch.Tensor:
            proxy._resid_activations[index] = acts
            return acts

        def _cache_q(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            proxy._policy_q_activations = acts
            return acts

        def _cache_k(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            proxy._policy_k_activations = acts
            return acts

        hooks = []
        for layer in range(self.n_layers):
            hooks.append((f"blocks.{layer}.{attn_input_hook}", partial(_cache, index=layer * 2)))
            hooks.append((f"blocks.{layer}.{mlp_input_hook}", partial(_cache, index=layer * 2 + 1)))
        
        hooks.append(("policy_head.hook_pre", partial(_cache, index=2 * self.n_layers)))
        # add policy head's q and k cache hooks
        hooks.append(("policy_head.hook_q", _cache_q))
        hooks.append(("policy_head.hook_k", _cache_k))

        return hooks

    def _live_caching_hooks(
        self,
        attn_input_hook: str,
        mlp_input_hook: str,
    ) -> List[Tuple[str, Callable]]:
        """Cache live tensors used by batched VJPs without backward hooks."""

        hooks = self._caching_hooks(attn_input_hook, mlp_input_hook)
        proxy = weakref.proxy(self)

        def _cache_embed(acts: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            if not acts.requires_grad:
                acts = acts.detach().requires_grad_()
            proxy._embed_activation = acts
            return acts

        def _cache_attn(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            if not acts.requires_grad:
                acts = acts.detach().requires_grad_()
            proxy._attn_output_activations[layer] = acts
            return acts

        def _cache_mlp(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            if not acts.requires_grad:
                acts = acts.detach().requires_grad_()
            proxy._mlp_output_activations[layer] = acts
            return acts

        hooks.append(("hook_embed", _cache_embed))
        for layer in range(self.n_layers):
            hooks.append(
                (f"blocks.{layer}.{self._attn_output_hook}", partial(_cache_attn, layer=layer))
            )
            hooks.append(
                (f"blocks.{layer}.{self._mlp_output_hook}", partial(_cache_mlp, layer=layer))
            )
        return hooks

    @contextlib.contextmanager
    def install_live_hooks(self, model: "ReplacementModel"):
        """Install forward-only hooks for the autograd.grad attribution path."""

        with model.hooks(fwd_hooks=self._live_caching_hooks(model.attn_input_hook, model.mlp_input_hook)):
            yield

    def _source_refs(self) -> List[torch.Tensor]:
        if self._embed_activation is None:
            raise RuntimeError("Embedding activation was not cached")
        refs = [self._embed_activation]
        for layer in range(self.n_layers):
            attn_ref = self._attn_output_activations[layer]
            mlp_ref = self._mlp_output_activations[layer]
            if attn_ref is None or mlp_ref is None:
                raise RuntimeError(f"Output activations were not cached for layer {layer}")
            refs.extend((attn_ref, mlp_ref))
        return refs

    @staticmethod
    def _layer_spans(activation_matrix: torch.Tensor) -> List[Tuple[int, int]]:
        layers = activation_matrix.indices()[0]
        if layers.numel() == 0:
            return [(0, 0)] * activation_matrix.shape[0]
        counts = torch.bincount(layers, minlength=activation_matrix.shape[0])
        edges = torch.cat((counts.new_zeros(1), counts.cumsum(0))).tolist()
        return list(zip(edges[:-1], edges[1:]))

    def _rows_from_root(self, root: torch.Tensor, *, retain_graph: bool) -> torch.Tensor:
        """Return source input-times-gradient rows for independent batch lanes."""

        refs = self._source_refs()
        grads = torch.autograd.grad(
            root,
            refs,
            retain_graph=retain_graph,
            allow_unused=True,
            materialize_grads=True,
        )
        batch_size = refs[0].shape[0]
        dtype = self._lorsa_decoder_vecs.dtype
        device = refs[0].device
        rows = torch.zeros(batch_size, self._row_size, dtype=dtype, device=device)

        embed_grad = grads[0].to(dtype)
        token_offset = (
            self._lorsa_activation_matrix._nnz()
            + self._tc_activation_matrix._nnz()
            + 2 * self.n_layers * self.n_pos
        )
        rows[:, token_offset : token_offset + self.n_pos] = torch.einsum(
            "bpd,pd->bp", embed_grad, self._token_vectors.to(device=device, dtype=dtype)
        )

        lorsa_positions = self._lorsa_activation_matrix.indices()[1]
        tc_positions = self._tc_activation_matrix.indices()[1]
        lorsa_spans = self._layer_spans(self._lorsa_activation_matrix)
        tc_spans = self._layer_spans(self._tc_activation_matrix)
        n_lorsa = self._lorsa_activation_matrix._nnz()
        n_tc = self._tc_activation_matrix._nnz()

        def contract_features(
            grad: torch.Tensor,
            positions: torch.Tensor,
            vectors: torch.Tensor,
            start: int,
            end: int,
        ) -> torch.Tensor:
            """Contract feature sources without a full ``[batch, features, d]`` temporary."""

            count = end - start
            result = rows.new_empty((batch_size, count))
            # At BT4 dimensions, 128 features caps the FP32 gather temporary at
            # 32 MiB for a 64-lane VJP instead of several hundred MiB per layer.
            feature_chunk_size = 128
            for local_start in range(0, count, feature_chunk_size):
                local_end = min(local_start + feature_chunk_size, count)
                source_slice = slice(start + local_start, start + local_end)
                selected = grad.index_select(1, positions[source_slice])
                result[:, local_start:local_end] = torch.einsum(
                    "bnd,nd->bn", selected, vectors[source_slice]
                )
            return result

        for layer in range(self.n_layers):
            attn_grad = grads[1 + 2 * layer].to(dtype)
            mlp_grad = grads[2 + 2 * layer].to(dtype)

            start, end = lorsa_spans[layer]
            if end > start:
                rows[:, start:end] = contract_features(
                    attn_grad,
                    lorsa_positions,
                    self._lorsa_decoder_vecs,
                    start,
                    end,
                )
            attn_error_offset = n_lorsa + n_tc + layer * self.n_pos
            rows[:, attn_error_offset : attn_error_offset + self.n_pos] = torch.einsum(
                "bpd,pd->bp", attn_grad, self._error_vectors[layer].to(dtype)
            )

            start, end = tc_spans[layer]
            if end > start:
                rows[:, n_lorsa + start : n_lorsa + end] = contract_features(
                    mlp_grad,
                    tc_positions,
                    self._tc_decoder_vecs,
                    start,
                    end,
                )
            mlp_error_offset = n_lorsa + n_tc + self.n_layers * self.n_pos + layer * self.n_pos
            rows[:, mlp_error_offset : mlp_error_offset + self.n_pos] = torch.einsum(
                "bpd,pd->bp", mlp_grad, self._error_vectors[self.n_layers + layer].to(dtype)
            )

        return rows

    def compute_vjp_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        attention_patterns: torch.Tensor | None = None,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Compute feature rows in parallel using one scalar-root VJP."""

        n_targets = layers.numel()
        if n_targets == 0:
            return inject_values.new_zeros((0, self._row_size))
        if n_targets > self._resid_activations[0].shape[0]:
            raise ValueError(
                f"VJP batch has {n_targets} targets but the replicated forward has "
                f"only {self._resid_activations[0].shape[0]} lanes"
            )

        root = inject_values.new_zeros(())
        batch_indices = torch.arange(n_targets, device=inject_values.device)
        for layer in torch.unique(layers).tolist():
            mask = layers == layer
            lanes = batch_indices[mask]
            activation = self._resid_activations[int(layer)]
            if activation is None:
                raise RuntimeError(f"Residual activation {layer} was not cached")
            if attention_patterns is None:
                target_values = activation[lanes, positions[mask]]
                root = root + (target_values * inject_values[mask]).sum()
            else:
                distributed = inject_values[mask, None, :] * attention_patterns[mask, :, None]
                root = root + (activation.index_select(0, lanes) * distributed).sum()
        return self._rows_from_root(root, retain_graph=retain_graph)[:n_targets]

    def compute_qk_vjp_batch(
        self,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        q_values: torch.Tensor,
        k_values: torch.Tensor,
        *,
        castle_tensor: torch.Tensor | None = None,
        retain_graph: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute independent Q and K logit rows in one autograd traversal."""

        if self._policy_q_activations is None or self._policy_k_activations is None:
            raise RuntimeError("Policy Q/K activations were not cached")
        n_targets = q_values.shape[0]
        if 2 * n_targets > self._policy_q_activations.shape[0]:
            raise ValueError("Replicated forward needs at least two lanes per Q/K logit target")

        device = self._policy_q_activations.device
        q_positions = q_positions.to(device=device, dtype=torch.long).reshape(n_targets, -1)
        k_positions = k_positions.to(device=device, dtype=torch.long).reshape(n_targets, -1)
        if castle_tensor is not None:
            castle = castle_tensor.to(device=device, dtype=torch.bool).reshape(-1)
            end = k_positions[:, 0]
            row, col = torch.div(end, 8, rounding_mode="floor"), end.remainder(8)
            end = torch.where(castle & (col == 6), row * 8 + 7, end)
            end = torch.where(castle & (col == 2), row * 8, end)
            k_positions = k_positions.clone()
            k_positions[:, 0] = end

        q_lanes = torch.arange(n_targets, device=device)
        k_lanes = q_lanes + n_targets
        q_root = q_values.new_zeros(())
        k_root = k_values.new_zeros(())
        for column in range(q_positions.shape[1]):
            pos = q_positions[:, column]
            q_term = q_values[:, column] if q_values.ndim == 4 else q_values
            q_root = q_root + (
                self._policy_q_activations[q_lanes, pos] * q_term[q_lanes, pos]
            ).sum()
        for column in range(k_positions.shape[1]):
            pos = k_positions[:, column]
            k_term = k_values[:, column] if k_values.ndim == 4 else k_values
            k_root = k_root + (
                self._policy_k_activations[k_lanes, pos] * k_term[q_lanes, pos]
            ).sum()

        rows = self._rows_from_root(q_root + k_root, retain_graph=retain_graph)
        return rows[:n_targets], rows[n_targets : 2 * n_targets]

    def compute_policy_qk_gradients(
        self,
        policy_logits: torch.Tensor,
        positive_indices: torch.Tensor | None,
        negative_indices: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Get detached policy-logit gradients with respect to Q and K in one VJP."""

        if self._policy_q_activations is None or self._policy_k_activations is None:
            raise RuntimeError("Policy Q/K activations were not cached")
        if positive_indices is None and negative_indices is None:
            raise ValueError("At least one policy-logit index tensor is required")
        root = policy_logits.new_zeros(())
        lane_offset = 0
        positive_lanes = None
        negative_lanes = None
        if positive_indices is not None:
            positive = positive_indices.to(device=policy_logits.device, dtype=torch.long).reshape(-1)
            positive_lanes = torch.arange(positive.numel(), device=policy_logits.device)
            root = root + policy_logits[positive_lanes, positive].sum()
            lane_offset = positive.numel()
        if negative_indices is not None:
            negative = negative_indices.to(device=policy_logits.device, dtype=torch.long).reshape(-1)
            negative_lanes = torch.arange(negative.numel(), device=policy_logits.device) + lane_offset
            root = root - policy_logits[negative_lanes, negative].sum()
        q_grad, k_grad = torch.autograd.grad(
            root,
            (self._policy_q_activations, self._policy_k_activations),
            retain_graph=True,
        )
        return (
            q_grad.index_select(0, positive_lanes).detach() if positive_lanes is not None else None,
            k_grad.index_select(0, positive_lanes).detach() if positive_lanes is not None else None,
            q_grad.index_select(0, negative_lanes).detach() if negative_lanes is not None else None,
            k_grad.index_select(0, negative_lanes).detach() if negative_lanes is not None else None,
        )


    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> Tuple[str, Callable]:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and writes the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            grads_read = grads[read_index]
            if grads_read.dtype != output_vecs.dtype:
                grads_read = grads_read.to(output_vecs.dtype)
            result = torch.einsum("bpd,pd->pb", grads_read, output_vecs)
            proxy._batch_buffer[write_index] += result


        return hook_name, _hook_fn


    def _make_attribution_hooks(
        self,
        lorsa_activation_matrix: torch.sparse.Tensor,
        tc_activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        lorsa_decoder_vecs: torch.Tensor,
        tc_decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        mlp_output_hook: str,
    ) -> List[Tuple[str, Callable]]:
        """
        Create the complete backward-hook for computing attribution scores.
        """
        _, n_pos, _ = tc_activation_matrix.shape
        lorsa_error_vectors = error_vectors[:self.n_layers]
        tc_error_vectors = error_vectors[self.n_layers:]
        # Token-embedding nodes
        # lorsa_offset + tc_offset + mlp_error_offset + lorsa_error_offset
        token_offset = lorsa_activation_matrix._nnz() + tc_activation_matrix._nnz() + 2 * self.n_layers * n_pos
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                token_vectors,
                write_index=np.s_[token_offset : token_offset + n_pos],
            )
        ]
        return token_hook + self._make_attribution_hooks_lorsa( 
            lorsa_activation_matrix,
            lorsa_error_vectors,
            lorsa_decoder_vecs,
            attn_output_hook,
            tc_offset=tc_activation_matrix._nnz() 
        ) + self._make_attribution_hooks_tc(
            tc_activation_matrix,
            tc_error_vectors,
            tc_decoder_vecs,
            mlp_output_hook,
            lorsa_offset=lorsa_activation_matrix._nnz()  # TC starts from the end of Lorsa
        )

    def _make_attribution_hooks_lorsa(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        attn_output_hook: str,
        tc_offset: int,
    ) -> List[Tuple[str, Callable]]:
        """
        Create the complete backward-hook for computing attribution scores.
        activation_matrix:
            size (n_layers, n_pos, n_features)
            indices: (3, n_active_features)
            values: (n_active_features,)
        error_vectors:
            size (n_layers, n_pos, d_model)
        token_vectors:
            size (n_pos, d_model)
        decoder_vecs:
            size (n_active_features, d_model)
        """
        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))
        
        
        # Feature nodes
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{attn_output_hook}",
                decoder_vecs[start:end],
                write_index=np.s_[start:end],
                read_index=np.s_[:, nnz_positions[start:end]],
            )
            for layer, (start, end) in enumerate(layer_spans)
            if start != end
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return activation_matrix._nnz() + tc_offset + layer * n_pos
        
        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{attn_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        return feature_hooks + error_hooks

    def _make_attribution_hooks_tc(
        self,
        activation_matrix: torch.sparse.Tensor,
        error_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        mlp_output_hook: str,
        lorsa_offset: int,
    ) -> List[Tuple[str, Callable]]:
        """
        Create attribution hooks for single-layer transcoders.
        activation_matrix:
            size (n_layers, n_pos, n_features)
            indices: (3, n_active_features)
            values: (n_active_features,)
        error_vectors:
            size (n_layers, n_pos, d_model)
        decoder_vecs:
            size (n_active_features, d_model) - for single-layer transcoders
        """
        n_layers, n_pos, _ = activation_matrix.shape
        nnz_layers, nnz_positions, _ = activation_matrix.indices()

        # Map each layer → slice in flattened active-feature list
        if activation_matrix._nnz() == 0:
            return []  # Return empty list if no features are active
            
        _, counts = torch.unique_consecutive(nnz_layers, return_counts=True)
        edges = [0] + counts.cumsum(0).tolist()
        layer_spans = list(zip(edges[:-1], edges[1:]))
        

        # Simple assertion: decoder_vecs should match total active features
        assert edges[-1] == activation_matrix._nnz(), f'got {edges[-1]} but expected {activation_matrix._nnz()}'
        assert decoder_vecs.size(0) == activation_matrix._nnz(), f'got {decoder_vecs.size(0)} but expected {activation_matrix._nnz()}'

        # Feature nodes
        feature_hooks = []
        for layer, (start, end) in enumerate(layer_spans):
            if start != end:
                hook = self._compute_score_hook(
                    f"blocks.{layer}.{mlp_output_hook}",
                    decoder_vecs[start:end],
                    write_index=np.s_[lorsa_offset+start:lorsa_offset+end],
                    read_index=np.s_[:, nnz_positions[start:end]],
                )
                feature_hooks.append(hook)

        # Error nodes
        def error_offset(layer: int) -> int:
            # lorsa_offset + tc_offset + attn_error_offset + layer_offset
            return lorsa_offset + activation_matrix._nnz() + self.n_layers * n_pos + layer * n_pos

        error_hooks = []
        for layer in range(n_layers):
            hook = self._compute_score_hook(
                f"blocks.{layer}.{mlp_output_hook}",
                error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            error_hooks.append(hook)

        return feature_hooks + error_hooks

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        if self._attribution_hooks is None:
            self._attribution_hooks = self._make_attribution_hooks(
                self._lorsa_activation_matrix,
                self._tc_activation_matrix,
                self._error_vectors,
                self._token_vectors,
                self._lorsa_decoder_vecs,
                self._tc_decoder_vecs,
                self._attn_output_hook,
                self._mlp_output_hook,
            )
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.attn_input_hook, model.mlp_input_hook),
            bwd_hooks=self._attribution_hooks,
        ):
            yield


    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        attention_patterns: torch.Tensor | None = None,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """
        for resid_activation in self._resid_activations:
            assert resid_activation is not None, "Residual activations are not cached"

        batch_size = self._resid_activations[0].shape[0]
        # print(f"DEBUG: batch_size = {batch_size}") [1]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)
        
        
        def _inject(grads, *, batch_indices, pos_indices, patterns, values):
            if batch_indices.max() >= grads.shape[0]:
                raise IndexError(f"Batch indices max ({batch_indices.max()}) >= grads batch size ({grads.shape[0]})")
            if pos_indices.max() >= grads.shape[1]:
                raise IndexError(f"Position indices max ({pos_indices.max()}) >= grads seq length ({grads.shape[1]})")
            
            grads_out = grads.clone().to(values.dtype)
            
            if patterns is not None:
                if patterns.shape[1] > grads.shape[1]:
                    raise IndexError(f"Patterns seq_len ({patterns.shape[1]}) > grads seq_len ({grads.shape[1]})")
                
                distributed_values = values[:, None, :] * patterns[:, :, None]
                grads_out.index_put_((batch_indices,), distributed_values)
            else:
                grads_out.index_put_((batch_indices, pos_indices), values)
            
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()
        # print(f'{layers_in_batch = }')
        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            
            if int(layer) >= len(self._resid_activations):
                raise IndexError(f"Layer {layer} out of range")
            
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                patterns=attention_patterns[mask] if attention_patterns is not None else None,
                values=inject_values[mask],
            )
            # print(f"{len(self._resid_activations) = }")
            handles.append(self._resid_activations[int(layer)].register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            gradient = torch.zeros_like(self._resid_activations[last_layer])
            
            self._resid_activations[last_layer].backward(
                gradient=gradient,
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        result = buf.T[: len(layers)]
        return result


    def compute_start_end_batch_from_q(
        self,
        move_positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for start positions of moves in single backward pass, starting from policy head q.

        This function performs one backward pass with gradients injected at
        start positions only for each move, starting the backward
        pass from the cached policy head q activations instead of the last layer.

        Args:
            layers: 1-D tensor of layer indices for the source nodes.
            move_positions: `(batch, 1)` tensor where move_positions[i] is start pos
                for the i-th move.
            inject_values: `(batch, seq_len, d_model)` tensor where we extract
                start position values for injection.

        Returns:
            torch.Tensor: `(batch, row_size)` matrix where each row corresponds to 
                the attribution of one start position.
        """
        
        for resid_activation in self._resid_activations:
            assert resid_activation is not None, "Residual activations are not cached"
        
        assert self._policy_q_activations is not None, "Policy head q activations are not cached"

        # Detach policy head k activations to isolate q tracing
        def detach_k_hook(acts, hook):
            """Detach k activations to prevent gradient flow"""
            return acts.detach()
        
        # # add detach k hook
        # k_detach_handle = None
        # if self._policy_k_activations is not None and hasattr(self._policy_k_activations, 'grad'):
        #     # if k activations exist, detach it directly
        #     if self._policy_k_activations.requires_grad:
        #         self._policy_k_activations = self._policy_k_activations.detach()
        #         print("DEBUG: Detached policy head k activations")

        k_batch = move_positions.shape[0]
        device = inject_values.device
        
        # Ensure all tensors are on the same device
        start_pos = move_positions.to(dtype=torch.long, device=device)

            
        batch_size = self._policy_q_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=device,
        )

        # batch indices correspond to grads batch dim; grads batch size is 1 here
        batch_idx = torch.arange(len(start_pos), device=start_pos.device)

        def _inject_start_only(grads, *, batch_indices, start_positions, start_values):
            """Inject gradients only at start positions"""
            grads_out = grads.clone().to(start_values.dtype)
            
            # Only inject start positions, other positions remain 0
            grads_out.index_put_((batch_indices, start_positions), start_values)
            
            return grads_out.to(grads.dtype)

        handles = []

        layer_start_inject = (
            inject_values[batch_idx, start_pos]
            if k_batch > 0
            else torch.empty(0, inject_values.shape[-1], device=device)
        )
        

        if layer_start_inject.shape[0] > 0:  # Only register if there are items
            fn = partial(
                _inject_start_only,
                batch_indices=batch_idx,  # all batch indices
                start_positions=start_pos,  # all start positions
                start_values=layer_start_inject,  # all injection values
            )
            handles.append(self._policy_q_activations.register_hook(fn))
        
        try:
            self._policy_q_activations.backward(
                gradient=torch.zeros_like(self._policy_q_activations),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[:k_batch]  # Return k_batch rows, one per start position

    def compute_start_end_batch_from_k(
        self,
        move_positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
        castle_tensor: torch.Tensor = None,
    ) -> torch.Tensor:
        """Return attribution rows for end positions of moves in single backward pass, starting from policy head k.

        This function performs one backward pass with gradients injected at
        end positions only for each move, starting the backward
        pass from the cached policy head k activations instead of the last layer.

        Args:
            move_positions: `(batch, 1)` tensor where move_positions[i] is end pos
                for the i-th move.
            inject_values: `(batch, seq_len, d_model)` tensor where we extract
                end position values for injection.
            castle_tensor: `(batch,)` bool tensor indicating which moves are castling moves.
                If None, will auto-detect castling moves.

        Returns:
            torch.Tensor: `(batch, row_size)` matrix where each row corresponds to 
                the attribution of one end position.
        """
        
        for resid_activation in self._resid_activations:
            assert resid_activation is not None, "Residual activations are not cached"
        
        assert self._policy_k_activations is not None, "Policy head k activations are not cached"

        k_batch = move_positions.shape[0]
        device = inject_values.device

        if castle_tensor is None:
            castle_tensor = torch.zeros(k_batch, dtype=torch.bool, device=device)
        else:
            castle_tensor = castle_tensor.to(device=device, dtype=torch.bool)
    
        end_pos = move_positions.to(dtype=torch.long, device=device)
        end_row = torch.div(end_pos, 8, rounding_mode="floor")
        end_col = end_pos.remainder(8)
        adjusted_end_pos = end_pos.clone()
        castle_short = castle_tensor & (end_col == 6)
        castle_long = castle_tensor & (end_col == 2)
        adjusted_end_pos[castle_short] = end_row[castle_short] * 8 + 7
        adjusted_end_pos[castle_long] = end_row[castle_long] * 8
        invalid_castle = castle_tensor & ~(castle_short | castle_long)
        if invalid_castle.any():
            logger.warning(
                "Found %d castle move(s) with non-castling end squares in K tracing",
                int(invalid_castle.sum().item()),
            )

        batch_size = self._policy_k_activations[0].shape[0]
        # print(f"DEBUG: batch_size = {batch_size}")
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=device,
        )
        # batch indices correspond to grads batch dim; grads batch size is 1 here
        batch_idx = torch.arange(len(adjusted_end_pos), device=adjusted_end_pos.device)

        def _inject_end_only(grads, *, batch_indices, end_positions, end_values):
            """Inject gradients only at end positions"""
            grads_out = grads.clone().to(end_values.dtype)

            grads_out.index_put_((batch_indices, end_positions), end_values)

            return grads_out.to(grads.dtype)
        handles = []
        layer_end_inject = (
            inject_values[batch_idx, adjusted_end_pos]
            if k_batch > 0
            else torch.empty(0, inject_values.shape[-1], device=device)
        )
        
        if layer_end_inject.shape[0] > 0:  # Only register if there are items
            fn = partial(
                _inject_end_only,
                batch_indices=batch_idx,  # all batch indices
                end_positions=adjusted_end_pos,  # Use adjusted positions
                end_values=layer_end_inject,  # all injection values
            )

            handles.append(self._policy_k_activations.register_hook(fn))
        
        try:
            self._policy_k_activations.backward(
                gradient=torch.zeros_like(self._policy_k_activations),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[:k_batch]  # Return k_batch rows, one per end position

def compute_logit_gradients_wrt_q(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = True,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    compute gradients of policy logits with respect to q activations
    """
    
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")
    
    lboard = LeelaBoard.from_fen(fen)
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    
    if logits.dim() > 1:
        logits = logits.flatten()
    
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} out of logits range [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        actual_max_logits = min(max_n_logits, logits.size(0))
        
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, actual_max_logits)
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]
    
    move_positions = []
    for idx in top_idx:
        try:
            uci_move = lboard.idx2uci(idx.item())
            positions = lboard.uci_to_positions(uci_move)
            move_positions.append(positions)
        except Exception as e:
            logger.warning(f"cannot get move position for index {idx.item()}: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)
    
    device = residual_input.device
    n_selected = len(top_idx)
    
    q_activations = None
    hook_handle = None
    
    def capture_q_hook(acts, hook):
        nonlocal q_activations
        # create new leaf variable, so that requires_grad can be set
        q_activations = acts.detach().clone().requires_grad_(True)
        return q_activations  # return our leaf variable, so that it is in the computation graph
    
    try:
        # register hook to policy_head.hook_q
        hook_handle = model.policy_head.hook_q.add_hook(capture_q_hook)
        
        residual_input = residual_input.detach().clone().requires_grad_(True)

        print("residual_input requires_grad:", residual_input.requires_grad)  # True
        
        # forward propagation to capture q activations
        policy_logits = model.policy_head(residual_input)
        
        # ensure q_activations are correctly captured
        if q_activations is None:
            raise ValueError("Failed to capture q activations through hook")
        
        # calculate Jacobian matrix of selected logits with respect to q
        batch_size, seq_len, d_model = q_activations.shape
        gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        
        for i, logit_idx in enumerate(top_idx):
            if q_activations.grad is not None:
                q_activations.grad.zero_()
            
            # calculate gradient of selected policy logit
            policy_logits[0, logit_idx].backward(retain_graph=True)
            
            if q_activations.grad is not None:
                grad = q_activations.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                gradient_matrix[i, :, :] = grad
        
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    
    # demean processing
    if demean:
        mean_gradient = gradient_matrix.mean(dim=0, keepdim=True)
        result_matrix = gradient_matrix - mean_gradient
    else:
        result_matrix = gradient_matrix
    
    return top_idx, top_p, result_matrix.detach(), move_positions_tensor

def compute_logit_gradients_wrt_k(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = True,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    compute gradients of policy logits with respect to k activations
    """
    
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")
    
    lboard = LeelaBoard.from_fen(fen)
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    
    if logits.dim() > 1:
        logits = logits.flatten()
    
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} out of logits range [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        actual_max_logits = min(max_n_logits, logits.size(0))
        
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, actual_max_logits)
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]
    
    move_positions = []
    for idx in top_idx:
        try:
            uci_move = lboard.idx2uci(idx.item())
            positions = lboard.uci_to_positions(uci_move)
            move_positions.append(positions)
        except Exception as e:
            logger.warning(f"cannot get move position for index {idx.item()}: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)
    
    # prepare to calculate gradient
    device = residual_input.device
    n_selected = len(top_idx)
    
    # capture k activations through hook
    k_activations = None
    hook_handle = None
    
    def capture_k_hook(acts, hook):
        nonlocal k_activations
        # create new leaf variable, so that requires_grad can be set
        k_activations = acts.detach().clone().requires_grad_(True)
        return k_activations  # return our leaf variable, so that it is in the computation graph
    
    try:
        # Register hook to policy_head.hook_k
        hook_handle = model.policy_head.hook_k.add_hook(capture_k_hook)
        
        residual_input = residual_input.detach().clone().requires_grad_(True)

        print("residual_input requires_grad:", residual_input.requires_grad)  # True
    
        policy_logits = model.policy_head(residual_input)
    
        if k_activations is None:
            raise ValueError("Failed to capture k activations through hook")

        batch_size, seq_len, d_model = k_activations.shape
        gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        
        for i, logit_idx in enumerate(top_idx):
            if k_activations.grad is not None:
                k_activations.grad.zero_()
            
            policy_logits[0, logit_idx].backward(retain_graph=True)
            if k_activations.grad is not None:
                grad = k_activations.grad[0, :, :].clone()
                gradient_matrix[i, :, :] = grad
        
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    
    if demean:
        mean_gradient = gradient_matrix.mean(dim=0, keepdim=True)
        result_matrix = gradient_matrix - mean_gradient
    else:
        result_matrix = gradient_matrix
    
    return top_idx, top_p, result_matrix.detach(), move_positions_tensor


# @torch.no_grad()
def compute_salient_logits_for_lc0(
    fen: str,
    logits: torch.Tensor,
    model=None,
    unembed_proj: torch.Tensor = None,  # Optional
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    residual_input=None,
    demean: bool = True,
    move_idx: int = None,  # New parameter: specify the move index to process
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute salient logits in LC0 model and return corresponding move positions.
    
    Args:
        fen: FEN string representing the current board state
        logits: Policy logits
        unembed_proj: Optional unembed projection matrix
        max_n_logits: Maximum number of logits to select
        desired_logit_prob: Desired cumulative probability threshold
        model: LC0 model (for computing Jacobian)
        residual_input: Residual input (for computing Jacobian)
        demean: Whether to perform demeaning operation, default is True
        move_idx: Specify the move index to process. If provided, directly process this index and ignore other parameters
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            * top_idx - Selected logit indices, shape (k,)
            * top_p - Corresponding probability values, shape (k,)
            * demeaned_vecs - Vectors, shape (k, seq_len, d_model). If demean=True, demeaned; otherwise original values
            * move_positions - Corresponding move positions, shape (k, 2), each row contains [start_position, end_position]
    """
    
    lboard = LeelaBoard.from_fen(fen)
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")

    if unembed_proj is None and (model is None or residual_input is None):
        raise ValueError("Either unembed_proj or (model + residual_input) must be provided")

    if unembed_proj is not None and model is not None:
        logger.warning("Both unembed_proj and model provided, using model for Jacobian calculation")

    if logits.dim() > 1:
        logits = logits.flatten()
    
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} out of logits range [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        actual_max_logits = min(max_n_logits, logits.size(0))
        
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, actual_max_logits)
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    move_positions = []
    for idx in top_idx:
        try:
            uci_move = lboard.idx2uci(idx.item())
            positions = lboard.uci_to_positions(uci_move)
            move_positions.append(positions)
        except Exception as e:
            # If unable to get move, use default value
            logger.warning(f"Cannot get move position for index {idx.item()}: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)

    if model is not None and residual_input is not None and hasattr(model, 'policy_head'):
        device = residual_input.device
        d_model = residual_input.shape[-1]
        
        # Ensure residual_input requires gradients
        # if not residual_input.requires_grad:
        #     residual_input = residual_input.detach().requires_grad_(True)
        residual_input = residual_input.detach().requires_grad_(True)
        # Forward pass to get policy logits
        policy_logits = model.policy_head(residual_input)
        
        # Compute Jacobian matrix for selected logits - differentiate with respect to all positions
        batch_size, seq_len = residual_input.shape[:2]
        policy_dim = policy_logits.shape[-1]
        
        full_jacobian_matrix = torch.zeros(policy_dim, seq_len, d_model, device=device)
        
        for i in range(policy_dim):
            
            if residual_input.grad is not None:
                residual_input.grad.zero_()

            policy_logits[0, i].backward(retain_graph=True)
            if residual_input.grad is not None:
                # residual_input.grad shape: (batch_size, seq_len, d_model)
                grad = residual_input.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                full_jacobian_matrix[i, :, :] = grad
                
        mean_jacobian = full_jacobian_matrix.mean(dim=0, keepdim=True)  # (1, seq_len, d_model)
        
        selected_jacobian_matrix = full_jacobian_matrix[top_idx]  # (k, seq_len, d_model)

        unembed_proj = selected_jacobian_matrix[:, -1, :].T.detach()  # shape: (d_model, k)
    
        if demean:
            result_matrix = selected_jacobian_matrix - mean_jacobian  # (k, seq_len, d_model)
        else:
            # Do not perform demeaning, use original values directly
            result_matrix = selected_jacobian_matrix
        
        # Return selected logit indices, probabilities, Jacobian matrix, and move positions
        print(f"{top_idx = }")
        return top_idx, top_p, result_matrix.detach(), move_positions_tensor
    
    elif unembed_proj is not None:
        # Use existing unembed_proj for computation
        cols = unembed_proj[:, top_idx]
        if demean:
            result = cols - unembed_proj.mean(dim=-1, keepdim=True)
        else:
            result = cols
        return top_idx, top_p, result.T, move_positions_tensor
    else:
        raise ValueError("Neither valid model nor unembed_proj provided")

def compute_logit_gradients_wrt_group_k(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = True,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Group mode: perform gradient difference for "positive move - all other legal moves from the same start (negative samples)" for a given start position.

    Returns move_positions_tensor as LongTensor with shape **[B, 2, M]**:
      - B = 1 (single group), dimension 0 is batch
      - Dimension 1 has size 2: row 0 is Q-side start position, row 1 is all K-side end positions
      - Dimension 2 has size M: column 0 contains positive sample end position, subsequent columns contain all negative sample end positions
        * Q row: only column 0 contains start position, other columns filled with -1 (downstream can use mask to filter)
        * K row: filled with [end_pos, end_neg1, end_neg2, ...]
    Downstream usage example:
        batch_move_positions = move_positions[i:i+bs]      # [bs, 2, M]
        batch_move_positions_q = batch_move_positions[:, 0:1, :1]   # [bs,1,1] only start position
        batch_move_positions_k = batch_move_positions[:, 1:2, :]    # [bs,1,M] all K-side end positions
        valid_k_mask = (batch_move_positions_k >= 0)                 # Filter -1 padding values

    Other returns are consistent with regular compute_logit_gradients_wrt_qk:
      top_idx, top_p, q_result_matrix, k_result_matrix, move_positions_tensor, residual_result_matrix
    """
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")

    lboard = LeelaBoard.from_fen(fen)

    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    if logits.dim() > 1:
        logits = logits.flatten()

    assert move_idx is not None, "move_idx must be given to compute_logit_gradients_wrt_group in group mode"
    if move_idx < 0 or move_idx >= logits.size(0):
        raise ValueError(f"move_idx {move_idx} exceed the range of logits [0, {logits.size(0)-1}]")

    top_idx = torch.tensor([move_idx], device=logits.device)
    probs = torch.softmax(logits, dim=-1)
    top_p = probs[move_idx].unsqueeze(0)

    # --- Build positive/negative sample UCI lists ---
    chosen_uci = lboard.idx2uci(int(move_idx))
    start_sq = chosen_uci[:2]  # Same start position
    legal_uci_all: List[str] = [mv.uci() for mv in lboard.generate_legal_moves()]
    negative_move_ucis = [u for u in legal_uci_all if u.startswith(start_sq) and u != chosen_uci]

    # --- Extract Q start & K end positions ---
    def uci_to_qkpos(uci: str) -> tuple[int, int]:
        pos = lboard.uci_to_positions(uci)  # Expected to return [q_pos, k_pos] or torch.Tensor([q,k])
        if isinstance(pos, torch.Tensor):
            return int(pos[0].item()), int(pos[1].item())
        return int(pos[0]), int(pos[1])

    qpos_pos, kpos_pos = uci_to_qkpos(chosen_uci)
    kpos_negs = [uci_to_qkpos(u)[1] for u in negative_move_ucis]

    # First create 2×M: row 0 is Q start (others -1), row 1 is all K end positions
    M = 1 + len(kpos_negs)
    move_pos_2d = torch.full((2, M), -1, dtype=torch.long, device=logits.device)
    move_pos_2d[0, 0] = qpos_pos
    move_pos_2d[1, 0] = kpos_pos
    if len(kpos_negs) > 0:
        move_pos_2d[1, 1:1+len(kpos_negs)] = torch.tensor(kpos_negs, dtype=torch.long, device=logits.device)

    # Then wrap with batch dimension => [1, 2, M]
    move_positions_tensor = move_pos_2d.unsqueeze(0)

    # ====== "Positive sample − Negative sample" gradient difference ======
    device = residual_input.device
    n_selected = 1
    q_activations = None
    k_activations = None
    q_hook_handle = None
    k_hook_handle = None

    def capture_q_hook(acts, hook):
        nonlocal q_activations
        q_activations = acts
        q_activations.retain_grad()
        return q_activations

    def capture_k_hook(acts, hook):
        nonlocal k_activations
        k_activations = acts
        k_activations.retain_grad()
        return k_activations

    try:
        q_hook_handle = model.policy_head.hook_q.add_hook(capture_q_hook)
        k_hook_handle = model.policy_head.hook_k.add_hook(capture_k_hook)

        residual_input = residual_input.detach().clone().requires_grad_(True)
        policy_logits = model.policy_head(residual_input)

        if q_activations is None:
            raise ValueError("Failed to capture q activations through hook")
        if k_activations is None:
            raise ValueError("Failed to capture k activations through hook")

        _, seq_len, d_model = q_activations.shape
        q_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        k_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        residual_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)

        # ---- Positive sample ----
        pos_idx = int(top_idx[0].item())
        model.zero_grad(set_to_none=True)
        if q_activations.grad is not None: q_activations.grad.zero_()
        if k_activations.grad is not None: k_activations.grad.zero_()
        if residual_input.grad is not None: residual_input.grad.zero_()

        policy_logits[0, pos_idx].backward(retain_graph=True)
        q_accum   = q_activations.grad[0].detach().clone()
        k_accum   = k_activations.grad[0].detach().clone()
        res_accum = residual_input.grad[0].detach().clone()

        # ---- Negative samples (same start) ----
        neg_indices: List[int] = [lboard.uci2idx(u) for u in negative_move_ucis]
        n_neg = len(neg_indices)
        neg_weight = (1.0 / n_neg) if n_neg > 0 else 0.0   # Can also be changed to 1.0 to represent "simple subtraction"

        for j, neg_idx in enumerate(neg_indices):
            model.zero_grad(set_to_none=True)
            if q_activations.grad is not None: q_activations.grad.zero_()
            if k_activations.grad is not None: k_activations.grad.zero_()
            if residual_input.grad is not None: residual_input.grad.zero_()

            retain = (j < n_neg - 1)
            policy_logits[0, int(neg_idx)].backward(retain_graph=retain)
            
            # x = k_activations.grad[0]        # shape [64, 768]
            # mask = (x != 0).any(dim=1)       # [64] bool
            # row_idx = mask.nonzero(as_tuple=True)[0]   # LongTensor, nonzero row indices
            # print(row_idx.tolist())
            
            q_accum   -= neg_weight * q_activations.grad[0].detach()
            k_accum   -= neg_weight * k_activations.grad[0].detach()
            res_accum -= neg_weight * residual_input.grad[0].detach()

        q_gradient_matrix[0] = q_accum
        k_gradient_matrix[0] = k_accum
        residual_gradient_matrix[0] = res_accum

    finally:
        if q_hook_handle is not None:
            q_hook_handle.remove()
        if k_hook_handle is not None:
            k_hook_handle.remove()

    # If demeaning is needed, enable demean branch here
    q_result_matrix = q_gradient_matrix
    k_result_matrix = k_gradient_matrix
    residual_result_matrix = residual_gradient_matrix

    return (
        top_idx,
        top_p,
        q_result_matrix.detach(),
        k_result_matrix.detach(),
        move_positions_tensor,           # [1, 2, M]
        residual_result_matrix.detach(),
    )


def compute_logit_gradients_wrt_qk(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = False,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            * top_idx - Selected logit indices, shape (k,)
            * top_p - Corresponding probability values, shape (k,)
            * q_gradient_matrix - Gradient matrix for q, shape (k, seq_len, d_model)
            * k_gradient_matrix - Gradient matrix for k, shape (k, seq_len, d_model)
            * move_positions - Corresponding move positions, shape (k, 2)
            * residual_gradient_matrix - Gradient matrix for residual_input, shape (k, seq_len, d_model)
    """
    
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")
    
    lboard = LeelaBoard.from_fen(fen)
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    
    # Ensure logits is a 1D tensor
    if logits.dim() > 1:
        logits = logits.flatten()
    
    # Select logit indices to process
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} out of logits range [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        # Original top logits selection logic
        actual_max_logits = min(max_n_logits, logits.size(0))
        
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, actual_max_logits)
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]
    
    # Compute move positions corresponding to selected logits
    move_positions = []
    for idx in top_idx:
        try:
            uci_move = lboard.idx2uci(idx.item())
            positions = lboard.uci_to_positions(uci_move)
            move_positions.append(positions)
        except Exception as e:
            logger.warning(f"Cannot get move position for index {idx.item()}: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)
    
    # Prepare to compute gradients
    device = residual_input.device
    n_selected = len(top_idx)
    
    # Unified management of q and k activations, hooks, and gradient matrices
    activations_dict = {'q': None, 'k': None}
    hook_handles = {'q': None, 'k': None}
    hook_points = {
        'q': model.policy_head.hook_q,
        'k': model.policy_head.hook_k
    }
    
    # Generic hook capture function
    def create_capture_hook(key):
        def capture_hook(acts, hook):
            activations_dict[key] = acts
            activations_dict[key].retain_grad()
            return activations_dict[key]
        return capture_hook
    
    try:
        # Register hooks
        for key in ['q', 'k']:
            hook_handles[key] = hook_points[key].add_hook(create_capture_hook(key))
        
        # Set residual_input as leaf node
        residual_input = residual_input.detach().clone().requires_grad_(True)
        
        # Forward pass to capture q and k activations
        policy_logits = model.policy_head(residual_input)
        
        # Ensure q and k activations are correctly captured
        for key in ['q', 'k']:
            if activations_dict[key] is None:
                raise ValueError(f"Failed to capture {key} activations through hook")
        
        # Get sequence length and model dimension
        batch_size, seq_len, d_model = activations_dict['q'].shape
        
        # Initialize gradient matrices
        gradient_matrices = {
            'q': torch.zeros(n_selected, seq_len, d_model, device=device),
            'k': torch.zeros(n_selected, seq_len, d_model, device=device)
        }
        residual_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        
        # Compute gradient for each selected logit
        for i, logit_idx in enumerate(top_idx):
            # Zero all gradients
            for key in ['q', 'k']:
                if activations_dict[key].grad is not None:
                    activations_dict[key].grad.zero_()
            if residual_input.grad is not None:
                residual_input.grad.zero_()
            
            # Compute gradient of selected policy logit
            policy_logits[0, logit_idx].backward(retain_graph=True)
            
            # Collect gradients for all activations
            for key in ['q', 'k']:
                if activations_dict[key].grad is not None:
                    grad = activations_dict[key].grad[0, :, :].clone()  # shape: (seq_len, d_model)
                    gradient_matrices[key][i, :, :] = grad
            
            # Collect gradient for residual_input
            if residual_input.grad is not None:
                grad = residual_input.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                residual_gradient_matrix[i, :, :] = grad
        
    finally:
        # Remove all hooks
        for key in ['q', 'k']:
            if hook_handles[key] is not None:
                hook_handles[key].remove()
    
    # Demean processing
    if demean:
        result_matrices = {}
        for key in ['q', 'k']:
            mean_gradient = gradient_matrices[key].mean(dim=0, keepdim=True)
            result_matrices[key] = gradient_matrices[key] - mean_gradient
        residual_mean_gradient = residual_gradient_matrix.mean(dim=0, keepdim=True)
        residual_result_matrix = residual_gradient_matrix - residual_mean_gradient
    else:
        result_matrices = gradient_matrices
        residual_result_matrix = residual_gradient_matrix
    
    return top_idx, top_p, result_matrices['q'].detach(), result_matrices['k'].detach(), move_positions_tensor, residual_result_matrix.detach()


def compute_logit_gradients_wrt_qk_legacy(
    fen: str,
    logits: torch.Tensor,
    model=None,
    residual_input=None,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demean: bool = True,
    move_idx: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute gradients of policy logits with respect to q and k activations in LC0 model.
    
    Args:
        fen: FEN string representing the current board state
        logits: Policy logits
        model: LC0 model (must be provided)
        residual_input: Residual input (must be provided)
        max_n_logits: Maximum number of logits to select
        desired_logit_prob: Desired cumulative probability threshold
        demean: Whether to perform demeaning operation, default is True
        move_idx: Specify the move index to process. If provided, directly process this index
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            * top_idx - Selected logit indices, shape (k,)
            * top_p - Corresponding probability values, shape (k,)
            * q_gradient_matrix - Gradient matrix for q, shape (k, seq_len, d_model)
            * k_gradient_matrix - Gradient matrix for k, shape (k, seq_len, d_model)
            * move_positions - Corresponding move positions, shape (k, 2)
            * residual_gradient_matrix - Gradient matrix for residual_input, shape (k, seq_len, d_model)
    """
    
    if model is None or residual_input is None:
        raise ValueError("Both model and residual_input must be provided")
    
    if not hasattr(model, 'policy_head'):
        raise ValueError("Model must have policy_head attribute")
    
    lboard = LeelaBoard.from_fen(fen)
    
    if logits.numel() == 0:
        raise ValueError("Input logits tensor is empty")
    
    # Ensure logits is a 1D tensor
    if logits.dim() > 1:
        logits = logits.flatten()
    
    # Select logit indices to process
    if move_idx is not None:
        if move_idx < 0 or move_idx >= logits.size(0):
            raise ValueError(f"move_idx {move_idx} out of logits range [0, {logits.size(0)-1}]")
        
        top_idx = torch.tensor([move_idx], device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        top_p = probs[move_idx].unsqueeze(0)
    else:
        # Original top logits selection logic
        actual_max_logits = min(max_n_logits, logits.size(0))
        
        probs = torch.softmax(logits, dim=-1)
        top_p, top_idx = torch.topk(probs, actual_max_logits)
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]
    
    # Compute move positions corresponding to selected logits
    move_positions = []
    for idx in top_idx:
        try:
            uci_move = lboard.idx2uci(idx.item())
            positions = lboard.uci_to_positions(uci_move)
            move_positions.append(positions)
        except Exception as e:
            logger.warning(f"Cannot get move position for index {idx.item()}: {e}")
            move_positions.append(torch.tensor([0, 0]))
    
    move_positions_tensor = torch.stack(move_positions)
    
    # Prepare to compute gradients
    device = residual_input.device
    n_selected = len(top_idx)
    
    # Capture q and k activations through hooks
    q_activations = None
    k_activations = None
    q_hook_handle = None
    k_hook_handle = None
    
    def capture_q_hook(acts, hook):
        nonlocal q_activations
        # Use retain_grad() to retain gradients instead of creating leaf node
        q_activations = acts
        q_activations.retain_grad()
        return q_activations
    
    def capture_k_hook(acts, hook):
        nonlocal k_activations
        # Use retain_grad() to retain gradients instead of creating leaf node
        k_activations = acts
        k_activations.retain_grad()
        return k_activations
    
    try:
        # register hook to policy_head.hook_q and hook_k
        q_hook_handle = model.policy_head.hook_q.add_hook(capture_q_hook)
        k_hook_handle = model.policy_head.hook_k.add_hook(capture_k_hook)
        
        # Set residual_input as leaf node
        residual_input = residual_input.detach().clone().requires_grad_(True)

        print("residual_input requires_grad:", residual_input.requires_grad)  # True
        
        # Forward pass to capture q and k activations
        policy_logits = model.policy_head(residual_input)
        
        # Ensure q and k activations are correctly captured
        if q_activations is None:
            raise ValueError("Failed to capture q activations through hook")
        if k_activations is None:
            raise ValueError("Failed to capture k activations through hook")
        
        # Compute Jacobian matrix of selected logits with respect to q, k, and residual_input
        batch_size, seq_len, d_model = q_activations.shape
        q_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        k_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        residual_gradient_matrix = torch.zeros(n_selected, seq_len, d_model, device=device)
        
        for i, logit_idx in enumerate(top_idx):
            # Zero all gradients
            if q_activations.grad is not None:
                q_activations.grad.zero_()
            if k_activations.grad is not None:
                k_activations.grad.zero_()
            if residual_input.grad is not None:
                residual_input.grad.zero_()
            
            # Compute gradient of selected policy logit
            policy_logits[0, logit_idx].backward(retain_graph=True)
            
            # Collect gradient for q
            if q_activations.grad is not None:
                grad = q_activations.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                q_gradient_matrix[i, :, :] = grad

            if k_activations.grad is not None:
                grad = k_activations.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                k_gradient_matrix[i, :, :] = grad
            
            # Collect gradient for residual_input
            if residual_input.grad is not None:
                grad = residual_input.grad[0, :, :].clone()  # shape: (seq_len, d_model)
                residual_gradient_matrix[i, :, :] = grad
        
    finally:
        # Remove hooks
        if q_hook_handle is not None:
            q_hook_handle.remove()
        if k_hook_handle is not None:
            k_hook_handle.remove()
    
    # Demean processing
    if demean:
        q_mean_gradient = q_gradient_matrix.mean(dim=0, keepdim=True)
        k_mean_gradient = k_gradient_matrix.mean(dim=0, keepdim=True)
        residual_mean_gradient = residual_gradient_matrix.mean(dim=0, keepdim=True)
        q_result_matrix = q_gradient_matrix - q_mean_gradient
        k_result_matrix = k_gradient_matrix - k_mean_gradient
        residual_result_matrix = residual_gradient_matrix - residual_mean_gradient
    else:
        q_result_matrix = q_gradient_matrix
        k_result_matrix = k_gradient_matrix
        residual_result_matrix = residual_gradient_matrix
    
    return top_idx, top_p, q_result_matrix.detach(), k_result_matrix.detach(), move_positions_tensor, residual_result_matrix.detach()



@torch.no_grad()  # modified
def select_scaled_decoder_vecs_tc(
    activations: torch.sparse.Tensor,
    transcoders: Dict[int, SparseAutoEncoder]
) -> torch.Tensor:
    """Return decoder rows for **active** features only.

    The return value is already scaled by the feature activation, making it
    suitable as ``inject_values`` during gradient overrides.
    
    For transcoders, each layer has its own independent encoder/decoder,
    unlike TC where features can span multiple layers.
    """
    # Assert that the values in transcoders are of type SparseAutoEncoder
    assert all(isinstance(t, SparseAutoEncoder) for t in transcoders.values())

    rows: List[torch.Tensor] = []
    
    # Convert activations to coalesced sparse tensors for each layer
    feature_act_rows = [activations[layer].coalesce() for layer in range(len(transcoders))]
    
    for layer in range(len(transcoders)):
        _, feat_idx = feature_act_rows[layer].indices()
        
        # Retrieve the decoder weights from the current layer's transcoder
        W_D = transcoders[str(layer)].W_D  # Shape: [d_sae, d_model]
        # Scale the decoder row by the feature activations
        # W_D[feat_idx]: [n_active_features, d_model]
        # feature_act_rows[layer].values(): [n_active_features]
        scaled_row = W_D[feat_idx] * feature_act_rows[layer].values()[:, None]
        
        rows.append(scaled_row)
    
    # Concatenate all the scaled rows
    return torch.cat(rows)

@torch.no_grad()
def select_scaled_decoder_vecs_lorsa(
    activation_matrix: torch.Tensor,
    lorsas: LowRankSparseAttention
) -> torch.Tensor:
    """Return decoder rows for active Lorsa heads, scaled by activations."""
    decoder_rows: List[torch.Tensor] = []
    sparse_rows: List[torch.sparse.Tensor] = []
    for layer, row in enumerate(activation_matrix):
        if row.layout != torch.sparse_coo:
            row = row.to_sparse()
        row = row.coalesce()
        sparse_rows.append(row)
        _, head_idx = row.indices()
        decoder_rows.append(lorsas[layer].W_O[head_idx])

    if not decoder_rows:
        return torch.empty(0, lorsas[0].cfg.d_model, device=activation_matrix.device)

    stacked_decoders = torch.cat(decoder_rows)
    stacked_values = torch.cat([row.values() for row in sparse_rows])[:, None]
    return stacked_decoders * stacked_values


@torch.no_grad()
def select_encoder_rows_tc(
    activation_matrix: torch.sparse.Tensor, 
    transcoders: Dict[int, SparseAutoEncoder]
) -> torch.Tensor:
    """Return encoder rows for **active** features only.
    
    For transcoders, each layer has its own independent encoder/decoder,
    unlike TC where features can span multiple layers.
    """
    rows: List[torch.Tensor] = []
    
    # Iterate through activation matrix for each layer
    for layer, row in enumerate(activation_matrix):
        _, feat_idx = row.coalesce().indices()
        
        # Use string key to access transcoder for this layer
        # W_E.T[feat_idx]: [n_active_features, d_model] 
        rows.append(transcoders[str(layer)].W_E.T[feat_idx]) 
        
    return torch.cat(rows)

@torch.no_grad()
def select_encoder_rows_lorsa(
    activation_matrix: torch.sparse.Tensor,
    attention_pattern: torch.Tensor,
    lorsas: LowRankSparseAttention
) -> torch.Tensor:
    """Return encoder rows for **active** features only."""
    rows: List[torch.Tensor] = []
    patterns: List[torch.Tensor] = []
    for layer, row in enumerate(activation_matrix):
        qpos, head_idx = row.coalesce().indices()
        # qk_idx = head_idx // lorsas[layer].cfg.d_qk_head
        qk_idx: Tensor = head_idx // (lorsas[layer].cfg.n_ov_heads // lorsas[layer].cfg.n_qk_heads)
        # torch.cuda.synchronize()
        # print(f'{attention_pattern.shape = }, {layer = }, {qk_idx = }, {qpos = }')
        pattern = attention_pattern[layer, qk_idx, qpos]
        patterns.append(pattern)
        rows.append(lorsas[layer].W_V[head_idx])
    return torch.cat(rows), torch.cat(patterns)

@torch.no_grad()
def select_encoder_bias_tc(
    activation_matrix: torch.sparse.Tensor,
    transcoders: Dict[str, "SparseAutoEncoder"],  # Consistent with rows version, use str(layer) as key
) -> torch.Tensor:
    rows: List[torch.Tensor] = []

    for layer, row in enumerate(activation_matrix):
        idx2d = row.coalesce().indices()
        if idx2d.numel() == 0:
            continue 

        _, feat_idx = idx2d  
        tc = transcoders[str(layer)]

        if getattr(tc, "b_E", None) is None:
            dev, dt = tc.W_E.device, tc.W_E.dtype
            layer_bias = torch.zeros(feat_idx.numel(), device=dev, dtype=dt)
        else:
            layer_bias = tc.b_E.index_select(0, feat_idx.to(device=tc.b_E.device))

        rows.append(layer_bias)

    if not rows:
        if len(transcoders) > 0:
            any_tc = next(iter(transcoders.values()))
            return torch.empty(0, device=any_tc.W_E.device, dtype=any_tc.W_E.dtype)
        return torch.empty(0, device=activation_matrix.device, dtype=activation_matrix.dtype)

    return torch.cat(rows, dim=0)

@torch.no_grad()
def select_encoder_bias_lorsa(
    activation_matrix: torch.sparse.Tensor,
    lorsas: LowRankSparseAttention,
) -> torch.Tensor:
    """
    Return encoder bias terms for Lorsa active features only.

    For each layer, gather the bias vector entries corresponding to the
    active heads (same indexing as select_encoder_rows_lorsa). If the
    Lorsa layer has no encoder bias attribute (e.g., b_V), return zeros
    for that layer's active heads to keep alignment with rows.
    """
    rows: List[torch.Tensor] = []

    for layer, row in enumerate(activation_matrix):
        idx2d = row.coalesce().indices()
        if idx2d.numel() == 0:
            continue

        _, head_idx = idx2d  # active head indices for this layer
        lrs = lorsas[layer]

        # Prefer b_V if present; otherwise produce zeros matching dtype/device
        bias_tensor = getattr(lrs, "b_V", None)
        if bias_tensor is None:
            dev, dt = lrs.W_V.device, lrs.W_V.dtype
            layer_bias = torch.zeros(head_idx.numel(), device=dev, dtype=dt)
        else:
            layer_bias = bias_tensor.index_select(0, head_idx.to(device=bias_tensor.device))

        rows.append(layer_bias)

    if not rows:
        # Fall back to an empty tensor on a reasonable device/dtype
        if len(lorsas) > 0:
            any_layer = next(iter(range(len(lorsas))))
            dev, dt = lorsas[any_layer].W_V.device, lorsas[any_layer].W_V.dtype
            return torch.empty(0, device=dev, dtype=dt)
        return torch.empty(0, device=activation_matrix.device, dtype=activation_matrix.dtype)

    return torch.cat(rows, dim=0)

# def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
#     normalized_matrix = normalized_matrix.abs_()
#     normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

#     influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
#     prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
#     prod[-len(logit_p) :] = logit_p

#     for _ in range(max_iter):
#         prod = prod[row_to_node_index] @ normalized_matrix
#         if not prod.any():
#             break
#         influences += prod
#     else:
#         raise RuntimeError("Failed to converge")

#     return influences

def _normalize_rows(rows: torch.Tensor, sign_mode: str = "abs") -> torch.Tensor:
    denominator = rows.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    if sign_mode == "abs":
        return rows.abs() / denominator
    if sign_mode == "signed":
        return rows / denominator
    raise ValueError("sign_mode must be 'abs' or 'signed'")


def compute_partial_influences(
    edge_matrix,
    logit_p,
    row_to_node_index,
    max_iter=128,
    device=None,
    sign_mode="abs",
    pre_normalized: bool = False,
):  # 'abs' | 'signed'
    device = device or edge_matrix.device
    W = edge_matrix.to(device)

    if not pre_normalized:
        W = _normalize_rows(W, sign_mode)

    influences = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)
    prod = torch.zeros(W.shape[1], device=W.device, dtype=W.dtype)
    if len(logit_p) > 0:
        prod[-len(logit_p):] = logit_p.to(device=W.device, dtype=W.dtype)
    row_to_node_index = row_to_node_index.to(W.device)

    for _ in range(max_iter):
        prod = prod.index_select(0, row_to_node_index) @ W
        influences += prod

    return influences


def partial_influence_queue_config(order_mode: str) -> tuple[str, bool]:
    """Map ``order_mode`` to ``(sign_mode, descending)`` for feature-queue ranking.

    ``sign_mode`` is passed to :func:`compute_partial_influences` (``\"abs\"`` or
    ``\"signed\"``). ``descending`` is the sort order of accumulated partial
    influences on feature nodes.

    Modes:
        - ``abs`` (default): use :math:`|W|` in the random walk, sort descending.
          Matches the historical default when ``order_mode`` was ``\"positive\"``.
        - ``positive``: signed :math:`W`, sort descending (largest net scores first;
          tends to prioritize promoters over suppressors).
        - ``negative``: signed :math:`W`, sort ascending (smallest / most negative first).
        - ``move_pair`` / ``group``: same queue rule as ``abs`` (descending, abs edges).

    Args:
        order_mode: One of ``abs``, ``positive``, ``negative``, ``move_pair``, ``group``.

    Returns:
        Tuple ``(sign_mode, descending)`` for :func:`compute_partial_influences` and
        ``torch.argsort(..., descending=...)``.
    """
    if order_mode == "negative":
        return "signed", False
    if order_mode == "positive":
        return "signed", True
    if order_mode in ("abs", "move_pair", "group"):
        return "abs", True
    logger.warning("Unknown order_mode %r; using abs weights + descending sort.", order_mode)
    return "abs", True


def _batched_index_queue(indices: torch.Tensor, batch_size: int) -> deque[torch.Tensor]:
    """Split a 1-D index tensor into FIFO batches."""
    if indices.numel() == 0:
        return deque()
    return deque(indices.split(batch_size))


def run_joint_feature_attribution(
    *,
    ctx: AttributionContext,
    requested_sides: Sequence[str],
    edge_matrices: Dict[str, torch.Tensor],
    normalized_matrices: Dict[str, torch.Tensor],
    row_to_node_indices: Dict[str, torch.Tensor],
    total_active_feats: int,
    max_feature_nodes: int,
    update_interval: int,
    selection_batch_size: int,
    vjp_batch_size: int,
    n_logits: int,
    logit_p: torch.Tensor,
    logit_offset: int,
    idx_to_layer: Callable[[torch.Tensor], torch.Tensor],
    idx_to_pos: Callable[[torch.Tensor], torch.Tensor],
    idx_to_encoder_rows: Callable[[torch.Tensor], torch.Tensor],
    idx_to_pattern: Callable[[torch.Tensor], torch.Tensor],
    order_mode: str,
    initial_queue: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Advance independent Q/K greedy queues while sharing feature-row VJPs."""

    sign_mode, descending = partial_influence_queue_config(order_mode)
    device = next(iter(edge_matrices.values())).device
    selection_batch_size = max(1, selection_batch_size)
    states: Dict[str, Dict[str, Any]] = {}
    for side in requested_sides:
        manual = deque()
        if initial_queue is not None and initial_queue.numel() > 0:
            manual = _batched_index_queue(torch.unique(initial_queue.to(device)), selection_batch_size)
        states[side] = {
            "visited": torch.zeros(total_active_feats, dtype=torch.bool, device=device),
            "n_visited": 0,
            "st": n_logits,
            "manual": manual,
            "auto": deque(),
        }

    cache_side = torch.full((total_active_feats,), -1, dtype=torch.int8, device=device)
    cache_row = torch.full((total_active_feats,), -1, dtype=torch.long, device=device)
    side_number = {side: i for i, side in enumerate(requested_sides)}
    number_side = {i: side for side, i in side_number.items()}

    def next_batch(side: str) -> torch.Tensor:
        state = states[side]
        if state["n_visited"] >= max_feature_nodes:
            return torch.empty(0, dtype=torch.long, device=device)
        queue = state["manual"]
        if not queue:
            queue = state["auto"]
        if not queue:
            if n_logits == 0:
                return torch.empty(0, dtype=torch.long, device=device)
            visited = state["visited"]
            remaining = max_feature_nodes - state["n_visited"]
            if max_feature_nodes == total_active_feats:
                pending = torch.nonzero(~visited, as_tuple=True)[0][:remaining]
            else:
                st = state["st"]
                influences = compute_partial_influences(
                    normalized_matrices[side][:st],
                    logit_p,
                    row_to_node_indices[side][:st],
                    max_iter=2 * ctx.n_layers + 2,
                    sign_mode=sign_mode,
                    pre_normalized=True,
                )
                available = torch.nonzero(~visited, as_tuple=True)[0]
                queue_size = min(update_interval * selection_batch_size, remaining, available.numel())
                if queue_size == 0:
                    return torch.empty(0, dtype=torch.long, device=device)
                scores = influences[:total_active_feats].index_select(0, available)
                top = torch.topk(scores, k=queue_size, largest=descending, sorted=True).indices
                pending = available.index_select(0, top)
            state["auto"] = _batched_index_queue(pending, selection_batch_size)
            queue = state["auto"]
        batch = queue.popleft()
        remaining = max_feature_nodes - state["n_visited"]
        return batch[:remaining]

    progress = {side: tqdm(total=max_feature_nodes, desc=f"{side.upper()} feature influence") for side in requested_sides}
    try:
        while any(states[side]["n_visited"] < max_feature_nodes for side in requested_sides):
            batches = {side: next_batch(side) for side in requested_sides}
            active = {side: gids for side, gids in batches.items() if gids.numel() > 0}
            if not active:
                break
            union = torch.unique(torch.cat(tuple(active.values())))
            union_rows = edge_matrices[requested_sides[0]].new_zeros((union.numel(), logit_offset))
            cached = cache_side.index_select(0, union) >= 0
            for number, cached_side_name in number_side.items():
                mask = cached & (cache_side.index_select(0, union) == number)
                if mask.any():
                    rows = cache_row.index_select(0, union[mask])
                    union_rows[mask] = edge_matrices[cached_side_name].index_select(0, rows)[:, :logit_offset]

            missing_gids = union[~cached]
            if missing_gids.numel() > 0:
                chunks = []
                for gid_chunk in missing_gids.split(vjp_batch_size):
                    patterns = idx_to_pattern(gid_chunk).detach()
                    rows = ctx.compute_vjp_batch(
                        layers=idx_to_layer(gid_chunk),
                        positions=idx_to_pos(gid_chunk),
                        inject_values=idx_to_encoder_rows(gid_chunk).detach(),
                        attention_patterns=patterns,
                        retain_graph=True,
                    )
                    # Attribution rows are final numeric results. Keeping their
                    # parameter-side grad_fn here makes every slice assignment
                    # extend the edge matrix's autograd history across all
                    # feature batches, causing GPU memory to grow linearly.
                    chunks.append(rows.detach().to(union_rows.dtype))
                union_rows[~cached] = torch.cat(chunks, dim=0)

            newly_cached: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
            for side, gids in active.items():
                state = states[side]
                offsets = torch.searchsorted(union, gids)
                rows = union_rows.index_select(0, offsets)
                st = state["st"]
                end = st + gids.numel()
                edge_matrices[side][st:end, :logit_offset] = rows
                normalized_matrices[side][st:end, :logit_offset] = _normalize_rows(
                    rows.float(), sign_mode
                ).to(normalized_matrices[side].dtype)
                row_to_node_indices[side][st:end] = gids
                state["visited"][gids] = True
                state["n_visited"] += gids.numel()
                state["st"] = end
                progress[side].update(gids.numel())
                newly_cached.append((side, gids, torch.arange(st, end, device=device)))

            missing_set = ~cached
            if missing_set.any():
                missing_union = union[missing_set]
                for side, gids, rows in newly_cached:
                    is_missing = torch.isin(gids, missing_union) & (cache_side.index_select(0, gids) < 0)
                    if is_missing.any():
                        selected = gids[is_missing]
                        cache_side[selected] = side_number[side]
                        cache_row[selected] = rows[is_missing]
    finally:
        for bar in progress.values():
            bar.close()

    return {
        side: {
            "visited": states[side]["visited"],
            "edge_matrix": edge_matrices[side],
            "normalized_matrix": normalized_matrices[side],
            "row_to_node_index": row_to_node_indices[side],
        }
        for side in requested_sides
    }


def attribute(
    prompt: Union[str, torch.Tensor, List[int]],
    model: ReplacementModel,
    is_castle: bool = False,
    *,
    max_n_logits: int = 10,
    side: str = 'both',                  # 'q' | 'k' | 'both'
    desired_logit_prob: float = 0.95,
    batch_size: int = 64,
    max_feature_nodes: Optional[int] = 4096,
    vjp_batch_size: Optional[int] = None,
    mixed_precision_edges: bool = True,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
    use_legal_moves_only: bool = False,
    fen: Optional[str] = None,
    lboard: Optional[Any] = None,
    move_idx: int | tuple[int, int] | None = None, 
    encoder_demean: bool = False,
    act_times_max: Optional[int] = None,
    mongo_client = None,
    sae_series: str = 'BT4-exp128',
    analysis_name: str = 'default',
    order_mode: str = 'abs',
    save_activation_info: bool = True,
    feature_trace_specs: Optional[Sequence[int | FeatureTraceSpec]] = None,
) -> Dict[str, Any]:
    """Compute an attribution graph for *prompt* and return a structured bundle.

    ``order_mode``: ``abs`` (default) — partial-influence queue uses :math:`|W|`,
    descending. ``positive`` — signed :math:`W`, descending (promoters tend first).
    ``negative`` — signed :math:`W`, ascending.
    """
    offload_handles = []

    input_ids = prompt
    
    try:
        return _run_attribution(
            model=model,
            prompt=input_ids,
            max_n_logits=max_n_logits,
            side=side,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            vjp_batch_size=vjp_batch_size,
            mixed_precision_edges=mixed_precision_edges,
            offload=offload,
            offload_handles=offload_handles,
            update_interval=update_interval,
            use_legal_moves_only=use_legal_moves_only,
            fen=fen,
            lboard=lboard,
            is_castle=is_castle,
            move_idx=move_idx,
            verbose=verbose,
            encoder_demean = encoder_demean,
            act_times_max = act_times_max,
            mongo_client = mongo_client,
            sae_series = sae_series,
            analysis_name = analysis_name,
            order_mode = order_mode,
            save_activation_info = save_activation_info,
            feature_trace_specs = feature_trace_specs,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

def _run_attribution(
    model,
    prompt: Union[str, torch.Tensor, List[int]],
    max_n_logits: int,
    side: str,                           # 'q' | 'k' | 'both'
    desired_logit_prob: float,
    batch_size: int,
    max_feature_nodes: Optional[int],
    vjp_batch_size: Optional[int],
    mixed_precision_edges: bool,
    offload: Literal["cpu", "disk", None],
    offload_handles: list,
    update_interval: int = 4,
    use_legal_moves_only: bool = True,
    fen: Optional[str] = None,
    lboard: Optional[Any] = None,
    is_castle: bool = False,
    move_idx: int | tuple[int, int] | None = None,
    verbose: bool = False,
    encoder_demean: bool = False,
    # for filtering
    act_times_max: Optional[int] = None,
    mongo_client = None,
    sae_series: str = 'BT4-exp128',
    analysis_name: str = 'default',
    order_mode: str = 'abs',  # abs | positive | negative | move_pair | group
    save_activation_info: bool = True,
    feature_trace_specs: Optional[Sequence[int | FeatureTraceSpec]] = None,
) -> Dict[str, Any]:
    start_time = time.time()

    # ========== type checking and move index processing ============
    positive_move_idx = None
    negative_move_idx = None
    feature_specs_requested = bool(feature_trace_specs)
    
    if order_mode in ('positive', 'abs'):
        positive_move_idx = move_idx
        print(f'{positive_move_idx = }')
    elif order_mode == 'negative':
        negative_move_idx = move_idx
        print(f'{negative_move_idx = }')
    elif order_mode == 'move_pair':
        assert isinstance(move_idx, tuple), f"move_idx must be a tuple in move_pair mode, now it is {type(move_idx)}"
        positive_move_idx, negative_move_idx = move_idx[0], move_idx[1]
        print(f'{positive_move_idx = }, {negative_move_idx = }')
    elif order_mode == 'group':
        assert side == 'k', f"side must be k during attributing in the group mode"
        positive_move_idx = move_idx
        print(f'{positive_move_idx = }')
        
    # ========== Phase 0: Precomputation ==========
    print("Phase 0: Precomputing activations and vectors")
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()

    input_ids = prompt
    if vjp_batch_size is None:
        vjp_batch_size = batch_size
    if vjp_batch_size < 1:
        raise ValueError("vjp_batch_size must be positive")
    policy_lane_count = 2 * max_n_logits
    if order_mode == "group":
        if fen is None or positive_move_idx is None:
            raise ValueError("group attribution requires fen and move_idx")
        group_board = LeelaBoard.from_fen(fen)
        group_uci = group_board.idx2uci(int(positive_move_idx))
        policy_lane_count = 1 + sum(
            candidate.uci().startswith(group_uci[:2])
            and candidate.uci() != group_uci
            for candidate in group_board.generate_legal_moves()
        )
    live_batch_size = max(vjp_batch_size, policy_lane_count, 2)
    if isinstance(input_ids, str):
        replicated_inputs: Union[List[str], torch.Tensor] = [input_ids] * live_batch_size
    elif isinstance(input_ids, torch.Tensor):
        base_inputs = input_ids.unsqueeze(0) if input_ids.ndim == 1 else input_ids
        if base_inputs.shape[0] != 1:
            raise ValueError("Attribution expects one logical prompt before VJP replication")
        replicated_inputs = base_inputs.expand(live_batch_size, *base_inputs.shape[1:])
    else:
        replicated_inputs = torch.as_tensor(input_ids).unsqueeze(0).expand(live_batch_size, -1)

    live_resid: List[Optional[torch.Tensor]] = [None] * (2 * model.cfg.n_layers + 1)
    live_attn_outputs: List[Optional[torch.Tensor]] = [None] * model.cfg.n_layers
    live_mlp_outputs: List[Optional[torch.Tensor]] = [None] * model.cfg.n_layers
    live_refs: Dict[str, torch.Tensor] = {}

    def _cache_live_ref(acts, hook, *, key):
        if not acts.requires_grad:
            acts = acts.detach().requires_grad_()
        live_refs[key] = acts
        return acts

    def _cache_live_slot(acts, hook, *, slots, index, make_leaf=False):
        if make_leaf and not acts.requires_grad:
            acts = acts.detach().requires_grad_()
        slots[index] = acts
        return acts

    live_hooks = [("hook_embed", partial(_cache_live_ref, key="embed"))]
    for layer in range(model.cfg.n_layers):
        live_hooks.extend(
            [
                (
                    f"blocks.{layer}.{model.attn_input_hook}",
                    partial(_cache_live_slot, slots=live_resid, index=2 * layer),
                ),
                (
                    f"blocks.{layer}.{model.mlp_input_hook}",
                    partial(_cache_live_slot, slots=live_resid, index=2 * layer + 1),
                ),
                (
                    f"blocks.{layer}.{model.attn_output_hook}",
                    partial(
                        _cache_live_slot,
                        slots=live_attn_outputs,
                        index=layer,
                        make_leaf=True,
                    ),
                ),
                (
                    f"blocks.{layer}.{model.mlp_output_hook}",
                    partial(
                        _cache_live_slot,
                        slots=live_mlp_outputs,
                        index=layer,
                        make_leaf=True,
                    ),
                ),
            ]
        )
    live_hooks.extend(
        [
            (
                "policy_head.hook_pre",
                partial(_cache_live_slot, slots=live_resid, index=2 * model.cfg.n_layers),
            ),
            ("policy_head.hook_q", partial(_cache_live_ref, key="policy_q")),
            ("policy_head.hook_k", partial(_cache_live_ref, key="policy_k")),
        ]
    )

    model_out, lorsa_activation_matrix, lorsa_attention_pattern, tc_activation_matrix, error_vecs, token_vecs = model.setup_attribution(
        replicated_inputs,
        sparse=True,
        extra_fwd_hooks=live_hooks,
        enable_grad=True,
        first_batch_only=True,
    )
    print("set up attribution! ")
    
    lorsa_decoder_vecs = select_scaled_decoder_vecs_lorsa(lorsa_activation_matrix, model.lorsas)
    lorsa_encoder_rows, lorsa_attention_patterns = select_encoder_rows_lorsa(lorsa_activation_matrix, lorsa_attention_pattern, model.lorsas)
    lorsa_encoder_bias = select_encoder_bias_lorsa(lorsa_activation_matrix, model.lorsas)
    
    tc_decoder_vecs = select_scaled_decoder_vecs_tc(tc_activation_matrix, model.transcoders)
    tc_encoder_rows = select_encoder_rows_tc(tc_activation_matrix, model.transcoders)
    tc_encoder_bias = select_encoder_bias_tc(tc_activation_matrix, model.transcoders)

    ctx = AttributionContext(
        lorsa_activation_matrix,
        tc_activation_matrix,
        error_vecs,
        token_vecs,
        lorsa_decoder_vecs,
        tc_decoder_vecs,
        model.attn_output_hook,
        model.mlp_output_hook
    )
    ctx._resid_activations = live_resid
    ctx._embed_activation = live_refs["embed"]
    ctx._attn_output_activations = live_attn_outputs
    ctx._mlp_output_activations = live_mlp_outputs
    ctx._policy_q_activations = live_refs["policy_q"]
    ctx._policy_k_activations = live_refs["policy_k"]
    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {tc_activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # BT4 returns a list of head outputs; policy logits are the first head.
    # The setup caches and live VJP graph are collected by the same replicated forward.
    live_policy_logits = model_out[0]
    activation_info = None

    if offload:
        offload_handles += offload_modules(
            [block.mlp for block in model.blocks] + [block.attn for block in model.blocks],
            offload,
        )

    # ========== Phase 2: Prepare logit related ==========
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()

    policy_out = model_out[0]
    n_layers, n_pos, _ = tc_activation_matrix.shape
    total_active_feats = lorsa_activation_matrix._nnz() + tc_activation_matrix._nnz()
    phase2_time = time.time() - phase_start
    print(f"Phase 2: Building input vectors completed in {phase2_time:.2f}s")
    logger.info(f"Phase 2: Building input vectors completed in {phase2_time:.2f}s")

    # Initialize variables
    logit_idx_positive = None
    logit_p_positive = None
    logit_vecs_q_positive = None
    logit_vecs_k_positive = None
    move_positions_positive = None
    logit_vecs_positive = None
    
    logit_idx_negative = None
    logit_p_negative = None
    logit_vecs_q_negative = None
    logit_vecs_k_negative = None
    move_positions_negative = None
    logit_vecs_negative = None
    
    group_negative_indices = None

    def _selected_move(move: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fen is None:
            raise ValueError("fen is required for policy Q/K attribution")
        idx = torch.tensor([move], device=policy_out.device, dtype=torch.long)
        probability = torch.softmax(policy_out[0], dim=-1).index_select(0, idx)
        board = LeelaBoard.from_fen(fen)
        uci = board.idx2uci(int(move))
        positions = torch.as_tensor(
            board.uci_to_positions(uci),
            device=policy_out.device,
            dtype=torch.long,
        ).reshape(1, 2)
        return idx, probability, positions

    def _selected_group(
        move: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if fen is None:
            raise ValueError("fen is required for group policy attribution")
        board = LeelaBoard.from_fen(fen)
        idx, probability, positions = _selected_move(move)
        chosen_uci = board.idx2uci(move)
        alternatives = [
            candidate.uci()
            for candidate in board.generate_legal_moves()
            if candidate.uci().startswith(chosen_uci[:2]) and candidate.uci() != chosen_uci
        ]
        negative_indices = torch.tensor(
            [board.uci2idx(candidate) for candidate in alternatives],
            device=policy_out.device,
            dtype=torch.long,
        )
        k_positions = [int(positions[0, 1].item())]
        k_positions.extend(
            int(torch.as_tensor(board.uci_to_positions(candidate))[1].item())
            for candidate in alternatives
        )
        grouped_positions = torch.full(
            (1, 2, len(k_positions)),
            -1,
            device=policy_out.device,
            dtype=torch.long,
        )
        grouped_positions[0, 0, 0] = positions[0, 0]
        grouped_positions[0, 1] = torch.tensor(
            k_positions, device=policy_out.device, dtype=torch.long
        )
        return idx, probability, grouped_positions, negative_indices

    # Process positive move (positive injection)
    if not feature_specs_requested and positive_move_idx is not None:
        if order_mode == "group":
            (
                logit_idx_positive,
                logit_p_positive,
                move_positions_positive,
                group_negative_indices,
            ) = _selected_group(
                int(positive_move_idx)
            )
        else:
            logit_idx_positive, logit_p_positive, move_positions_positive = _selected_move(
                int(positive_move_idx)
            )
    
    # Process negative move (negative injection)
    if not feature_specs_requested and negative_move_idx is not None:
        logit_idx_negative, logit_p_negative, move_positions_negative = _selected_move(
            int(negative_move_idx)
        )

    if not feature_specs_requested:
        positive_for_grad = logit_idx_positive if positive_move_idx is not None else None
        negative_for_grad = (
            group_negative_indices
            if order_mode == "group"
            else logit_idx_negative if negative_move_idx is not None else None
        )
        (
            logit_vecs_q_positive,
            logit_vecs_k_positive,
            logit_vecs_q_negative,
            logit_vecs_k_negative,
        ) = ctx.compute_policy_qk_gradients(
            live_policy_logits,
            positive_for_grad,
            negative_for_grad,
        )
        if order_mode == "group" and group_negative_indices is not None and group_negative_indices.numel() > 0:
            assert logit_vecs_q_positive is not None
            assert logit_vecs_k_positive is not None
            assert logit_vecs_q_negative is not None
            assert logit_vecs_k_negative is not None
            logit_vecs_q_positive = (
                logit_vecs_q_positive
                + logit_vecs_q_negative.sum(dim=0, keepdim=True)
                / group_negative_indices.numel()
            )
            logit_vecs_k_positive = (
                logit_vecs_k_positive
                + logit_vecs_k_negative.sum(dim=0, keepdim=True)
                / group_negative_indices.numel()
            )
        if positive_move_idx is not None:
            logit_vecs_q = logit_vecs_q_positive
            logit_vecs_k = logit_vecs_k_positive
        else:
            logit_vecs_q = logit_vecs_q_negative
            logit_vecs_k = logit_vecs_k_negative
        assert logit_vecs_q is not None
        logit_vecs = torch.zeros_like(logit_vecs_q)
    
    # Determine the main logit information (for subsequent processing)
    if positive_move_idx is not None:
        logit_idx, logit_p, logit_vecs_q, logit_vecs_k, move_positions, logit_vecs = (
            logit_idx_positive, logit_p_positive, logit_vecs_q_positive, 
            logit_vecs_k_positive, move_positions_positive, logit_vecs_positive
        )
    elif negative_move_idx is not None:
        logit_idx, logit_p, logit_vecs_q, logit_vecs_k, move_positions, logit_vecs = (
            logit_idx_negative, logit_p_negative, logit_vecs_q_negative,
            logit_vecs_k_negative, move_positions_negative, logit_vecs_negative
        )
    elif feature_specs_requested:
        device = policy_out[0].device
        dtype = policy_out[0].dtype
        logit_idx = torch.zeros(0, dtype=torch.long, device=device)
        logit_p = torch.zeros(0, dtype=torch.float32, device=device)
        logit_vecs_q = torch.zeros(0, dtype=dtype, device=device)
        logit_vecs_k = torch.zeros(0, dtype=dtype, device=device)
        move_positions = torch.zeros((0, 2), dtype=torch.long, device=device)
        logit_vecs = torch.zeros(0, dtype=dtype, device=device)
    else:
        raise ValueError("No move_idx provided, and no feature_trace_specs provided, cannot determine the end point.")

    assert logit_idx is not None
    assert logit_p is not None
    
    # print(f'{move_positions = }')
    logger.info(
        f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}"
    )

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = total_active_feats + 2 * n_layers * n_pos + n_pos
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits

    requested_feature_nodes = total_active_feats if max_feature_nodes is None else max_feature_nodes
    max_feature_nodes = min(requested_feature_nodes, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    requested_sides = ("q", "k") if side.lower() == "both" else (side.lower(),)
    if any(requested not in ("q", "k") for requested in requested_sides):
        raise ValueError("side must be 'q', 'k', or 'both'")
    if mixed_precision_edges and policy_out.device.type == "cuda":
        edge_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        edge_dtype = torch.float32
    influence_sign_mode, _ = partial_influence_queue_config(order_mode)
    edge_matrices = {
        requested: torch.zeros(
            max_feature_nodes + n_logits,
            total_nodes,
            dtype=edge_dtype,
            device=policy_out.device,
        )
        for requested in requested_sides
    }
    normalized_matrices = {requested: torch.zeros_like(matrix) for requested, matrix in edge_matrices.items()}
    row_to_node_indices = {
        requested: torch.full(
            (max_feature_nodes + n_logits,),
            total_nodes,
            dtype=torch.long,
            device=policy_out.device,
        )
        for requested in requested_sides
    }

    # ========== Phase 3: logit attribution (write the first n_logits rows) ==========
    def bias_attr_now(model):
        vals = []
        for name, b in model._get_requires_grad_bias_params():
            if b.grad is not None and 'input' not in name:
                vals.append((b.detach() * b.grad).sum())
        return (
            torch.stack(vals).sum()
            if vals
            else torch.zeros((), device=policy_out.device, dtype=policy_out.dtype)
        )

    logger.info("Phase 3: Computing logit attributions")
    if feature_specs_requested:
        logger.info("Note: feature_trace_specs provided, skipping logit injection")
    elif positive_move_idx is not None and negative_move_idx is not None:
        logger.info("Note: Using DIFFERENTIAL gradient injection (positive - negative moves)")
        print("Note: Using DIFFERENTIAL gradient injection (positive - negative moves)")
    elif positive_move_idx is not None:
        logger.info("Note: Using POSITIVE gradient injection to find features that promote the logit")
        print("Note: Using POSITIVE gradient injection to find features that promote the logit")
    elif negative_move_idx is not None:
        logger.info("Note: Using NEGATIVE gradient injection to find features that suppress the logit")
        print("Note: Using NEGATIVE gradient injection to find features that suppress the logit")
    phase_start = time.time()
    model.zero_grad(set_to_none=True)

    rows_q_last = None
    rows_k_last = None

    if not feature_specs_requested:
        selected_positions = (
            move_positions_positive if positive_move_idx is not None else move_positions_negative
        )
        if order_mode == "group":
            q_positions = selected_positions[:, 0, :1]
            k_positions = selected_positions[:, 1, :]
        else:
            q_positions = selected_positions[:, 0:1]
            k_positions = selected_positions[:, 1:2]
        q_values = logit_vecs_q_positive if positive_move_idx is not None else logit_vecs_q_negative
        k_values = logit_vecs_k_positive if positive_move_idx is not None else logit_vecs_k_negative
        if order_mode == "group":
            k_values = k_values.unsqueeze(1).expand(-1, k_positions.shape[1], -1, -1)
        if positive_move_idx is not None and negative_move_idx is not None:
            q_positions = torch.cat((q_positions, move_positions_negative[:, 0:1]), dim=1)
            k_positions = torch.cat((k_positions, move_positions_negative[:, 1:2]), dim=1)
            q_values = torch.stack((logit_vecs_q_positive, logit_vecs_q_negative), dim=1)
            k_values = torch.stack((logit_vecs_k_positive, logit_vecs_k_negative), dim=1)
        rows_q, rows_k = ctx.compute_qk_vjp_batch(
            q_positions=q_positions,
            k_positions=k_positions,
            q_values=q_values,
            k_values=k_values,
            castle_tensor=torch.full((n_logits,), is_castle, device=policy_out.device),
            retain_graph=True,
        )
        rows_q_last, rows_k_last = rows_q.detach(), rows_k.detach()
        for requested, rows in (("q", rows_q), ("k", rows_k)):
            if requested not in edge_matrices:
                continue
            rows = rows.detach()
            edge_matrices[requested][:n_logits, :logit_offset] = rows.to(edge_dtype)
            normalized_matrices[requested][:n_logits, :logit_offset] = _normalize_rows(
                rows, influence_sign_mode
            ).to(edge_dtype)
            row_to_node_indices[requested][:n_logits] = torch.arange(
                logit_offset,
                logit_offset + n_logits,
                device=policy_out.device,
            )
    print(f"Logit attributions completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # ========== Phase 4: feature attribution (by side) ==========
    logger.info("Phase 4: Computing feature attributions")
    print("Phase 4: Computing feature attributions")
    phase_start = time.time()

    # Layer-wise means (for encoder_demean)
    with torch.no_grad():
        layer_means: List[torch.Tensor] = []
        for l in range(n_layers):
            tc = model.transcoders[str(l)]
            # W_E: [d_model, d_sae]  ->  W_E.T: [d_sae, d_model]
            mean_vec = tc.W_E.T.mean(dim=0)  # [d_model]
            layer_means.append(mean_vec)
        layer_means = torch.stack(layer_means, dim=0)  # [n_layers, d_model]
        layer_means = layer_means.to(device=tc_encoder_rows.device, dtype=tc_encoder_rows.dtype)

    lorsa_feat_layer, lorsa_feat_pos, lorsa_feat_idx = lorsa_activation_matrix.indices()
    tc_feat_layer, tc_feat_pos, tc_feat_idx = tc_activation_matrix.indices()
    n_lorsa_active = int(lorsa_activation_matrix._nnz())

    lorsa_lookup = {
        (int(layer), int(pos), int(feature_idx)): gid
        for gid, (layer, pos, feature_idx) in enumerate(
            zip(lorsa_feat_layer.tolist(), lorsa_feat_pos.tolist(), lorsa_feat_idx.tolist())
        )
    }
    tc_lookup = {
        (int(layer), int(pos), int(feature_idx)): n_lorsa_active + gid
        for gid, (layer, pos, feature_idx) in enumerate(
            zip(tc_feat_layer.tolist(), tc_feat_pos.tolist(), tc_feat_idx.tolist())
        )
    }

    def _resolve_feature_trace_spec(spec: FeatureTraceSpec) -> Optional[int]:
        """Resolve the user-provided spec into a global feature gid."""
        feature_type = spec.get("type", "tc")
        layer = spec.get("layer")
        position = spec.get("position")
        feature_idx = spec.get("feature_idx")
        if layer is None or position is None or feature_idx is None:
            logger.warning(f"[feature-trace] Invalid spec missing keys: {spec}")
            return None
        lookup_key = (int(layer), int(position), int(feature_idx))
        if feature_type == "lorsa":
            gid = lorsa_lookup.get(lookup_key)
            if gid is None:
                logger.warning(f"[feature-trace] Lorsa feature not found for spec: {spec}")
                return None
            return gid
        gid = tc_lookup.get(lookup_key)
        if gid is None:
            logger.warning(f"[feature-trace] TC feature not found for spec: {spec}")
            return None
        return gid

    resolved_feature_trace_gids: list[int] = []
    if feature_trace_specs:
        for item in feature_trace_specs:
            if isinstance(item, int):
                if 0 <= item < total_active_feats:
                    resolved_feature_trace_gids.append(int(item))
                else:
                    logger.warning(f"[feature-trace] gid {item} out of range 0..{total_active_feats-1}")
            elif isinstance(item, dict):
                gid = _resolve_feature_trace_spec(item)
                if gid is not None:
                    resolved_feature_trace_gids.append(gid)
            else:
                logger.warning(f"[feature-trace] Unsupported spec type: {type(item)}")
        resolved_feature_trace_gids = sorted(set(resolved_feature_trace_gids))

    has_feature_terminals = len(resolved_feature_trace_gids) > 0
    feature_queue_tensor: Optional[torch.Tensor] = None
    if has_feature_terminals:
        device_for_queue = tc_feat_layer.device if len(tc_feat_layer) > 0 else torch.device("cpu")
        feature_queue_tensor = torch.tensor(
            resolved_feature_trace_gids,
            dtype=torch.long,
            device=device_for_queue,
        )
    if feature_specs_requested and (positive_move_idx is not None or negative_move_idx is not None):
        raise ValueError("feature_trace_specs and move_idx are mutually exclusive, please select only one end point.")

    if feature_specs_requested and not resolved_feature_trace_gids:
        raise ValueError("all features in feature_trace_specs are not present in the activations of the current prompt, please check layer/pos/feature_idx.")

    has_feature_terminals = len(resolved_feature_trace_gids) > 0

    # —— Build allow mask: True=retain, False=discard —— #
    # Initially all features are allowed
    allow_mask = torch.ones(total_active_feats, dtype=torch.bool, device=policy_out.device)

    gid_to_layer = torch.cat(
        [
            2 * lorsa_feat_layer,
            2 * tc_feat_layer + 1,
        ],
        dim=0,
    )
    gid_to_pos = torch.cat([lorsa_feat_pos, tc_feat_pos], dim=0)
    gid_to_feature_id = torch.cat([lorsa_feat_idx, tc_feat_idx], dim=0)

    encoder_device = lorsa_encoder_rows.device
    encoder_dtype = lorsa_encoder_rows.dtype
    gid_to_encoder_rows = torch.cat(
        [
            lorsa_encoder_rows,
            tc_encoder_rows.to(device=encoder_device, dtype=encoder_dtype),
        ],
        dim=0,
    )

    bias_device = lorsa_encoder_bias.device
    bias_dtype = lorsa_encoder_bias.dtype
    gid_to_encoder_bias = torch.cat(
        [
            lorsa_encoder_bias,
            tc_encoder_bias.to(device=bias_device, dtype=bias_dtype),
        ],
        dim=0,
    )

    pattern_device = lorsa_attention_patterns.device
    pattern_dtype = lorsa_attention_patterns.dtype
    tc_identity_patterns = torch.nn.functional.one_hot(
        tc_feat_pos.to(device=pattern_device),
        num_classes=n_pos,
    ).to(dtype=pattern_dtype)
    gid_to_pattern = torch.cat(
        [
            lorsa_attention_patterns.to(device=pattern_device, dtype=pattern_dtype),
            tc_identity_patterns,
        ],
        dim=0,
    )

    def idx_to_layer(idx: torch.Tensor) -> torch.Tensor:
        return gid_to_layer.index_select(0, idx)

    def idx_to_pos(idx: torch.Tensor) -> torch.Tensor:
        return gid_to_pos.index_select(0, idx)

    def idx_to_feature_id(idx: torch.Tensor) -> torch.Tensor:
        return gid_to_feature_id.index_select(0, idx)

    def idx_to_encoder_rows(idx: torch.Tensor) -> torch.Tensor:
        rows = gid_to_encoder_rows.index_select(0, idx)
        if encoder_demean and idx.numel() > 0:
            tc_mask = idx >= n_lorsa_active
            if tc_mask.any():
                rows = rows.clone()
                tc_idx = idx[tc_mask] - n_lorsa_active
                tc_layers = tc_feat_layer.index_select(0, tc_idx)
                rows[tc_mask] -= layer_means.index_select(0, tc_layers)
        return rows

    def idx_to_encoder_bias(idx: torch.Tensor) -> torch.Tensor:
        return gid_to_encoder_bias.index_select(0, idx)

    def idx_to_pattern(idx: torch.Tensor) -> torch.Tensor:
        return gid_to_pattern.index_select(0, idx)

    def idx_to_activation_values(idx: torch.Tensor) -> torch.Tensor:
        is_lorsa = idx < len(lorsa_feat_layer)
        if is_lorsa.squeeze().item():
            return lorsa_activation_matrix.values()[idx]
        else:
            local_idx = (idx - len(lorsa_feat_layer)).to(torch.long)
            layer = tc_feat_layer[local_idx]
            feat_idx = tc_feat_idx[local_idx]

            if torch.is_tensor(layer):
                layer_key = str(int(layer.item()))
            else:
                layer_key = str(int(layer))

            tc = model.transcoders[layer_key]

            # If no b_E, use 0 placeholder
            b_E = getattr(tc, "b_E", None)
            if b_E is None:
                bias_val = 0.0
            else:
                bias_val = b_E[feat_idx.to(device=b_E.device, dtype=torch.long)]

            return tc_activation_matrix.values()[local_idx] - bias_val

    logger.info("Entering joint Q/K feature attribution loop")
    fa_result = run_joint_feature_attribution(
        ctx=ctx,
        requested_sides=requested_sides,
        edge_matrices=edge_matrices,
        normalized_matrices=normalized_matrices,
        row_to_node_indices=row_to_node_indices,
        total_active_feats=total_active_feats,
        max_feature_nodes=max_feature_nodes,
        update_interval=update_interval,
        selection_batch_size=batch_size,
        vjp_batch_size=vjp_batch_size,
        n_logits=n_logits,
        logit_p=logit_p,
        logit_offset=logit_offset,
        idx_to_layer=idx_to_layer,
        idx_to_pos=idx_to_pos,
        idx_to_encoder_rows=idx_to_encoder_rows,
        idx_to_pattern=idx_to_pattern,
        order_mode=order_mode,
        initial_queue=feature_queue_tensor,
    )

    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # ========== Phase 5: Packaging (each side) ==========
    print("Phase 5: Packaging")
    phase_start = time.time()
    def package_side(
        visited: torch.Tensor,
        edge_matrix: torch.Tensor,
        row_to_node_index: torch.Tensor,
        *,
        allow_mask: Optional[torch.Tensor] = None,   # ★ New: allowed features (gid level)
        move_idx: Optional[torch.Tensor] = None,     # ★ New: move index
        side: Optional[str] = None,                  # ★ New: 'q' or 'k'
        # New parameters for dense feature filtering
        mongo_client = None,
        sae_series: str = 'BT4-exp128',
        analysis_name: str = 'default',
        lorsa_feat_layer: torch.Tensor = None,
        lorsa_feat_idx: torch.Tensor = None,
        tc_feat_layer: torch.Tensor = None,
        tc_feat_idx: torch.Tensor = None,
        lorsa_activation_matrix: torch.sparse.Tensor = None,
        tc_activation_matrix: torch.sparse.Tensor = None,
        act_times_max: Optional[int] = None,
    ) -> Dict[str, Any]:
        total_nodes = logit_offset + n_logits
        
        # 1) First select the top max_feature_nodes most important features
        # ``visited`` is authoritative. This also handles feature-seeded traces,
        # which intentionally stop after their manual queue even when the
        # configured feature cap exceeds the number of active features.
        selected_features = torch.where(visited)[0].to(edge_matrix.device)
        
        # 2) Perform dense feature filtering on the selected features
        if mongo_client is not None and act_times_max is not None and len(selected_features) > 0:
            print(f'wash dense nodes in selected features only (side: {side})')
            print(f'Selected {len(selected_features)} features, checking for dense features...')
            
            # Initialize allow_mask (if not provided)
            if allow_mask is None:
                allow_mask = torch.ones(total_active_feats, dtype=torch.bool, device='cpu')
            
            # Use cache to avoid duplicate queries
            cache = {}
            
            def get_act_times_cached(L, F, feature_type):
                """Query activation times with cache"""
                key = (int(L), int(F), feature_type)
                if key not in cache:
                    try:
                        if feature_type == 'lorsa':
                            sae_name = f"lc0-lorsa-L{L}"
                        else:  # tc
                            sae_name = f"lc0_L{L}M_16x_k30_lr2e-03_auxk_sparseadam"
                        
                        fr = mongo_client.get_feature(sae_name, sae_series, F)
                        at = None
                        if fr:
                            for ana in fr.analyses:
                                if ana.name == analysis_name:
                                    at = ana.act_times
                                    break
                        cache[key] = at
                    except Exception:
                        cache[key] = None
                return cache[key]
            
            # Only check selected features
            dense_count = 0
            for gid in selected_features:
                gid = gid.item()
                if gid < lorsa_activation_matrix._nnz():
                    # Lorsa feature
                    layer = lorsa_feat_layer[gid].item()
                    feat_idx = lorsa_feat_idx[gid].item()
                    act_times = get_act_times_cached(layer, feat_idx, 'lorsa')
                    if act_times is not None and act_times > act_times_max:
                        allow_mask[gid] = False
                        dense_count += 1
                else:
                    # TC feature
                    tc_gid = gid - lorsa_activation_matrix._nnz()
                    layer = tc_feat_layer[tc_gid].item()
                    feat_idx = tc_feat_idx[tc_gid].item()
                    act_times = get_act_times_cached(layer, feat_idx, 'tc')
                    if act_times is not None and act_times > act_times_max:
                        allow_mask[gid] = False
                        dense_count += 1
            
            print(f"Filtered {dense_count} dense features out of {len(selected_features)} selected features")

        # print(f'{selected_features.shape = }') # 1024
            
        if allow_mask is not None:
            am = allow_mask.to(device=selected_features.device, dtype=torch.bool)
            keep_cols = am.index_select(0, selected_features)
            selected_features = selected_features[keep_cols]
            
        non_feature_nodes = torch.arange(total_active_feats, total_nodes, device=edge_matrix.device)
        col_read = torch.cat([selected_features, non_feature_nodes], dim=0)

        # Apply column selection (here already removed the disallowed features)
        edge_matrix_read = edge_matrix[:, col_read]

        # 2) Row sorting: first sort the rows by the natural order of row_to_node_index (stable), then move the "allowed feature rows" forward
        sort_idx = row_to_node_index.argsort()
        edge_matrix_sorted = edge_matrix_read.index_select(0, sort_idx)
        r2n_sorted = row_to_node_index.index_select(0, sort_idx)

        # Mark feature rows and logit rows
        is_feature_row = (r2n_sorted < total_active_feats)
        is_logit_row = (r2n_sorted >= logit_offset) & (r2n_sorted < total_nodes)

        # Calculate the gid corresponding to the feature rows, and filter the "rows" according to allow_mask
        if allow_mask is not None:
            feat_row_gids = r2n_sorted.masked_select(is_feature_row).to(torch.long)    # [n_feature_rows_sorted]
            allow_on_rows = allow_mask.to(r2n_sorted.device, dtype=torch.bool).index_select(0, feat_row_gids)
        else:
            # All allowed
            allow_on_rows = torch.ones(int(is_feature_row.sum().item()), dtype=torch.bool, device=r2n_sorted.device)

        # In the "sorted" coordinate system, get the allowed feature row indices, disallowed feature row indices, and logit row indices
        feat_row_idx_sorted = torch.nonzero(is_feature_row, as_tuple=True)[0]          # All feature rows (sorted coordinates)
        allow_feat_rows_sorted = feat_row_idx_sorted[allow_on_rows]                    # Allowed feature rows
        deny_feat_rows_sorted  = feat_row_idx_sorted[~allow_on_rows]                   # Disallowed feature rows
        logit_rows_sorted      = torch.nonzero(is_logit_row, as_tuple=True)[0]         # Logit rows

        # Goal: make the first max_feature_nodes rows all "allowed feature rows"
        # First construct a new row permutation: allowed feature rows -> disallowed feature rows -> other (here only logit rows)
        # So edge_matrix_perm[:max_feature_nodes] must be allowed feature rows (if enough)
        row_perm = torch.cat([allow_feat_rows_sorted, deny_feat_rows_sorted, logit_rows_sorted], dim=0)

        edge_matrix_perm = edge_matrix_sorted.index_select(0, row_perm)
        r2n_perm = r2n_sorted.index_select(0, row_perm)

        # debugging
        n_feature_rows_total = int(is_feature_row.sum().item())
        n_feature_rows_allowed = int(allow_feat_rows_sorted.numel())
        n_feature_cols_selected = int(selected_features.numel())

        print(f"[dbg] feature rows: total={n_feature_rows_total}, allowed={n_feature_rows_allowed}, "
            f"selected feature cols={n_feature_cols_selected}")

        if allow_mask is not None:
            present_feat_gids = torch.unique(r2n_sorted[is_feature_row].to(torch.long))
            allowed_feat_gids = torch.nonzero(allow_mask, as_tuple=True)[0].to(present_feat_gids.device)
            missing = allowed_feat_gids[~torch.isin(allowed_feat_gids, present_feat_gids)]
            print(f"[dbg] allowed-but-no-row gids (allowed in mask but absent as rows): {missing.numel()}")
        # end of debugging   
        
        # Count the number of available allowed rows, decide how many rows to fill
        allowed_rows_available = int(min(max_feature_nodes, allow_feat_rows_sorted.numel()))
        if allowed_rows_available < max_feature_nodes:
            print(f"[info] allowed feature rows = {allow_feat_rows_sorted.numel()}, "
                f"less than max_feature_nodes={max_feature_nodes}; "
                f"top block will use {allowed_rows_available} rows.")

        # A merged Q/K trace consumes the compact rows directly. Avoid two
        # temporary square matrices before allocating the final merged square.
        final_node_count = edge_matrix_perm.shape[1]
        full_edge_matrix = None
        if len(requested_sides) == 1:
            full_edge_matrix = torch.zeros(
                final_node_count,
                final_node_count,
                device=edge_matrix_perm.device,
                dtype=edge_matrix_perm.dtype,
            )
            if allowed_rows_available > 0:
                full_edge_matrix[:allowed_rows_available] = edge_matrix_perm[
                    :allowed_rows_available
                ]
            if n_logits > 0:
                full_edge_matrix[-n_logits:] = edge_matrix_perm[-n_logits:]

        # 4) Return the "permuted" row_to_node_index, ensuring DFS can correctly decode gid according to the new row order
        row_to_node_index_final = r2n_perm.clone()

        # Record the actual number of "meaningful feature rows" used in the metadata
        meta = {
            "n_logits": int(n_logits),
            "logit_offset": int(logit_offset),
            "final_node_count": int(final_node_count),
            "max_feature_rows": int(allowed_rows_available),
            "filtered_feature_cols": int(selected_features.numel()),
        }

        print(f"Packaging completed in {time.time() - phase_start:.2f}s")
        logger.info(f"Packaging completed in {time.time() - phase_start:.2f}s")

        # Process move_idx, extract the corresponding position information based on side
        side_move_positions = None
        if move_idx is not None and side is not None:
            try:
                if side.lower() == 'q':
                    # Extract q position (move_idx[i][0])
                    if move_idx.dim() == 3:  # Group mode: [batch, 2, M]
                        # Only take the first position (start position)
                        side_move_positions = move_idx[:, 0, 0]  # [batch]
                    elif move_idx.dim() == 2:  # Regular mode: [batch, 2]
                        side_move_positions = move_idx[:, 0]  # [batch]
                    else:
                        side_move_positions = torch.tensor([move_idx[i][0] for i in range(len(move_idx))], 
                                                         dtype=torch.long, device=move_idx.device)
                elif side.lower() == 'k':
                    # Extract k position (move_idx[i][1])
                    if move_idx.dim() == 3:  # Group mode: [batch, 2, M]
                        # Take all K positions
                        side_move_positions = move_idx[:, 1, :]  # [batch, M]
                    elif move_idx.dim() == 2:  # Regular mode: [batch, 2]
                        side_move_positions = move_idx[:, 1]  # [batch]
                    else:
                        side_move_positions = torch.tensor([move_idx[i][1] for i in range(len(move_idx))], 
                                                         dtype=torch.long, device=move_idx.device)
            except Exception as e:
                print(f"Warning: Failed to extract move positions for side {side}: {e}")
                side_move_positions = None

        # Collect activation information (if needed)
        side_activation_info = None
        if save_activation_info:
            side_activation_info = _collect_activation_info_after_forward(
                lorsa_activation_matrix=lorsa_activation_matrix,
                tc_activation_matrix=tc_activation_matrix,
                lorsa_attention_pattern=lorsa_attention_pattern,
                model=model,
                input_ids=input_ids,
                n_layers=n_layers,
                n_pos=n_pos,
                ctx=ctx,
                selected_features=selected_features
            )

        return {
            "selected_features": selected_features,       # Filtered (columns)
            "col_read": col_read,
            "edge_matrix": edge_matrix_perm,              # Rows are reordered
            "row_to_node_index": row_to_node_index_final, # Synchronized with edge_matrix
            "full_edge_matrix": full_edge_matrix,         # Square matrix (top = allowed feature rows, bottom = logit rows)
            "meta": meta,
            "activation_info": side_activation_info,      # Activation information for this side
            "move_positions": side_move_positions,
        }

    packaged_q = None
    packaged_k = None
    if 'q' in fa_result:
        packaged_q = package_side(
            visited=fa_result['q']['visited'],
            edge_matrix=fa_result['q']['edge_matrix'],
            row_to_node_index=fa_result['q']['row_to_node_index'],
            allow_mask=allow_mask,
            move_idx=move_positions,
            side='q',
            mongo_client=mongo_client,
            sae_series=sae_series,
            analysis_name=analysis_name,
            lorsa_feat_layer=lorsa_feat_layer,
            lorsa_feat_idx=lorsa_feat_idx,
            tc_feat_layer=tc_feat_layer,
            tc_feat_idx=tc_feat_idx,
            lorsa_activation_matrix=lorsa_activation_matrix,
            tc_activation_matrix=tc_activation_matrix,
            act_times_max=act_times_max,
        )
    if 'k' in fa_result:
        packaged_k = package_side(
            visited=fa_result['k']['visited'],
            edge_matrix=fa_result['k']['edge_matrix'],
            row_to_node_index=fa_result['k']['row_to_node_index'],
            allow_mask=allow_mask,
            move_idx=move_positions,
            side='k',
            mongo_client=mongo_client,
            sae_series=sae_series,
            analysis_name=analysis_name,
            lorsa_feat_layer=lorsa_feat_layer,
            lorsa_feat_idx=lorsa_feat_idx,
            tc_feat_layer=tc_feat_layer,
            tc_feat_idx=tc_feat_idx,
            lorsa_activation_matrix=lorsa_activation_matrix,
            tc_activation_matrix=tc_activation_matrix,
            act_times_max=act_times_max,
        )

    if rows_q_last is None:
        rows_q_last = torch.zeros(1, logit_offset, device=tc_feat_layer.device)
    if rows_k_last is None:
        rows_k_last = torch.zeros(1, logit_offset, device=tc_feat_layer.device)

    rows_q_raw = rows_q_last
    rows_k_raw = rows_k_last

    mask_float = allow_mask.to(dtype=rows_q_raw.dtype, device=rows_q_raw.device).view(1, -1)
    rows_q_filtered = rows_q_raw.clone()
    rows_k_filtered = rows_k_raw.clone()
    # Only mask feature section (0..total_active_feats-1)
    rows_q_filtered[:, :total_active_feats] *= mask_float
    rows_k_filtered[:, :total_active_feats] *= mask_float
    # Activation information has been collected in Phase 1 (if save_activation_info=True)

    feature_seed_trace: Optional[Dict[str, torch.Tensor]] = None
    if resolved_feature_trace_gids:
        print(f"Computing feature-seeded trace for {len(resolved_feature_trace_gids)} features")
        gid_tensor = torch.tensor(
            resolved_feature_trace_gids,
            dtype=torch.long,
            device=tc_feat_layer.device,
        )
        feature_seed_trace = run_feature_seed_trace(
            ctx=ctx,
            model=model,
            feature_gids=gid_tensor,
            idx_to_layer=idx_to_layer,
            idx_to_pos=idx_to_pos,
            idx_to_encoder_rows=idx_to_encoder_rows,
            idx_to_encoder_bias=idx_to_encoder_bias,
            idx_to_pattern=idx_to_pattern,
            bias_attr_now=bias_attr_now,
        )

    # ========== Return unified ==========
    graph_bundle = {
        "meta": {
            "time_sec": float(time.time() - start_time),
            "side": side,
            "verbose": verbose,
            "use_legal_moves_only": use_legal_moves_only,
            "offload": offload,
        },
        "input": {
            "input_ids": prompt,
            "input_embedding": token_vecs,  # Add input_embedding (hook_embed)
        },
        "logits": {
            "indices": logit_idx,
            "probabilities": logit_p,
            "move_positions": move_positions,
            "n_logits": int(n_logits),
        },
        "dims": {
            "n_layers": int(n_layers),
            "n_pos": int(n_pos),
            "logit_offset": int(logit_offset),
            "total_active_feats": int(total_active_feats),
            "max_feature_nodes": int(max_feature_nodes),
        },
        "lorsa_activations": {
            "indices": lorsa_activation_matrix.indices().T,   # [nnz, 3]
            "values": lorsa_activation_matrix.values(),       # [nnz]
            "lorsa_activation_matrix": lorsa_activation_matrix,
        },
        "tc_activations": {
            "indices": tc_activation_matrix.indices().T,   # [nnz, 3]
            "values": tc_activation_matrix.values(),       # [nnz]
            "tc_activation_matrix": tc_activation_matrix,      
        },
        "q": packaged_q,   # Or None
        "k": packaged_k,   # Or None

        # rows_*: Filtered version (for DFS root selection); also include raw for debugging
        "rows_q": rows_q_filtered,
        "rows_k": rows_k_filtered,
        "rows_q_raw": rows_q_raw,
        "rows_k_raw": rows_k_raw,

        # For downstream debugging/reuse
        "feature_allow_mask": allow_mask,
        "feature_seed_trace": feature_seed_trace,
        "feature_trace_specs": list(feature_trace_specs) if feature_trace_specs else None,
        "feature_trace_gids": resolved_feature_trace_gids,
        
        # Activation information (if saved)
        "activation_info": {
            "q": packaged_q["activation_info"] if packaged_q and "activation_info" in packaged_q else None,
            "k": packaged_k["activation_info"] if packaged_k and "activation_info" in packaged_k else None,
        } if save_activation_info else None,
    }

    return graph_bundle


def _collect_activation_info_after_forward(
    lorsa_activation_matrix: torch.sparse.Tensor,
    tc_activation_matrix: torch.sparse.Tensor,
    lorsa_attention_pattern: torch.Tensor,
    model,
    input_ids: torch.Tensor,
    n_layers: int,
    n_pos: int,
    ctx,
    selected_features: torch.Tensor
) -> Dict[str, Any]:
    """Collect activation information after forward propagation, including the actual z_patterns
    
    Args:
        lorsa_activation_matrix: Lorsa feature activation matrix [n_layers, n_pos, n_features]
        tc_activation_matrix: TC feature activation matrix [n_layers, n_pos, n_features] 
        lorsa_attention_pattern: Lorsa attention pattern [n_layers, n_qk_heads, n_pos, n_pos]
        model: Model instance
        input_ids: Input token ids
        n_layers: Number of layers
        n_pos: Sequence length
        ctx: AttributionContext instance (forward propagation completed, activations cached)
        selected_features: Selected feature global ID list
        
    Returns:
        Dictionary containing activation information for each selected feature, compatible with the frontend UI
    """
    # ========== Process Lorsa Features activation information ==========
    lorsa_indices = lorsa_activation_matrix.indices()  # [3, nnz] - (layer, pos, head_idx)
    lorsa_values = lorsa_activation_matrix.values()    # [nnz]
    
    # Store activation information for each selected feature
    features_activation_info = []
    
    # Convert selected_features to a set on CPU for fast lookup
    selected_features_set = set(selected_features.cpu().numpy().tolist())
    
    # Process each Lorsa feature, only process selected ones
    for i in range(lorsa_activation_matrix._nnz()):
        # The global ID of Lorsa features is i
        if i not in selected_features_set:
            continue
            
        layer = lorsa_indices[0, i].item()
        pos = lorsa_indices[1, i].item()
        head_idx = lorsa_indices[2, i].item()
        activation_value = lorsa_values[i].item()
        
        # Create an activation array for the current feature at 64 positions
        feature_activations = [0.0] * 64
        if 0 <= pos < 64:
            feature_activations[pos] = activation_value
        
        # Initialize z_pattern for the current feature
        feature_z_pattern_indices = [[], []]  # [q_positions, k_positions]
        feature_z_pattern_values = []
        
        # ========== Calculate the z_pattern for the current Lorsa feature ==========
        try:
            # Get the corresponding Lorsa SAE
            lorsa_sae = model.lorsas[layer]
            
            # Get the activation of the current layer from the cached activations
            layer_activation = ctx._resid_activations[layer * 2]  # attention input
            
            if layer_activation is not None:
                # Calculate the z_pattern for the current head
                z_pattern = lorsa_sae.encode_z_pattern_for_head(
                    layer_activation,  # [1, seq, d_model]
                    torch.tensor([head_idx], device=layer_activation.device)
                )  # [1, n_ctx, n_ctx]
                
                # Only take the pattern at the current position
                z_pattern_for_pos = z_pattern[0, pos, :]  # [n_ctx]
                
                # Apply the activation value weights
                z_pattern_weighted = z_pattern_for_pos * activation_value
                
                # Filter small values
                small_mask = z_pattern_weighted.abs() < 1e-3 * abs(activation_value)
                z_pattern_weighted = z_pattern_weighted.masked_fill(small_mask, 0)
                
                # Convert to sparse format - fix dimension error
                nonzero_result = z_pattern_weighted.nonzero()
                if nonzero_result.numel() > 0:
                    nonzero_indices = nonzero_result.squeeze(-1) if nonzero_result.shape[-1] == 1 else nonzero_result[:, 0]
                    nonzero_values = z_pattern_weighted[nonzero_indices]
                else:
                    nonzero_indices = torch.tensor([], dtype=torch.long, device=z_pattern_weighted.device)
                    nonzero_values = torch.tensor([], dtype=z_pattern_weighted.dtype, device=z_pattern_weighted.device)
                
                if len(nonzero_indices) > 0:
                    # Add q position (start) and k position (focus position) for each non-zero value
                    for k_pos, value in zip(nonzero_indices.detach().cpu().numpy(), nonzero_values.detach().cpu().numpy()):
                        feature_z_pattern_indices[0].append(pos)  # q position (start)
                        feature_z_pattern_indices[1].append(int(k_pos))  # k position (focus position)
                        feature_z_pattern_values.append(float(value))
                        
            else:
                print(f"Warning: No cached activation for layer {layer}")
                        
        except Exception as e:
            print(f"Warning: Failed to compute z_pattern for Lorsa layer {layer}, head {head_idx}: {e}")
            # Fallback to using the simplified version of attention_pattern
            try:
                qk_head_idx = head_idx // (model.lorsas[layer].cfg.n_ov_heads // model.lorsas[layer].cfg.n_qk_heads)
                attention_pattern = lorsa_attention_pattern[layer, qk_head_idx, pos, :]
                
                weighted_pattern = attention_pattern * activation_value
                small_pattern_mask = weighted_pattern.abs() < 1e-3 * abs(activation_value)
                weighted_pattern = weighted_pattern.masked_fill(small_pattern_mask, 0)
                
                # Fix dimension error - handle empty tensor case
                nonzero_result = weighted_pattern.nonzero()
                if nonzero_result.numel() > 0:
                    nonzero_indices = nonzero_result.squeeze(-1) if nonzero_result.shape[-1] == 1 else nonzero_result[:, 0]
                    nonzero_values = weighted_pattern[nonzero_indices]
                else:
                    nonzero_indices = torch.tensor([], dtype=torch.long, device=weighted_pattern.device)
                    nonzero_values = torch.tensor([], dtype=weighted_pattern.dtype, device=weighted_pattern.device)
                
                if len(nonzero_indices) > 0:
                    # Add q position (start) and k position (focus position) for each non-zero value
                    for k_pos, value in zip(nonzero_indices.detach().cpu().numpy(), nonzero_values.detach().cpu().numpy()):
                        feature_z_pattern_indices[0].append(pos)
                        feature_z_pattern_indices[1].append(int(k_pos))
                        feature_z_pattern_values.append(float(value))
                        
            except Exception as e2:
                print(f"Warning: Also failed fallback computation for layer {layer}, head {head_idx}: {e2}")
        
        feature_info = {
            "featureId": i,
            "type": "lorsa",
            "layer": layer,
            "position": pos,
            "head_idx": head_idx,
            "activation_value": activation_value,
            "activations": feature_activations,
            "zPatternIndices": feature_z_pattern_indices,
            "zPatternValues": feature_z_pattern_values
        }
        features_activation_info.append(feature_info)
    
    tc_indices = tc_activation_matrix.indices()  # [3, nnz] - (layer, pos, feature_idx)
    tc_values = tc_activation_matrix.values()    # [nnz]
    
    tc_id_offset = lorsa_activation_matrix._nnz()
    
    for i in range(tc_activation_matrix._nnz()):
        tc_global_id = tc_id_offset + i
        if tc_global_id not in selected_features_set:
            continue
            
        layer = tc_indices[0, i].item()
        pos = tc_indices[1, i].item()
        feature_idx = tc_indices[2, i].item()
        activation_value = tc_values[i].item()
        
        feature_activations = [0.0] * 64
        if 0 <= pos < 64:
            feature_activations[pos] = activation_value
        
        feature_z_pattern_indices = [[], []]
        feature_z_pattern_values = []
        
        feature_info = {
            "featureId": tc_global_id,
            "type": "tc",
            "layer": layer,
            "position": pos,
            "feature_idx": feature_idx,
            "activation_value": activation_value,
            "activations": feature_activations,
            "zPatternIndices": feature_z_pattern_indices,
            "zPatternValues": feature_z_pattern_values
        }
        features_activation_info.append(feature_info)
    
    activation_info = {
        "features": features_activation_info,
        
        "meta": {
            "total_features": len(features_activation_info),
            "n_lorsa_features": lorsa_activation_matrix._nnz(),
            "n_tc_features": tc_activation_matrix._nnz(),
            "n_layers": n_layers,
            "n_pos": n_pos,
            "sequence": input_ids,
            "collected_after_forward": True
        }
    }
    
    print(f"Collected activation info for {len(features_activation_info)} features: {lorsa_activation_matrix._nnz()} Lorsa + {tc_activation_matrix._nnz()} TC")
    lorsa_z_patterns = sum(len(f["zPatternValues"]) for f in features_activation_info if f["type"] == "lorsa")
    print(f"Total z_pattern entries: {lorsa_z_patterns}")
    
    return activation_info



    # graph = Graph(
    #     input_string=model.tokenizer.decode(input_ids),
    #     input_tokens=input_ids,
    #     logit_tokens=logit_idx,
    #     logit_probabilities=logit_p,
    #     lorsa_active_features=lorsa_activation_matrix.indices().T,
    #     lorsa_activation_values=lorsa_activation_matrix.values(),
    #     clt_active_features=tc_activation_matrix.indices().T,
    #     clt_activation_values=tc_activation_matrix.values(),
    #     selected_features=selected_features,
    #     adjacency_matrix=full_edge_matrix,
    #     cfg=model.cfg,
    #     scan=None,
    # )

    # total_time = time.time() - start_time
    # logger.info(f"Attribution completed in {total_time:.2f}s")

    # return graph


def run_feature_attribution(
    *,
    ctx,
    model,
    tc_activation_matrix: torch.Tensor,
    total_active_feats: int,
    max_feature_nodes: int,
    update_interval: int,
    batch_size: int,
    n_logits: int,
    logit_p: torch.Tensor,
    logit_offset: int,
    # These mapping are defined outside
    idx_to_layer,
    idx_to_pos,
    idx_to_encoder_rows,
    idx_to_encoder_bias,
    idx_to_pattern,
    idx_to_activation_values,
    compute_partial_influences,
    bias_attr_now,
    # Only provide one edge_matrix and row_to_node_index
    edge_matrix: torch.Tensor,
    row_to_node_index: torch.Tensor,
    logger=None,
    order_mode: str = 'abs',
    initial_queue: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute feature attribution for a single side.
    Returns:
      - visited: [total_active_feats] bool tensor (which features were visited/enqueued)
      - edge_matrix: Matrix after computation (rows = feature + logit, columns = all nodes) (in-place same as input object)
      - row_to_node_index: Mapping after computation (row -> global gid) (in-place same as input object)
    """
    influence_sign_mode, feature_descending = partial_influence_queue_config(order_mode)
    if order_mode == "negative":
        if logger:
            logger.info("order_mode=negative: signed partial influence, ascending sort")
    elif order_mode == "positive":
        if logger:
            logger.info("order_mode=positive: signed partial influence, descending sort")
    elif order_mode == "abs":
        if logger:
            logger.info("order_mode=abs: |edge| partial influence, descending sort")

    if logger:
        logger.info(f"Phase: Computing feature attributions")

    # The computation graph has been rebuilt outside, so there is no need to clear again
    # print("Clear the computation state in ctx...")
    # model.zero_grad(set_to_none=True)
    # if hasattr(ctx, 'clear'):
    #     ctx.clear()
    # elif hasattr(ctx, 'reset'):
    #     ctx.reset()

    phase_start = time.time()
    st = n_logits  # Row start: first put logit rows
    debug_conservation = os.environ.get("ATTRIBUTION_DEBUG_CONSERVATION", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0
    index_device = logit_p.device

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation")

    manual_queue: deque[torch.Tensor] = deque()
    if initial_queue is not None and initial_queue.numel() > 0:
        unique_manual = torch.unique(initial_queue.cpu())
        manual_queue = _batched_index_queue(unique_manual, batch_size)

    auto_queue: deque[torch.Tensor] = deque()

    while n_visited < max_feature_nodes:
        if manual_queue:
            idx_batch = manual_queue.popleft()
        else:
            if not auto_queue:
                if n_logits == 0:
                    break  # No logit seed and no manual queue, cannot continue
                if max_feature_nodes == total_active_feats:
                    pending = torch.nonzero(~visited, as_tuple=True)[0]
                else:
                    influences = compute_partial_influences(
                        edge_matrix[:st],
                        logit_p,
                        row_to_node_index[:st],
                        sign_mode=influence_sign_mode,
                    )
                    queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
                    feature_scores = influences[:total_active_feats]
                    available = torch.nonzero(~visited, as_tuple=True)[0].to(feature_scores.device)
                    if available.numel() == 0:
                        break
                    if queue_size >= available.numel():
                        pending = available.cpu()
                    else:
                        available_scores = feature_scores.index_select(0, available)
                        _, top_local = torch.topk(
                            available_scores,
                            k=queue_size,
                            largest=feature_descending,
                            sorted=True,
                        )
                        pending = available.index_select(0, top_local).cpu()

                if pending.numel() == 0:
                    break
                auto_queue = _batched_index_queue(pending, batch_size)
            idx_batch = auto_queue.popleft()
            if idx_batch.numel() == 0:
                continue

        if idx_batch.numel() == 0:
            continue
        n_visited += len(idx_batch)
        idx_batch_device = idx_batch.to(device=index_device)
        layers = idx_to_layer(idx_batch_device)
        positions = idx_to_pos(idx_batch_device)
        inject_values = idx_to_encoder_rows(idx_batch_device).detach()
        encoder_bias = idx_to_encoder_bias(idx_batch_device)
        attn_patterns = idx_to_pattern(idx_batch_device)

        if isinstance(attn_patterns, torch.Tensor):
            attn_patterns = attn_patterns.detach()

        model.zero_grad(set_to_none=True)

        has_more_in_this_phase = (n_visited < max_feature_nodes)
        rows_feature = ctx.compute_batch(
            layers=layers,
            positions=positions,
            inject_values=inject_values,
            attention_patterns=attn_patterns,
            retain_graph=has_more_in_this_phase,
        )

        model_bias_attr = bias_attr_now(model)
        _ = model_bias_attr + encoder_bias
        
        # DEBUG MODE
        if debug_conservation:
            n_layers, n_pos, _ = tc_activation_matrix.shape
            rows_feature_cpu = rows_feature.detach().cpu()
            activation_values = idx_to_activation_values(idx_batch_device).detach().cpu()
            encoder_bias_cpu = encoder_bias.detach().cpu()
            model_bias_value = float(model_bias_attr.detach().cpu().item())

            feature_slice = slice(0, total_active_feats)
            error_slice = slice(total_active_feats, total_active_feats + 2 * n_layers * n_pos)
            token_slice = slice(total_active_feats + 2 * n_layers * n_pos, logit_offset)

            for row_idx, gid in enumerate(idx_batch.tolist()):
                feature_contribution = float(rows_feature_cpu[row_idx, feature_slice].sum().item())
                error_contribution = float(rows_feature_cpu[row_idx, error_slice].sum().item())
                token_contribution = float(rows_feature_cpu[row_idx, token_slice].sum().item())
                edge_contribution = feature_contribution + error_contribution + token_contribution
                overall_activation = float(activation_values[row_idx].item())
                encoder_bias_value = float(encoder_bias_cpu[row_idx].item())

                approx_without_encoder_bias = edge_contribution + model_bias_value
                approx_with_encoder_bias = approx_without_encoder_bias + encoder_bias_value
                close_without_encoder_bias = torch.isclose(
                    torch.tensor(overall_activation),
                    torch.tensor(approx_without_encoder_bias),
                    rtol=1e-1,
                    atol=1e-4,
                ).item()
                close_with_encoder_bias = torch.isclose(
                    torch.tensor(overall_activation),
                    torch.tensor(approx_with_encoder_bias),
                    rtol=1e-1,
                    atol=1e-4,
                ).item()

                print(f"[ATTRIBUTION_DEBUG] gid={gid} layer={int(layers[row_idx].item())} pos={int(positions[row_idx].item())}")
                print(f"[ATTRIBUTION_DEBUG] attention_pattern={attn_patterns[row_idx] if isinstance(attn_patterns, torch.Tensor) else None}")
                print(f"[ATTRIBUTION_DEBUG] model_bias_contribution={model_bias_value:.6f}")
                print(f"[ATTRIBUTION_DEBUG] encoder_bias_contribution={encoder_bias_value:.6f}")
                print(f"[ATTRIBUTION_DEBUG] feature_contribution={feature_contribution:.6f}")
                print(f"[ATTRIBUTION_DEBUG] error_contribution={error_contribution:.6f}")
                print(f"[ATTRIBUTION_DEBUG] token_contribution={token_contribution:.6f}")
                print(f"[ATTRIBUTION_DEBUG] edge_contribution_sum={edge_contribution:.6f}")
                print(f"[ATTRIBUTION_DEBUG] overall_activation={overall_activation:.6f}")
                print(
                    f"[ATTRIBUTION_DEBUG] activation≈edges+model_bias: "
                    f"{close_without_encoder_bias} ({approx_without_encoder_bias:.6f})"
                )
                print(
                    f"[ATTRIBUTION_DEBUG] activation≈edges+model_bias+encoder_bias: "
                    f"{close_with_encoder_bias} ({approx_with_encoder_bias:.6f})"
                )
                print("[ATTRIBUTION_DEBUG] --------------------------------")

        bs = rows_feature.shape[0]
        end = st + bs
        edge_matrix[st:end, :logit_offset] = rows_feature.detach().cpu()
        row_to_node_index[st:end] = idx_batch.to(
            device=row_to_node_index.device,
            dtype=row_to_node_index.dtype,
        )
        visited[idx_batch] = True
        st = end
        pbar.update(len(idx_batch))

    pbar.close()
    if logger:
        logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    return {
        "visited": visited,                     # [total_active_feats] bool
        "edge_matrix": edge_matrix,
        "row_to_node_index": row_to_node_index,
        "tc_activation_matrix": tc_activation_matrix,
    }


def run_feature_seed_trace(
    *,
    ctx: AttributionContext,
    model: ReplacementModel,
    feature_gids: torch.Tensor,
    idx_to_layer: Callable[[torch.Tensor], torch.Tensor],
    idx_to_pos: Callable[[torch.Tensor], torch.Tensor],
    idx_to_encoder_rows: Callable[[torch.Tensor], torch.Tensor],
    idx_to_encoder_bias: Callable[[torch.Tensor], torch.Tensor],
    idx_to_pattern: Callable[[torch.Tensor], torch.Tensor],
    bias_attr_now: Callable[[ReplacementModel], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Execute one gradient injection for a specified feature gid and return the corresponding edge rows."""

    if feature_gids.numel() == 0:
        return {
            "feature_gids": torch.empty(0, dtype=torch.long),
            "edge_rows": torch.empty(0),
            "encoder_bias": torch.empty(0),
        }

    layers = idx_to_layer(feature_gids)
    positions = idx_to_pos(feature_gids)
    inject_values = idx_to_encoder_rows(feature_gids).detach()
    encoder_bias = idx_to_encoder_bias(feature_gids)
    attn_patterns = idx_to_pattern(feature_gids)
    if isinstance(attn_patterns, torch.Tensor):
        attn_patterns = attn_patterns.detach()

    rows = ctx.compute_vjp_batch(
        layers=layers,
        positions=positions,
        inject_values=inject_values,
        attention_patterns=attn_patterns,
        retain_graph=True,
    )

    return {
        "feature_gids": feature_gids.detach().cpu(),
        "edge_rows": rows.detach().cpu(),
        "encoder_bias": encoder_bias.detach().cpu(),
    }

def merge_qk_graph(attribution_result):
    pkg_q = attribution_result.get('q')
    pkg_k = attribution_result.get('k')
    assert pkg_q is not None and pkg_k is not None, "side='both' requires both q/k branches"

    # Shared dimension information
    dims = attribution_result['dims']
    total_active_feats = dims['total_active_feats']
    logit_offset       = dims['logit_offset']
    n_logits           = attribution_result['logits']['n_logits']
    total_nodes        = logit_offset + n_logits

    device = pkg_q["edge_matrix"].device
    sel_q = pkg_q['selected_features'].to(device)
    sel_k = pkg_k['selected_features'].to(device)
    selected_union = torch.unique(torch.cat([sel_q, sel_k], dim=0))

    non_feature_cols = torch.arange(total_active_feats, total_nodes, dtype=torch.long, device=device)
    col_read_merged = torch.cat([selected_union, non_feature_cols], dim=0)
    final_node_count = col_read_merged.numel()
    full_edge_matrix_merged = pkg_q["edge_matrix"].new_zeros(
        (final_node_count, final_node_count)
    )
    target_columns, target_order = torch.sort(col_read_merged)

    def column_offsets(pkg):
        source_columns = pkg["col_read"].to(device)
        source_in_sorted = torch.searchsorted(target_columns, source_columns)
        return target_order.index_select(0, source_in_sorted)

    def add_side(pkg):
        edge_matrix = pkg["edge_matrix"].to(device)
        col_offsets = column_offsets(pkg)
        row_to_node = pkg["row_to_node_index"].to(device)
        feature_rows = torch.nonzero(row_to_node < total_active_feats, as_tuple=True)[0]
        gids = row_to_node.index_select(0, feature_rows)
        retained = torch.isin(gids, selected_union)
        feature_rows = feature_rows[retained]
        gids = gids[retained]
        if feature_rows.numel() > 0:
            selected_offsets = torch.searchsorted(selected_union, gids)
            feature_values = edge_matrix.index_select(0, feature_rows)
            full_edge_matrix_merged[
                selected_offsets[:, None], col_offsets[None, :]
            ] += feature_values
        if n_logits > 0:
            logit_offsets = torch.arange(
                final_node_count - n_logits,
                final_node_count,
                device=device,
            )
            full_edge_matrix_merged[
                logit_offsets[:, None], col_offsets[None, :]
            ] += edge_matrix[-n_logits:]

    add_side(pkg_q)
    add_side(pkg_k)

    # Merge activation information
    merged_activation_info = None
    if attribution_result.get("activation_info") is not None:
        activation_info = attribution_result["activation_info"]
        q_activation_info = activation_info.get("q")
        k_activation_info = activation_info.get("k")
        
        if q_activation_info is not None:
            merged_activation_info = q_activation_info.copy()
            
            # If k side also has activation information, it needs to be merged
            if k_activation_info is not None:
                # Merge features list
                if "features" in merged_activation_info and "features" in k_activation_info:
                    # Create a mapping from feature ID to activation information, avoiding duplicates
                    q_features_dict = {f["featureId"]: f for f in merged_activation_info["features"]}
                    k_features_dict = {f["featureId"]: f for f in k_activation_info["features"]}
                    
                    # Merge features, prioritize q side information (because q side is usually more complete)
                    all_feature_ids = set(q_features_dict.keys()) | set(k_features_dict.keys())
                    merged_features = []
                    
                    for feature_id in sorted(all_feature_ids):
                        if feature_id in q_features_dict:
                            merged_features.append(q_features_dict[feature_id])
                        elif feature_id in k_features_dict:
                            merged_features.append(k_features_dict[feature_id])
                    
                    merged_activation_info["features"] = merged_features
                
                # Update meta information
                if "meta" in merged_activation_info and "meta" in k_activation_info:
                    merged_activation_info["meta"]["total_features"] = len(merged_activation_info["features"])
                    merged_activation_info["meta"]["merged_from_qk"] = True
        elif k_activation_info is not None:
            # If only k side has activation information, use k side
            merged_activation_info = k_activation_info.copy()

    # Return the components you need for the graph
    return {
        "adjacency_matrix": full_edge_matrix_merged,
        "selected_features": selected_union,
        # Use k side's move position information, if k side does not exist, use q side
        "logit_position": pkg_k["move_positions"] if pkg_k and "move_positions" in pkg_k else (pkg_q["move_positions"] if pkg_q and "move_positions" in pkg_q else None),
        "col_read": col_read_merged,  # If needed for subsequent alignment
        "activation_info": merged_activation_info,  # Merged activation information
    }


def find_feature_gid(attribution_result, layer, feature_id, position, feature_type='tc'):
    """
    Find the global ID (gid) of a specified feature in the attribution result
    
    Args:
        attribution_result: return value of attribute() function
        layer: layer index
        feature_id: feature ID
        position: position index
        feature_type: feature type ('tc' or 'lorsa')
    
    Returns:
        tuple: (gid, activation_value) or (None, None)
    """
    if feature_type == 'tc':
        activations = attribution_result['tc_activations']
        indices = activations['indices']  # [nnz, 3]
        values = activations['values']    # [nnz]
        mask = (indices[:, 0] == layer) & (indices[:, 1] == position) & (indices[:, 2] == feature_id)
        if mask.any():
            matching_idx = mask.nonzero(as_tuple=True)[0][0]
            # TC features' gid needs to add the offset of LORSA features
            lorsa_offset = attribution_result['lorsa_activations']['indices'].shape[0]
            gid = lorsa_offset + matching_idx.item()
            activation_value = values[matching_idx].item()
            return gid, activation_value
    elif feature_type == 'lorsa':
        activations = attribution_result['lorsa_activations']
        indices = activations['indices']  # [nnz, 3]
        values = activations['values']    # [nnz]
        mask = (indices[:, 0] == layer) & (indices[:, 1] == position) & (indices[:, 2] == feature_id)
        if mask.any():
            matching_idx = mask.nonzero(as_tuple=True)[0][0]
            # LORSA features' gid is the index of them in the activation matrix
            gid = matching_idx.item()
            activation_value = values[matching_idx].item()
            return gid, activation_value
    
    return None, None
