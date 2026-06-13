"""
Low Rank Sparse Attention Sparse Dictionary Implementation

This module implements a Low Rank Sparse Attention layer as a sparse dictionary,
inheriting from SparseDictionary and supporting head parallelization.
"""

import math
from typing import Any, Literal, Optional, Sequence, Tuple, Union, cast, overload

import einops
import torch
import torch.distributed.tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformer_lens.components import Attention, GroupedQueryAttention
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from llamascopium.models.protocols import (
    DatasetNormStandardizable,
    EncoderBiasInitializable,
    NormComputing,
    NormConstrainable,
)
from llamascopium.models.sparse_dictionary import (
    SparseDictionary,
    SparseDictionaryConfig,
    register_sae_config,
    register_sae_model,
)
from llamascopium.utils.distributed import DimMap, masked_fill, mesh_dim_size
from llamascopium.utils.logging import get_distributed_logger

logger = get_distributed_logger("lorsa")


@register_sae_config("lorsa")
class LorsaConfig(SparseDictionaryConfig):
    """Configuration for Low Rank Sparse Attention."""

    sae_type: str = "lorsa"

    hook_point_in: str
    hook_point_out: str

    # Attention dimensions
    n_qk_heads: int
    d_qk_head: int
    positional_embedding_type: Literal["rotary", "none"] = "rotary"
    rotary_dim: int
    rotary_base: int = 10000
    rotary_base_local: int | None = None
    rotary_adjacent_pairs: bool = True
    rotary_scale: int = 1
    use_NTK_by_parts_rope: bool = False
    NTK_by_parts_factor: float = 1.0
    NTK_by_parts_low_freq_factor: float = 1.0
    NTK_by_parts_high_freq_factor: float = 1.0
    old_context_len: int = 2048
    n_ctx: int
    attn_type: Literal["global", "local"] = "global"
    window_size: int | None = None

    # Attention settings
    attn_scale: float | None = None
    use_post_qk_ln: bool = False
    normalization_type: Literal["LN", "RMS"] | None = None
    eps: float = 1e-6

    @property
    def n_ov_heads(self) -> int:
        return self.d_sae

    @property
    def ov_group_size(self) -> int:
        return self.n_ov_heads // self.n_qk_heads

    @property
    def associated_hook_points(self) -> list[str]:
        """All hook points used by Lorsa."""
        return [self.hook_point_in, self.hook_point_out]

    @property
    def hooks_in(self) -> list[str]:
        return [self.hook_point_in]

    @property
    def hooks_out(self) -> list[str]:
        return [self.hook_point_out]

    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert self.hook_point_in is not None and self.hook_point_out is not None, (
            "hook_point_in and hook_point_out must be set"
        )
        assert self.hook_point_in != self.hook_point_out, "hook_point_in and hook_point_out must be different"
        assert self.n_ov_heads % self.n_qk_heads == 0, "n_ov_heads must be divisible by n_qk_heads"
        if self.attn_type == "local":
            assert self.window_size is not None, "window_size must be provided for local attention"
            assert self.window_size > 0, "window_size must be positive for local attention"

        if self.attn_scale is None:
            self.attn_scale = self.d_qk_head**0.5


@register_sae_model("lorsa")
class LowRankSparseAttention(
    SparseDictionary,
    NormComputing,
    NormConstrainable,
    DatasetNormStandardizable,
    EncoderBiasInitializable,
):
    def __init__(self, cfg: LorsaConfig, device_mesh: Optional[DeviceMesh] = None):
        super().__init__(cfg, device_mesh=device_mesh)
        self.cfg = cfg

        if device_mesh is None:
            # Local parameters
            def _get_param_with_shape(shape: tuple[int, ...]) -> nn.Parameter:
                return nn.Parameter(
                    torch.empty(
                        shape,
                        dtype=self.cfg.dtype,
                        device=self.cfg.device,
                    )
                )

            self.W_Q = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_model, self.cfg.d_qk_head))
            self.W_K = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_model, self.cfg.d_qk_head))
            self.W_V = _get_param_with_shape((self.cfg.n_ov_heads, self.cfg.d_model))
            self.W_O = _get_param_with_shape((self.cfg.n_ov_heads, self.cfg.d_model))
            self.b_Q = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_qk_head))
            self.b_K = _get_param_with_shape((self.cfg.n_qk_heads, self.cfg.d_qk_head))
            self.b_V = _get_param_with_shape((self.cfg.n_ov_heads,))
            if self.cfg.use_decoder_bias:
                self.b_D = _get_param_with_shape((self.cfg.d_model,))
        else:
            # Distributed parameters with head sharding
            dim_maps = self.dim_maps()

            def _get_param_with_shape(shape: tuple[int, ...], placements: Sequence[Any]) -> nn.Parameter:
                return nn.Parameter(
                    torch.distributed.tensor.empty(
                        shape,
                        dtype=self.cfg.dtype,
                        device_mesh=device_mesh,
                        placements=placements,
                    )
                )

            self.W_Q = _get_param_with_shape(
                (self.cfg.n_qk_heads, self.cfg.d_model, self.cfg.d_qk_head),
                placements=dim_maps["W_Q"].placements(device_mesh),
            )
            self.W_K = _get_param_with_shape(
                (self.cfg.n_qk_heads, self.cfg.d_model, self.cfg.d_qk_head),
                placements=dim_maps["W_K"].placements(device_mesh),
            )
            self.W_V = _get_param_with_shape(
                (self.cfg.n_ov_heads, self.cfg.d_model), placements=dim_maps["W_V"].placements(device_mesh)
            )
            self.W_O = _get_param_with_shape(
                (self.cfg.n_ov_heads, self.cfg.d_model), placements=dim_maps["W_O"].placements(device_mesh)
            )
            self.b_Q = _get_param_with_shape(
                (self.cfg.n_qk_heads, self.cfg.d_qk_head), placements=dim_maps["b_Q"].placements(device_mesh)
            )
            self.b_K = _get_param_with_shape(
                (self.cfg.n_qk_heads, self.cfg.d_qk_head), placements=dim_maps["b_K"].placements(device_mesh)
            )
            self.b_V = _get_param_with_shape((self.cfg.n_ov_heads,), placements=dim_maps["b_V"].placements(device_mesh))
            if self.cfg.use_decoder_bias:
                self.b_D = _get_param_with_shape(
                    (self.cfg.d_model,), placements=dim_maps["b_D"].placements(device_mesh)
                )

        # Attention mask
        causal_mask = torch.tril(
            torch.ones(
                (self.cfg.n_ctx, self.cfg.n_ctx),
                device=self.cfg.device,
                dtype=self.cfg.dtype,
            ).bool(),
        )
        if self.cfg.attn_type == "global":
            mask = causal_mask
        elif self.cfg.attn_type == "local":
            if self.cfg.window_size is None:
                raise ValueError("window_size must be provided for local attention")
            mask = torch.triu(causal_mask, 1 - self.cfg.window_size)
        else:
            raise ValueError(f"Invalid attention type: {self.cfg.attn_type}")
        if self.device_mesh is not None:
            mask = DimMap({}).distribute(mask, self.device_mesh)
        self.register_buffer("mask", mask)

        if self.device_mesh is not None:
            IGNORE = DimMap({}).distribute(torch.tensor(-torch.inf, device=self.cfg.device), self.device_mesh)
        else:
            IGNORE = torch.tensor(-torch.inf, device=self.cfg.device)
        self.register_buffer("IGNORE", IGNORE)

        if self.cfg.use_post_qk_ln:
            # if self.cfg.normalization_type == "LN":
            #     # TODO: fix this
            #     pass
            if self.cfg.normalization_type == "RMS":
                self.qk_ln_type = RMSNormPerHead
            else:
                raise ValueError(f"Invalid normalization type for QK-norm: {self.cfg.normalization_type}")
        else:
            self.qk_ln_type = None

        if self.cfg.use_post_qk_ln:
            assert self.qk_ln_type is not None
            self.ln_q = self.qk_ln_type(self.cfg, n_heads=self.cfg.n_qk_heads, device_mesh=device_mesh)
            self.ln_k = self.qk_ln_type(self.cfg, n_heads=self.cfg.n_qk_heads, device_mesh=device_mesh)

        if self.cfg.positional_embedding_type == "rotary":
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position.
            if self.cfg.rotary_dim is None:  # keep mypy happy
                raise ValueError("Rotary dim must be provided for rotary positional embeddings")
            rope_base = (
                self.cfg.rotary_base_local
                if self.cfg.attn_type == "local" and self.cfg.rotary_base_local is not None
                else self.cfg.rotary_base
            )
            sin, cos = self._calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                base=rope_base,
                dtype=self.cfg.dtype,
                device=self.cfg.device,
            )
            if self.device_mesh is not None:
                sin = DimMap({}).distribute(sin, self.device_mesh)
                cos = DimMap({}).distribute(cos, self.device_mesh)
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)

        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()
        self.hook_attn_pattern = HookPoint()
        self.hook_attn_score = HookPoint()
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_q_post_rot = HookPoint()
        self.hook_k_post_rot = HookPoint()
        self.setup()

    @property
    def W_D(self) -> torch.Tensor:
        return self.W_O

    @property
    def W_E(self) -> torch.Tensor:
        return self.W_V.T

    @property
    def b_O(self):
        return self.b_D

    @property
    def attn_scale(self) -> float:
        assert self.cfg.attn_scale is not None, "attn_scale must be initialized during config post initialization"
        return self.cfg.attn_scale

    def init_parameters(self, **kwargs):
        """Initialize parameters."""
        super().init_parameters(**kwargs)

        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)

        W_V_bound = 1 / math.sqrt(self.cfg.d_sae)
        # torch.nn.init.uniform_(self.W_V, -W_V_bound, W_V_bound)
        torch.nn.init.normal_(self.W_V, mean=0, std=W_V_bound)

        W_O_bound = 1 / math.sqrt(self.cfg.d_model)
        # torch.nn.init.uniform_(self.W_O, -W_O_bound, W_O_bound)
        torch.nn.init.normal_(self.W_O, mean=0, std=W_O_bound)

        torch.nn.init.zeros_(self.b_Q)
        torch.nn.init.zeros_(self.b_K)
        torch.nn.init.zeros_(self.b_V)
        if self.cfg.use_decoder_bias:
            torch.nn.init.zeros_(self.b_D)

    @staticmethod
    def _expanded_mhsa_qk_norm_weights(
        mhsa: Attention | GroupedQueryAttention,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(mhsa, "ln_q") and hasattr(mhsa, "ln_k"):
            ln_q_w = cast(torch.Tensor, mhsa.ln_q.w)  # type: ignore[attr-defined]
            ln_k_w = cast(torch.Tensor, mhsa.ln_k.w)  # type: ignore[attr-defined]
            if mhsa.cfg.n_key_value_heads is not None and ln_k_w.size(0) != mhsa.W_Q.size(0):
                ln_k_w = torch.repeat_interleave(ln_k_w, mhsa.cfg.n_heads // mhsa.cfg.n_key_value_heads, dim=0)
            return ln_q_w, ln_k_w

        if hasattr(mhsa, "q_norm") and hasattr(mhsa, "k_norm"):
            q_norm_w = cast(torch.Tensor, mhsa.q_norm.w)  # type: ignore[attr-defined]
            k_norm_w = cast(torch.Tensor, mhsa.k_norm.w)  # type: ignore[attr-defined]
            return (
                q_norm_w.unsqueeze(0).repeat(mhsa.W_Q.size(0), 1),
                k_norm_w.unsqueeze(0).repeat(mhsa.W_Q.size(0), 1),
            )

        raise AttributeError("MHSA module does not expose supported QK-norm weights")

    @torch.no_grad()
    def init_lorsa_with_mhsa(self, mhsa: Attention | GroupedQueryAttention):
        """Initialize Lorsa with Original Multi Head Sparse Attention"""
        assert self.cfg.n_qk_heads % mhsa.W_Q.size(0) == 0
        assert self.cfg.d_qk_head == mhsa.W_Q.size(2)
        assert self.dataset_average_activation_norm is not None
        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        qk_exp_factor = self.cfg.n_qk_heads // mhsa.W_Q.size(0)
        if self.device_mesh is not None:
            model_parallel_rank = self.device_mesh.get_local_rank(mesh_dim="model")
            model_parallel_size = mesh_dim_size(self.device_mesh, "model")
            lorsa_qk_start_idx = model_parallel_rank * self.cfg.n_qk_heads // model_parallel_size
            lorsa_qk_end_idx = lorsa_qk_start_idx + self.cfg.n_qk_heads // model_parallel_size
            lorsa_qk_indices = torch.arange(lorsa_qk_start_idx, lorsa_qk_end_idx)
            W_Q_local = mhsa.W_Q[lorsa_qk_indices // qk_exp_factor] / input_norm_factor
            W_K_local = mhsa.W_K[lorsa_qk_indices // qk_exp_factor] / input_norm_factor
            W_Q = DTensor.from_local(
                W_Q_local,
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_Q"].placements(self.device_mesh),
            )
            W_K = DTensor.from_local(
                W_K_local,
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_K"].placements(self.device_mesh),
            )
            self.W_Q.copy_(W_Q)
            self.W_K.copy_(W_K)
            if self.cfg.use_post_qk_ln and self.cfg.normalization_type == "RMS":
                q_norm_w, k_norm_w = self._expanded_mhsa_qk_norm_weights(mhsa)
                ln_q_w_local = q_norm_w[lorsa_qk_indices // qk_exp_factor]
                ln_k_w_local = k_norm_w[lorsa_qk_indices // qk_exp_factor]
                ln_q_w = DTensor.from_local(
                    ln_q_w_local,
                    device_mesh=self.device_mesh,
                    placements=self.ln_q.dim_maps()["w"].placements(self.device_mesh),
                )
                ln_k_w = DTensor.from_local(
                    ln_k_w_local,
                    device_mesh=self.device_mesh,
                    placements=self.ln_k.dim_maps()["w"].placements(self.device_mesh),
                )
                self.ln_q.w.copy_(ln_q_w)
                self.ln_k.w.copy_(ln_k_w)
        else:
            self.W_Q = nn.Parameter(
                torch.repeat_interleave(mhsa.W_Q, qk_exp_factor, dim=0).to(self.cfg.dtype) / input_norm_factor
            )
            self.W_K = nn.Parameter(
                torch.repeat_interleave(mhsa.W_K, qk_exp_factor, dim=0).to(self.cfg.dtype) / input_norm_factor
            )
            if self.cfg.use_post_qk_ln and self.cfg.normalization_type == "RMS":
                q_norm_w, k_norm_w = self._expanded_mhsa_qk_norm_weights(mhsa)
                self.ln_q.w = nn.Parameter(torch.repeat_interleave(q_norm_w, qk_exp_factor, dim=0).to(self.cfg.dtype))
                self.ln_k.w = nn.Parameter(torch.repeat_interleave(k_norm_w, qk_exp_factor, dim=0).to(self.cfg.dtype))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def init_W_D_with_active_subspace_per_head(
        self, batch: dict[str, torch.Tensor], mhsa: Attention | GroupedQueryAttention
    ):
        """
        Initialize W_D with the active subspace for each head.
        """
        x = self.prepare_input(batch)[0]
        if isinstance(x, DTensor):
            x = x.to_local()

        captured_z = None

        def capture_hook(tensor, hook):
            nonlocal captured_z
            captured_z = tensor.clone().detach()
            return tensor

        mhsa.hook_z.add_hook(capture_hook)
        _ = mhsa.forward(
            query_input=x,
            key_input=x,
            value_input=x,
        )
        output_per_head = torch.einsum("b s n h, n h d -> b s n d", captured_z, mhsa.W_O)
        n_ov_per_orig_head = self.cfg.n_ov_heads // mhsa.cfg.n_heads
        if self.device_mesh is not None:
            assert isinstance(self.W_O, DTensor)
            assert isinstance(self.W_V, DTensor)
            model_parallel_rank = self.device_mesh.get_local_rank(mesh_dim="model")
            model_parallel_size = mesh_dim_size(self.device_mesh, "model")
            orig_start_idx = model_parallel_rank * mhsa.cfg.n_heads // model_parallel_size
            orig_end_idx = orig_start_idx + mhsa.cfg.n_heads // model_parallel_size
            W_O_local = torch.empty_like(self.W_O.to_local())
            W_V_local = torch.empty_like(self.W_V.to_local())
            for orig_head_index in range(orig_start_idx, orig_end_idx):
                output = output_per_head[:, :, orig_head_index, :]
                output_flattened = output.flatten(0, 1)
                demeaned_output = output_flattened - output_flattened.mean(dim=0)
                U, S, V = torch.svd(demeaned_output.T.to(torch.float32))
                proj_weight = U[:, : self.cfg.d_qk_head]
                start_idx = (orig_head_index - orig_start_idx) * n_ov_per_orig_head
                end_idx = min(start_idx + n_ov_per_orig_head, W_O_local.size(0))
                W_O_local[start_idx:end_idx] = (
                    self.W_O.to_local()[start_idx:end_idx, : self.cfg.d_qk_head] @ proj_weight.T
                )
                W_V_local[start_idx:end_idx] = (
                    W_O_local[start_idx:end_idx] @ (mhsa.W_V[orig_head_index] @ mhsa.W_O[orig_head_index]).T
                )
            W_V_local = W_V_local / W_V_local.norm(dim=1, keepdim=True)
            W_O_local = W_O_local / W_O_local.norm(dim=1, keepdim=True)
            torch.distributed.broadcast(tensor=W_O_local, group=self.device_mesh.get_group("data"), group_src=0)
            torch.distributed.broadcast(tensor=W_V_local, group=self.device_mesh.get_group("data"), group_src=0)
            W_O_global = DTensor.from_local(
                W_O_local, device_mesh=self.device_mesh, placements=self.dim_maps()["W_O"].placements(self.device_mesh)
            )
            W_V_global = DTensor.from_local(
                W_V_local, device_mesh=self.device_mesh, placements=self.dim_maps()["W_V"].placements(self.device_mesh)
            )
            self.W_O.copy_(W_O_global)
            self.W_V.copy_(W_V_global)
        else:
            for orig_head_index in range(mhsa.cfg.n_heads):
                output = output_per_head[:, :, orig_head_index, :]
                output_flattened = output.flatten(0, 1)
                demeaned_output = output_flattened - output_flattened.mean(dim=0)
                U, S, V = torch.svd(demeaned_output.T.to(torch.float32))
                proj_weight = U[:, : self.cfg.d_qk_head]
                self.W_O[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head] = (
                    self.W_O[
                        orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head,
                        : self.cfg.d_qk_head,
                    ]
                    @ proj_weight.T
                )
                self.W_V[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head] = (
                    self.W_O[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head]
                    @ (mhsa.W_V[orig_head_index] @ mhsa.W_O[orig_head_index]).T
                )
            self.W_V.copy_(self.W_V / self.W_V.norm(dim=1, keepdim=True))
            self.W_O.copy_(self.W_O / self.W_O.norm(dim=1, keepdim=True))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def init_W_V_with_active_subspace_per_head(
        self, batch: dict[str, torch.Tensor], mhsa: Attention | GroupedQueryAttention
    ):
        """
        Initialize W_D with the active subspace for each head.
        """
        x = self.prepare_input(batch)[0]
        if isinstance(x, DTensor):
            x = x.to_local()

        v_per_head = torch.einsum("bd,ndh->bnh", x.reshape(-1, self.cfg.d_model), mhsa.W_V)
        captured_v = torch.einsum("bnh,nhd->bnd", v_per_head, mhsa.W_V.permute(0, 2, 1))

        n_ov_per_orig_head = self.cfg.n_ov_heads // mhsa.cfg.n_heads
        if self.device_mesh is not None:
            assert isinstance(self.W_O, DTensor)
            assert isinstance(self.W_V, DTensor)
            model_parallel_rank = self.device_mesh.get_local_rank(mesh_dim="model")
            model_parallel_size = mesh_dim_size(self.device_mesh, "model")
            orig_start_idx = model_parallel_rank * mhsa.cfg.n_heads // model_parallel_size
            orig_end_idx = orig_start_idx + mhsa.cfg.n_heads // model_parallel_size
            W_O_local = torch.empty_like(self.W_O.to_local())
            W_V_local = torch.empty_like(self.W_V.to_local())
            for orig_head_index in range(orig_start_idx, orig_end_idx):
                v = captured_v[:, orig_head_index]
                demeaned_v = v - v.mean(dim=0)
                U, S, V = torch.svd(demeaned_v.T.to(torch.float32))
                proj_weight = U[:, : self.cfg.d_qk_head]
                start_idx = (orig_head_index - orig_start_idx) * n_ov_per_orig_head
                end_idx = min(start_idx + n_ov_per_orig_head, W_O_local.size(0))
                W_V_local[start_idx:end_idx] = (
                    self.W_V.to_local()[start_idx:end_idx, : self.cfg.d_qk_head] @ proj_weight.T
                )
                W_O_local[start_idx:end_idx] = (
                    W_V_local[start_idx:end_idx] @ mhsa.W_V[orig_head_index] @ mhsa.W_O[orig_head_index]
                )
            W_V_local = W_V_local / W_V_local.norm(dim=1, keepdim=True)
            W_O_local = W_O_local / W_O_local.norm(dim=1, keepdim=True)
            torch.distributed.broadcast(tensor=W_O_local, group=self.device_mesh.get_group("data"), group_src=0)
            torch.distributed.broadcast(tensor=W_V_local, group=self.device_mesh.get_group("data"), group_src=0)
            W_O_global = DTensor.from_local(
                W_O_local, device_mesh=self.device_mesh, placements=self.dim_maps()["W_O"].placements(self.device_mesh)
            )
            W_V_global = DTensor.from_local(
                W_V_local, device_mesh=self.device_mesh, placements=self.dim_maps()["W_V"].placements(self.device_mesh)
            )
            self.W_O.copy_(W_O_global)
            self.W_V.copy_(W_V_global)
        else:
            for orig_head_index in range(mhsa.cfg.n_heads):
                v = captured_v[:, orig_head_index]
                demeaned_v = v - v.mean(dim=0)
                U, S, V = torch.svd(demeaned_v.T.to(torch.float32))
                proj_weight = U[:, : self.cfg.d_qk_head]
                self.W_V[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head] = (
                    self.W_V[
                        orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head,
                        : self.cfg.d_qk_head,
                    ]
                    @ proj_weight.T
                )
                self.W_O[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head] = (
                    self.W_V[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head]
                    @ mhsa.W_V[orig_head_index]
                    @ mhsa.W_O[orig_head_index]
                )
            self.W_V.copy_(self.W_V / self.W_V.norm(dim=1, keepdim=True))
            self.W_O.copy_(self.W_O / self.W_O.norm(dim=1, keepdim=True))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def init_encoder_bias_with_mean_hidden_pre(self, batch: dict[str, torch.Tensor]):
        x = self.prepare_input(batch)[0]
        _, hidden_pre = self.encode(x, return_hidden_pre=True)

        self.b_V.sub_(hidden_pre.mean(dim=[0, 1]))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def reverse_for_off_center(self, batch: dict[str, torch.Tensor]):
        x = self.prepare_input(batch)[0]
        _, hidden_pre = self.encode(x, return_hidden_pre=True)
        hidden_pre_flatten = hidden_pre.flatten(0, 1)
        positive_freq = (hidden_pre_flatten > 0).sum(dim=0) / hidden_pre_flatten.size(0)
        reverse_mask = positive_freq < 2e-1
        reverse_factor = masked_fill(torch.ones_like(self.W_V), reverse_mask, -1)
        self.W_V.mul_(reverse_factor)
        self.W_O.mul_(reverse_factor)

    def _calculate_sin_cos_rotary(
        self,
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> Tuple[
        Float[torch.Tensor, "n_ctx rotary_dim"],
        Float[torch.Tensor, "n_ctx rotary_dim"],
    ]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

        Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
        To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
        """
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision, device=device)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision, device=device)

        # Llama-3.1 uses NTK-by-Parts Rotary Embedding introduced in Section 3.2 in https://arxiv.org/pdf/2309.00071
        # Implementation copied from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py#L310
        if self.cfg.use_NTK_by_parts_rope:
            inv_freq = 1.0 / (
                base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64, device=device).float() / rotary_dim)
            )
            factor = self.cfg.NTK_by_parts_factor
            low_freq_factor = self.cfg.NTK_by_parts_low_freq_factor
            high_freq_factor = self.cfg.NTK_by_parts_high_freq_factor
            old_context_len = self.cfg.old_context_len

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            freq = 1 / inv_freq_llama
        else:
            freq = base ** (dim / (rotary_dim / 2))
        if self.cfg.rotary_adjacent_pairs:
            freq = einops.repeat(freq, "d -> (d 2)")
        else:
            freq = einops.repeat(freq, "d -> (2 d)")
        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    @override
    def encoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Norm of encoder (Q/K weights)."""
        if not isinstance(self.W_V, DTensor):
            return torch.norm(self.W_V, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_V.to_local(), p=2, dim=1, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_V"].placements(self.device_mesh),
            )

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Norm of decoder (O weights)."""
        if not isinstance(self.W_O, DTensor):
            return torch.norm(self.W_O, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_O.to_local(), p=2, dim=1, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_O"].placements(self.device_mesh),
            )

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        """Norm of decoder bias."""
        if not self.cfg.use_decoder_bias:
            raise ValueError("Decoder bias not used")
        return torch.norm(self.b_D, p=2, dim=0, keepdim=True)

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        """Transform to unit decoder norm."""
        norm = self.decoder_norm(keepdim=True)
        self.W_O /= norm
        self.W_V *= norm
        self.b_V *= norm.squeeze()

    @override
    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self):
        """Standardize parameters for dataset norm."""
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None

        hook_point_in = self.cfg.hook_point_in
        hook_point_out = self.cfg.hook_point_out

        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point_in]
        output_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[hook_point_out]

        self.W_Q.data *= input_norm_factor
        self.W_K.data *= input_norm_factor

        self.W_V.data *= input_norm_factor

        self.W_O.data = self.W_O.data / output_norm_factor
        self.b_D.data = self.b_D.data / output_norm_factor

        self.cfg.norm_activation = "inference"

    def compute_hidden_pre(
        self, x: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> Float[torch.Tensor, "batch seq_len d_sae"]:
        """Compute the hidden pre-activations."""
        q, k, v = self._compute_qkv(x)
        return self._compute_hidden_pre_with_sdpa_or_manual(q, k, v)

    @overload
    def encode(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        return_hidden_pre: Literal[False] = False,
        **kwargs,
    ) -> Float[torch.Tensor, "batch seq_len d_sae"]: ...

    @overload
    def encode(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        return_hidden_pre: Literal[True],
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "batch seq_len d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
    ]: ...

    @override
    def encode(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        return_hidden_pre: bool = False,
        return_attention_pattern: bool = False,
        return_attention_score: bool = False,
        hook_attn_scores: bool = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch seq_len d_sae"],
        Tuple[
            Float[torch.Tensor, "batch seq_len d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        Tuple[
            Float[torch.Tensor, "batch seq_len d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
            Float[torch.Tensor, "batch n_qk_heads q_pos k_pos"],
        ],
    ]:
        """Encode to sparse head activations."""
        # Compute Q, K, V
        q, k, v = self._compute_qkv(x)

        pattern: Optional[torch.Tensor] = None
        scores: Optional[torch.Tensor] = None

        if not (return_attention_pattern or return_attention_score or hook_attn_scores):
            hidden_pre = self._compute_hidden_pre_with_sdpa_or_manual(q, k, v)
        else:
            # Attention pattern
            # n_qk_heads batch q_pos k_pos
            q = q.permute(2, 0, 1, 3)  # (n_qk_heads, batch, seq_len, d_qk_head)
            k = k.permute(2, 0, 3, 1)  # (n_qk_heads, batch, d_qk_head, seq_len)
            scores = torch.einsum("nbqd,nbdk->nbqk", q, k) / self.attn_scale
            scores = cast(
                torch.Tensor, self.hook_attn_score(self._apply_causal_mask(scores).permute(1, 0, 2, 3))
            )  # (n_qk_heads, batch, q_pos, k_pos)
            # Hook called on (batch, n_qk_heads, q_pos, k_pos); permute back for internal use
            pattern = cast(
                torch.Tensor,
                self.hook_attn_pattern(F.softmax(scores, dim=-1)).permute(1, 0, 2, 3),
            )
            # Head outputs
            hidden_pre = self._compute_head_outputs(pattern, v)

        hidden_pre = self.hook_hidden_pre(hidden_pre)

        # Scale feature activations by decoder norm if configured
        if self.cfg.sparsity_include_decoder_norm:
            hidden_pre = hidden_pre * self.decoder_norm()

        feature_acts = self.activation_function(hidden_pre)

        if self.cfg.sparsity_include_decoder_norm:
            feature_acts = feature_acts / self.decoder_norm()
            hidden_pre = hidden_pre / self.decoder_norm()

        return_values: list[torch.Tensor] = [self.hook_feature_acts(feature_acts)]
        if return_hidden_pre:
            return_values.append(hidden_pre)
        if return_attention_pattern and pattern is not None:
            return_values.append(pattern.permute(1, 0, 2, 3))
        if return_attention_score and scores is not None:
            return_values.append(scores.permute(1, 0, 2, 3))
        return tuple(return_values) if len(return_values) > 1 else return_values[0]  # type: ignore[return-value]

    @override
    def decode(self, feature_acts, **kwargs):
        """Decode head activations to output."""
        if feature_acts.layout == torch.sparse_coo:
            out = (
                torch.sparse.mm(
                    feature_acts.to(torch.float32),
                    self.W_O.to(torch.float32),
                ).to(self.cfg.dtype)
                + self.b_D
            )
            return self.hook_reconstructed(out)
        out = torch.einsum("bps,sd->bpd", feature_acts, self.W_O)
        if self.cfg.use_decoder_bias:
            out = out + self.b_D
        out = self.hook_reconstructed(out)
        if isinstance(out, DTensor):
            out = DimMap({"data": 0}).redistribute(out)
        return out

    def _compute_qkv(
        self, x: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> tuple[
        Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        Float[torch.Tensor, "batch seq_len n_ov_heads"],
    ]:
        """Compute queries, keys, values."""
        x = x.to(self.W_Q.dtype)
        q = self.hook_q(
            torch.einsum(
                "bsd,Qdq->bsQq",
                x,
                self.W_Q,
            )
            + self.b_Q
        )
        k = self.hook_k(
            torch.einsum(
                "bsd,Kdk->bsKk",
                x,
                self.W_K,
            )
            + self.b_K
        )
        v = (
            torch.einsum(
                "bsd,Vd->bsV",
                x,
                self.W_V,
            )
            + self.b_V
        )
        if self.cfg.use_post_qk_ln:
            q = self.ln_q(q)
            k = self.ln_k(k)

        # Apply positional embedding
        if self.cfg.positional_embedding_type == "rotary":
            q = self._apply_rotary(q)
            k = self._apply_rotary(k)

        q = self.hook_q_post_rot(q)
        k = self.hook_k_post_rot(k)

        return q, k, v

    def encode_z_pattern_for_head(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        head_idx: Int[torch.Tensor, " n_active_features"],
    ) -> Float[torch.Tensor, "n_active_features k_pos"]:
        assert x.size(0) == 1, f"x must be of shape (1, seq_len, d_model), but got {x.shape}"
        qk_idx = head_idx // self.cfg.ov_group_size
        q, k, v = self._compute_qkv(x)

        # (n_active_features, q_pos, k_pos)
        all_patterns = self._compute_attention_pattern(q, k)
        if isinstance(all_patterns, DTensor):
            all_patterns = all_patterns.to_local()

        pattern = all_patterns[qk_idx, 0]
        if isinstance(v, DTensor):
            v = v.to_local()
        return pattern.mul_(v[0, :, head_idx, None].permute(1, 2, 0))

    def encode_z_patterns(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
    ) -> Float[torch.Tensor, "n_active_features q_pos k_pos"]:
        assert x.size(0) == 1, f"x must be of shape (1, seq_len, d_model), but got {x.shape}"

        head_idx = torch.arange(self.cfg.d_sae)
        qk_idx = head_idx // self.cfg.ov_group_size
        q, k, v = self._compute_qkv(x)

        # (n_active_features, q_pos, k_pos)
        pattern = self._compute_attention_pattern(q, k)[qk_idx, 0]
        return pattern.mul_(v[0, :, head_idx, None].permute(1, 2, 0))

    def _apply_rotary(
        self,
        x: Float[torch.Tensor, "batch pos head_index d_head"],
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        # x = x.repeat(1, 1, 1, self.cfg.rotary_scale)
        x = x.repeat_interleave(self.cfg.rotary_scale, dim=-1)

        x_pos = x.size(1)
        # Avoid triggering https://github.com/pytorch/pytorch/issues/170427 in tensor parallelism (tp) settings
        x_rot = x[:, :, :, : self.cfg.rotary_dim] if self.cfg.rotary_dim < x.size(-1) else x
        x_pass = x[:, :, :, self.cfg.rotary_dim :]
        x_flip = self._rotate_every_two(x_rot)

        rotary_cos = self.rotary_cos[  # type: ignore
            None, :x_pos, None, :
        ]
        rotary_sin = self.rotary_sin[  # type: ignore
            None, :x_pos, None, :
        ]
        x_rotated = x_rot * rotary_cos + x_flip * rotary_sin

        return torch.cat([x_rotated, x_pass], dim=-1)

    def _compute_hidden_pre_with_sdpa_or_manual(
        self,
        q: Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        k: Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        v: Float[torch.Tensor, "batch seq_len n_ov_heads"],
    ) -> Float[torch.Tensor, "batch seq_len d_sae"]:
        if isinstance(q, DTensor) or isinstance(k, DTensor) or isinstance(v, DTensor):
            q_local = q.to_local() if isinstance(q, DTensor) else q
            k_local = k.to_local() if isinstance(k, DTensor) else k
            v_local = v.to_local() if isinstance(v, DTensor) else v

            if self.cfg.attn_type == "local" and self.cfg.d_qk_head != self.cfg.ov_group_size:
                hidden_pre_local = self._compute_local_hidden_pre_on_shard(q_local, k_local, v_local)
            else:
                query_local = q_local.permute(0, 2, 1, 3)
                key_local = k_local.permute(0, 2, 1, 3)
                value_local = v_local.reshape(*k_local.shape[:3], -1).permute(0, 2, 1, 3)

                attn_mask_local = None
                is_causal = self.cfg.attn_type == "global"
                if self.cfg.attn_type == "local":
                    mask = self.mask.to_local() if isinstance(self.mask, DTensor) else self.mask
                    attn_mask_local = mask[None, None, -query_local.size(-2) :, -key_local.size(-2) :].to(
                        query_local.device
                    )

                with sdpa_kernel(
                    backends=[
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.CUDNN_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH,
                    ]
                ):
                    z_local = F.scaled_dot_product_attention(
                        query_local,
                        key_local,
                        value_local,
                        scale=1 / self.attn_scale,
                        enable_gqa=True,
                        attn_mask=attn_mask_local,
                        is_causal=is_causal,
                    )
                hidden_pre_local = z_local.permute(0, 2, 1, 3).reshape(*v_local.shape)

            if self.device_mesh is not None:
                return DimMap({"data": 0, "model": 2}).from_local(hidden_pre_local, self.device_mesh)
            return hidden_pre_local

        query = q.permute(0, 2, 1, 3)  # (batch, qk_heads, q_pos, qk_dim)
        key = k.permute(0, 2, 1, 3)  # (batch, qk_heads, k_pos, qk_dim)

        value = v.reshape(*k.shape[:3], -1).permute(0, 2, 1, 3)
        attn_mask = None
        is_causal = self.cfg.attn_type == "global"
        if self.cfg.attn_type == "local":
            attn_mask = self.mask[None, None, -query.size(-2) :, -key.size(-2) :]  # type: ignore[index]
            if not isinstance(attn_mask, DTensor):
                attn_mask = attn_mask.to(query.device)

        with sdpa_kernel(
            backends=[
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            z = F.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=1 / self.attn_scale,
                enable_gqa=True,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        return z.permute(0, 2, 1, 3).reshape(*v.shape)

    def _compute_local_hidden_pre_on_shard(
        self,
        q: Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        k: Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        v: Float[torch.Tensor, "batch seq_len n_ov_heads"],
    ) -> Float[torch.Tensor, "batch seq_len d_sae"]:
        """Compute chunked local attention on one tensor-parallel shard."""
        assert self.cfg.window_size is not None, "window_size must be provided for local attention"
        batch_size, seq_len, n_qk_heads_local, _ = q.shape
        ov_group_size = self.cfg.ov_group_size
        if v.size(-1) != n_qk_heads_local * ov_group_size:
            raise ValueError(
                "Local tensor-parallel shard has incompatible V width for sliding-window attention. "
                f"Expected {n_qk_heads_local * ov_group_size}, got {v.size(-1)}."
            )

        value = v.reshape(batch_size, seq_len, n_qk_heads_local, ov_group_size)
        hidden_pre = torch.empty_like(v)
        chunk_size = min(self.cfg.window_size, 128)
        scale = 1 / self.attn_scale
        positions = torch.arange(seq_len, device=q.device)

        for q_start in range(0, seq_len, chunk_size):
            q_end = min(q_start + chunk_size, seq_len)
            k_start = max(0, q_start - self.cfg.window_size + 1)
            k_end = q_end

            q_chunk = q[:, q_start:q_end].permute(0, 2, 1, 3)  # [bhqd]
            k_chunk = k[:, k_start:k_end].permute(0, 2, 1, 3)  # [bhkd]
            v_chunk = value[:, k_start:k_end].permute(0, 2, 1, 3)  # [bhkr]

            scores = torch.einsum("bhqd,bhkd->bhqk", q_chunk, k_chunk).to(torch.float32) * scale

            q_positions = positions[q_start:q_end]
            k_positions = positions[k_start:k_end]
            valid = (k_positions[None, :] <= q_positions[:, None]) & (
                k_positions[None, :] >= (q_positions[:, None] - self.cfg.window_size + 1)
            )
            scores = scores.masked_fill(~valid[None, None, :, :], float("-inf"))

            probs = F.softmax(scores, dim=-1).to(v_chunk.dtype)
            z_chunk = torch.einsum("bhqk,bhkr->bhqr", probs, v_chunk)
            hidden_pre[:, q_start:q_end] = z_chunk.permute(0, 2, 1, 3).reshape(batch_size, q_end - q_start, -1)

        return hidden_pre

    def _rotate_every_two(self, x: Float[torch.Tensor, "... rotary_dim"]) -> Float[torch.Tensor, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x.clone()
        if self.cfg.rotary_adjacent_pairs:
            rot_x[..., ::2] = -x[..., 1::2]
            rot_x[..., 1::2] = x[..., ::2]
        else:
            n = x.size(-1) // 2
            rot_x[..., :n] = -x[..., n:]
            rot_x[..., n:] = x[..., :n]

        return rot_x

    def _compute_attention_pattern(
        self,
        q: Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        k: Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
    ) -> Float[torch.Tensor, "n_qk_heads batch q_pos k_pos"]:
        """Compute attention patterns."""
        q = q.permute(2, 0, 1, 3)  # (n_qk_heads, batch, seq_len, d_qk_head)
        k = k.permute(2, 0, 3, 1)  # (n_qk_heads, batch, d_qk_head, seq_len)
        scores = torch.einsum("sbqd,sbdk->sbqk", q, k) / self.attn_scale
        scores = self._apply_causal_mask(scores)
        return F.softmax(scores, dim=-1)

    def _compute_head_outputs(
        self,
        pattern: Float[torch.Tensor, "n_qk_heads batch q_pos k_pos"],
        v: Float[torch.Tensor, "batch seq_len n_ov_heads"],
    ) -> Float[torch.Tensor, "batch n_qk_heads d_qk_head seq_len"]:
        """Compute head outputs from pattern and values."""
        v = einops.rearrange(v, "b seq h -> b h seq")
        if self.cfg.n_qk_heads != self.cfg.n_ov_heads:
            # (batch, n_qk_heads, d_qk_head, seq_len)
            v = v.view(v.shape[0], self.cfg.n_qk_heads, self.cfg.ov_group_size, v.shape[2])
            v = v.permute(1, 0, 2, 3)  # (n_qk_heads, batch, d_qk_head, seq_len)
            z = torch.einsum("hbqk,hbrk->hbrq", pattern, v)
            z = z.permute(1, 0, 2, 3)  # (batch, n_qk_heads, d_qk_head, seq_len)
            z = z.flatten(start_dim=1, end_dim=2)
        else:
            z = torch.einsum("bhqk,bhk->bhq", pattern, v)
        return z.permute(0, 2, 1)

    def _apply_causal_mask(
        self, scores: Float[torch.Tensor, "batch n_qk_heads seq_len seq_len"]
    ) -> Float[torch.Tensor, "batch n_qk_heads seq_len seq_len"]:
        """Apply causal mask to attention scores."""
        seq_len = scores.size(-2)
        mask = self.mask[None, None, -seq_len:, -seq_len:]  # type: ignore
        ignore_value = self.IGNORE.to(scores.device)
        return torch.where(mask, scores, ignore_value)  # type: ignore

    @override
    @torch.no_grad()
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        """Set decoder weights to a fixed norm."""
        if force_exact:
            self.W_O.mul_(value / self.decoder_norm(keepdim=True).mean())
        else:
            self.W_O.mul_(value / torch.clamp(self.decoder_norm(keepdim=True).mean(), min=value))

    @override
    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        """Set encoder weights to fixed norm."""
        raise NotImplementedError("set_encoder_to_fixed_norm does not make sense for lorsa")

    @override
    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        # Extract and set dataset_average_activation_norm if present
        norm_keys = [k for k in state_dict.keys() if k.startswith("dataset_average_activation_norm.")]

        if norm_keys:
            dataset_norm = {key.split(".", 1)[1]: state_dict[key].item() for key in norm_keys}
            self.set_dataset_average_activation_norm(dataset_norm)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dataset_average_activation_norm.")}

        ckpt_n_qk_heads = state_dict["W_Q"].size(0)
        if self.cfg.n_qk_heads != ckpt_n_qk_heads:
            assert self.cfg.n_qk_heads % ckpt_n_qk_heads == 0
            qk_exp_factor = self.cfg.n_qk_heads // ckpt_n_qk_heads
            for k in ["W_Q", "W_K", "b_Q", "b_K"]:
                state_dict[k] = torch.repeat_interleave(state_dict[k], qk_exp_factor, dim=0)
            if self.cfg.use_post_qk_ln:
                for k in ["ln_q.w", "ln_k.w"]:
                    state_dict[k] = torch.repeat_interleave(state_dict[k], qk_exp_factor, dim=0)

        self.load_state_dict(state_dict, strict=strict)

    @override
    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        base_maps = super().dim_maps()
        return {
            **base_maps,
            "W_Q": DimMap({"model": 0}),
            "W_K": DimMap({"model": 0}),
            "W_V": DimMap({"model": 0}),
            "W_O": DimMap({"model": 0}),
            "b_Q": DimMap({"model": 0}),
            "b_K": DimMap({"model": 0}),
            "b_V": DimMap({"model": 0}),
            "b_D": DimMap({}),
        }

    @override
    def prepare_input(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Prepare input tensor."""
        x = batch[self.cfg.hook_point_in]
        return x, {}, {}

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs):
        """Prepare label tensor."""
        label = batch[self.cfg.hook_point_out]
        return label


class RMSNormPerHead(nn.Module):
    def __init__(self, cfg: LorsaConfig, n_heads: Optional[int] = None, device_mesh: DeviceMesh | None = None):
        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        self.cfg = cfg
        self.eps = self.cfg.eps

        self.n_heads = n_heads if n_heads is not None else self.cfg.n_qk_heads

        if device_mesh is not None:
            self.w = nn.Parameter(
                torch.distributed.tensor.ones(
                    (self.n_heads, self.cfg.d_qk_head),
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["w"].placements(device_mesh),
                )
            )
        else:
            self.w = nn.Parameter(torch.ones((self.n_heads, self.cfg.d_qk_head), dtype=cfg.dtype, device=cfg.device))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x: Float[torch.Tensor, "batch pos length"]) -> Float[torch.Tensor, "batch pos length"]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)
        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale((x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt())
        x = x / scale * self.w
        x = self.hook_normalized(x.to(self.cfg.dtype))  # [batch, pos, length]
        return x

    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        return {
            "w": DimMap({"model": 0}),
        }
