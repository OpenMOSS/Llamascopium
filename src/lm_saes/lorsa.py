"""
Low Rank Sparse Attention Implementation
"""

import math
from typing import Any, Literal, Optional, Sequence, Tuple, Union, overload

import einops
import torch
import torch.distributed.tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch._tensor import Tensor
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Partial
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformer_lens.components import Attention, GroupedQueryAttention
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from .abstract_sae import AbstractSparseAutoEncoder
from .config import LorsaConfig
from .utils.distributed import DimMap, mesh_dim_size
from .utils.distributed.utils import replace_placements
from .utils.logging import get_distributed_logger
from .utils.timer import timer

logger = get_distributed_logger("lorsa")

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
    
    def forward(self, input: Tensor) -> Tensor:
        output = input.matmul(self.weight.transpose(-1, -2))
        if self.bias is not None:
            view_shape = [1] * (output.dim() - 1) + [self.bias.shape[-1]]
            output = output + self.bias.view(*view_shape)
        return output


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        factory_kwargs = {"device": device, "dtype": dtype}
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
    
    def forward(self, input: Tensor) -> Tensor:
        dims = tuple(range(input.dim() - len(self.normalized_shape), input.dim()))
        mean = input.mean(dim=dims, keepdim=True)
        var = input.var(dim=dims, unbiased=False, keepdim=True)
        output = (input - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine and self.weight is not None and self.bias is not None:
            view_shape = [1] * (input.dim() - len(self.normalized_shape)) + list(self.normalized_shape)
            output = output * self.weight.view(*view_shape) + self.bias.view(*view_shape)
        return output



class SmolGenAttention(nn.Module):
    """SmolGen module adapted for LORSA, with optional distributed parameter layouts."""

    def __init__(
        self,
        d_model: int,
        n_qk_heads: int,
        d_qk_head: int,
        n_ctx: int,
        device_mesh: Optional[DeviceMesh] = None,
        dim_maps: Optional[dict[str, DimMap]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            d_model: Input embedding dimension.
            n_qk_heads: Number of query/key heads that SmolGen produces weights for.
            d_qk_head: Per-head query/key dimension.
            n_ctx: Context length for the attention block.
            device_mesh: Optional tensor parallel mesh; when provided parameters are sharded.
            dim_maps: Placement metadata keyed by parameter name, required if ``device_mesh`` is set.
            dtype: Override for parameter dtype when constructing distributed tensors.
        """
        super().__init__()
        self.d_model = d_model
        self.n_qk_heads = n_qk_heads
        self.d_qk_head = d_qk_head
        self.n_ctx = n_ctx
        self.device_mesh = device_mesh
        self._dim_maps = dim_maps

        compressed_size = 32 * n_ctx

        if device_mesh is None:
            # Use provided device and dtype, or defaults
            if dtype is None:
                dtype = torch.get_default_dtype()
            self.compress = Linear(d_model, 32, bias=False, device=device, dtype=dtype)
            self.dense1 = Linear(compressed_size, 256, device=device, dtype=dtype)
            self.ln1 = LayerNorm(256, device=device, dtype=dtype)
            self.dense2 = Linear(256, n_qk_heads * 256, device=device, dtype=dtype)
            self.ln2 = LayerNorm(n_qk_heads * 256, device=device, dtype=dtype)
            self.smol_weight_gen = Linear(256, n_ctx * n_ctx, bias=False, device=device, dtype=dtype)
        else:
            assert dim_maps is not None, "dim_maps must be provided when device_mesh is set"
            if dtype is None:
                dtype = torch.get_default_dtype()

            self.compress = Linear(d_model, 32, bias=False)
            if "smolgen.compress.weight" in dim_maps:
                self.compress.weight = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (32, d_model),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.compress.weight"].placements(device_mesh),
                    )
                )

            self.dense1 = Linear(compressed_size, 256)
            if "smolgen.dense1.weight" in dim_maps:
                self.dense1.weight = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (256, compressed_size),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.dense1.weight"].placements(device_mesh),
                    )
                )
                self.dense1.bias = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (256,),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.dense1.bias"].placements(device_mesh),
                    )
                )

            self.ln1 = LayerNorm(256)
            if "smolgen.ln1.weight" in dim_maps:
                self.ln1.weight = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (256,),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.ln1.weight"].placements(device_mesh),
                    )
                )
                self.ln1.bias = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (256,),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.ln1.bias"].placements(device_mesh),
                    )
                )

            self.dense2 = Linear(256, n_qk_heads * 256)
            if "smolgen.dense2.weight" in dim_maps:
                self.dense2.weight = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (n_qk_heads * 256, 256),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.dense2.weight"].placements(device_mesh),
                    )
                )
                self.dense2.bias = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (n_qk_heads * 256,),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.dense2.bias"].placements(device_mesh),
                    )
                )

            self.ln2 = LayerNorm(n_qk_heads * 256)
            if "smolgen.ln2.weight" in dim_maps:
                self.ln2.weight = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (n_qk_heads * 256,),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.ln2.weight"].placements(device_mesh),
                    )
                )
                self.ln2.bias = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (n_qk_heads * 256,),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.ln2.bias"].placements(device_mesh),
                    )
                )

            self.smol_weight_gen = Linear(256, n_ctx * n_ctx, bias=False)
            if "smolgen.smol_weight_gen.weight" in dim_maps:
                self.smol_weight_gen.weight = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (n_ctx * n_ctx, 256),
                        dtype=dtype,
                        device_mesh=device_mesh,
                        placements=dim_maps["smolgen.smol_weight_gen.weight"].placements(device_mesh),
                    )
                )

    def forward(self, x: torch.Tensor | DTensor) -> torch.Tensor | DTensor:
        """
        Args:
            x: Input tensor with shape ``[batch, n_ctx, d_model]`` (dense or DTensor).
        Returns:
            Attention weights tensor with shape ``[batch, n_qk_heads, n_ctx, n_ctx]``.
            Under model parallelism this returns the local head shard for the current rank.
        """
        if self.device_mesh is not None:
            x_local = x.to_local() if isinstance(x, DTensor) else x
            batch_size = x_local.shape[0]

            def _to_local_param(param: torch.Tensor | DTensor | None) -> torch.Tensor | None:
                if param is None:
                    return None
                if not isinstance(param, DTensor):
                    return param

                grad_placements = list(param.placements)
                mesh_dim_names = param.device_mesh.mesh_dim_names or ()
                for mesh_dim in ("data", "model"):
                    if mesh_dim not in mesh_dim_names:
                        continue
                    mesh_dim_index = mesh_dim_names.index(mesh_dim)
                    if grad_placements[mesh_dim_index].is_replicate():
                        grad_placements[mesh_dim_index] = Partial("sum")
                return param.to_local(grad_placements=tuple(grad_placements))

            compress_weight = _to_local_param(self.compress.weight)
            assert compress_weight is not None
            compressed = F.linear(x_local, compress_weight, None)
            x_flat = compressed.reshape(batch_size, -1)

            dense1_weight = _to_local_param(self.dense1.weight)
            dense1_bias = _to_local_param(self.dense1.bias)
            assert dense1_weight is not None
            x_local_ff = F.linear(x_flat, dense1_weight, dense1_bias)
            x_local_ff = F.silu(x_local_ff)

            ln1_weight = _to_local_param(self.ln1.weight)
            ln1_bias = _to_local_param(self.ln1.bias)
            x_local_ff = F.layer_norm(
                x_local_ff,
                self.ln1.normalized_shape,
                ln1_weight,
                ln1_bias,
                self.ln1.eps,
            )

            dense2_weight = _to_local_param(self.dense2.weight)
            dense2_bias = _to_local_param(self.dense2.bias)
            assert dense2_weight is not None
            x_local_ff = F.linear(x_local_ff, dense2_weight, dense2_bias)
            x_local_ff = F.silu(x_local_ff)

            ln2_weight = _to_local_param(self.ln2.weight)
            ln2_bias = _to_local_param(self.ln2.bias)
            x_local_ff = F.layer_norm(
                x_local_ff,
                self.ln2.normalized_shape,
                ln2_weight,
                ln2_bias,
                self.ln2.eps,
            )
            x_local_ff = x_local_ff.reshape(batch_size, self.n_qk_heads, 256)

            head_slice = DimMap({"model": 1}).local_slices(
                (batch_size, self.n_qk_heads, 256),
                self.device_mesh,
            )[1]
            x_local_ff = x_local_ff[:, head_slice, :]
            local_n_qk_heads = x_local_ff.shape[1]

            smol_weight = _to_local_param(self.smol_weight_gen.weight)
            assert smol_weight is not None
            weights_flat = F.linear(
                x_local_ff.reshape(batch_size * local_n_qk_heads, 256),
                smol_weight,
                None,
            )
            return weights_flat.reshape(batch_size, local_n_qk_heads, self.n_ctx, self.n_ctx)

        batch_size, _, _ = x.shape
        compressed = self.compress(x)
        x_flat = compressed.reshape(batch_size, -1)
        x = self.ln1(F.silu(self.dense1(x_flat)))
        x = self.ln2(F.silu(self.dense2(x)))
        x = x.reshape(batch_size, self.n_qk_heads, 256)
        weights_flat = self.smol_weight_gen(x.reshape(batch_size * self.n_qk_heads, 256))
        weights = weights_flat.reshape(batch_size, self.n_qk_heads, self.n_ctx, self.n_ctx)
        return weights

class LowRankSparseAttention(AbstractSparseAutoEncoder):
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
        mask = torch.tril(
            torch.ones(
                (self.cfg.n_ctx, self.cfg.n_ctx),
                device=self.cfg.device,
                dtype=self.cfg.dtype,
            ).bool(),
        )
        if self.device_mesh is not None:
            mask = DimMap({}).distribute(mask, self.device_mesh)
        self.register_buffer("mask", mask)

        if self.device_mesh is not None:
            IGNORE = DimMap({}).distribute(torch.tensor(-torch.inf, device=self.cfg.device), self.device_mesh)
        else:
            IGNORE = torch.tensor(-torch.inf, device=self.cfg.device)
        self.register_buffer("IGNORE", IGNORE)

        if self.cfg.use_post_qk_ln:
            if self.cfg.normalization_type == "LN":
                # self.qk_ln_type = LayerNormPerHead
                # TODO: fix this
                pass
            elif self.cfg.normalization_type == "RMS":
                self.qk_ln_type = RMSNormPerHead
            else:
                raise ValueError(f"Invalid normalization type for QK-norm: {self.cfg.normalization_type}")
        else:
            self.qk_ln_type = None

        if self.cfg.use_post_qk_ln:
            assert self.qk_ln_type is not None
            self.ln_q = self.qk_ln_type(self.cfg, n_heads=self.cfg.n_qk_heads, device_mesh=device_mesh)
            self.ln_k = self.qk_ln_type(self.cfg, n_heads=self.cfg.n_qk_heads, device_mesh=device_mesh)

        self.hook_k = HookPoint()  # [batch, pos, q_head_index, d_qk_head]
        self.hook_q = HookPoint()  # [batch, pos, q_head_index, d_qk_head]

        if self.cfg.positional_embedding_type == "rotary":
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position.
            if self.cfg.rotary_dim is None:  # keep mypy happy
                raise ValueError("Rotary dim must be provided for rotary positional embeddings")
            sin, cos = self._calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                base=self.cfg.rotary_base,
                dtype=self.cfg.dtype,
                device=self.cfg.device,
            )
            if self.device_mesh is not None:
                sin = DimMap({}).distribute(sin, self.device_mesh)
                cos = DimMap({}).distribute(cos, self.device_mesh)
            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)

        # SmolGen module for generating attention biases
        if self.cfg.use_smolgen:
            if device_mesh is None:
                self.smolgen = SmolGenAttention(
                    d_model=self.cfg.d_model,
                    n_qk_heads=self.cfg.n_qk_heads,
                    d_qk_head=self.cfg.d_qk_head,
                    n_ctx=self.cfg.n_ctx,
                    device=self.cfg.device,
                    dtype=self.cfg.dtype,
                )
            else:
                dim_maps = self.dim_maps()
                self.smolgen = SmolGenAttention(
                    d_model=self.cfg.d_model,
                    n_qk_heads=self.cfg.n_qk_heads,
                    d_qk_head=self.cfg.d_qk_head,
                    n_ctx=self.cfg.n_ctx,
                    device_mesh=device_mesh,
                    dim_maps=dim_maps,
                    dtype=self.cfg.dtype,
                )
            # Scale factor for SmolGen scores
            smolgen_scale = torch.tensor(1.0, device=self.cfg.device, dtype=self.cfg.dtype)
            if self.device_mesh is not None:
                smolgen_scale = DimMap({}).distribute(smolgen_scale, self.device_mesh)
            self.register_buffer("smolgen_score_scale", smolgen_scale)

        # Learnable attention scale
        if self.cfg.use_learnable_attn_scale:
            assert self.cfg.attn_scale is not None, "attn_scale must be initialized before using learnable_attn_scale"
            if device_mesh is None:
                self._attn_scale_param = nn.Parameter(
                    torch.tensor(
                        self.cfg.attn_scale,
                        dtype=self.cfg.dtype,
                        device=self.cfg.device,
                    )
                )
            else:
                # For distributed setting, replicate the scalar parameter
                placements = [torch.distributed.tensor.Replicate() for _ in range(device_mesh.ndim)]
                self._attn_scale_param = nn.Parameter(
                    torch.distributed.tensor.empty(
                        (),
                        dtype=self.cfg.dtype,
                        device_mesh=device_mesh,
                        placements=placements,
                    )
                )
                # Initialize with config value
                with torch.no_grad():
                    if isinstance(self._attn_scale_param, DTensor):
                        self._attn_scale_param.to_local().fill_(self.cfg.attn_scale)
                    else:
                        self._attn_scale_param.fill_(self.cfg.attn_scale)

        # Initialize dead latents tracking buffers after all parameters are set up
        if self.cfg.use_auxk and self.cfg.act_fn == "topk":
            if device_mesh is None:
                self.register_buffer('tokens_since_last_activation', torch.zeros(self.cfg.d_sae, device=cfg.device, dtype=torch.long))
                self.register_buffer('is_dead', torch.zeros(self.cfg.d_sae, device=cfg.device, dtype=torch.bool))
            else:
                self.register_buffer('tokens_since_last_activation', torch.distributed.tensor.zeros(
                    self.cfg.d_sae,
                    dtype=torch.long,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["tokens_since_last_activation"].placements(device_mesh),
                ))
                self.register_buffer('is_dead', torch.distributed.tensor.zeros(
                    self.cfg.d_sae,
                    dtype=torch.bool,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["is_dead"].placements(device_mesh),
                ))
    

    @property
    def attn_scale(self) -> float:
        """Return attention scale, either learnable or fixed from config."""
        if self.cfg.use_learnable_attn_scale and hasattr(self, "_attn_scale_param"):
            # Extract scalar value from parameter
            if isinstance(self._attn_scale_param, DTensor):
                return float(self._attn_scale_param.to_local().item())
            else:
                return float(self._attn_scale_param.item())
        else:
            assert self.cfg.attn_scale is not None, "attn_scale must be initialized during config post initialization"
            return self.cfg.attn_scale

    def init_parameters(self, **kwargs):
        """Initialize parameters."""
        super().init_parameters(**kwargs)

        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)

        W_V_bound = 1 / math.sqrt(self.cfg.d_sae)
        torch.nn.init.uniform_(self.W_V, -W_V_bound, W_V_bound)

        W_O_bound = 1 / math.sqrt(self.cfg.d_model)
        torch.nn.init.uniform_(self.W_O, -W_O_bound, W_O_bound)

        torch.nn.init.zeros_(self.b_Q)
        torch.nn.init.zeros_(self.b_K)
        torch.nn.init.zeros_(self.b_V)
        if self.cfg.use_decoder_bias:
            torch.nn.init.zeros_(self.b_D)
        
        # Initialize SmolGen module if enabled
        if self.cfg.use_smolgen and hasattr(self, "smolgen"):
            for module in self.smolgen.modules():
                if isinstance(module, Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, LayerNorm):
                    torch.nn.init.constant_(module.weight, 1.0)
                    torch.nn.init.zeros_(module.bias)

    @torch.no_grad()
    def init_lorsa_with_mhsa(self, encoder_layer):
        assert self.cfg.n_qk_heads % encoder_layer.n_heads == 0
        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        output_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_out]
        
        qk_exp_factor = self.cfg.n_qk_heads // encoder_layer.n_heads
        
        orig_n_heads = encoder_layer.n_heads
        orig_d_head = encoder_layer.d_model // encoder_layer.n_heads
        orig_d_model = encoder_layer.d_model
        orig_W_Q = encoder_layer.mha.q_proj.weight.view(orig_n_heads, orig_d_head, orig_d_model).permute(0, 2, 1)
        orig_b_Q = encoder_layer.mha.q_proj.bias.view(orig_n_heads, orig_d_head)
        orig_W_K = encoder_layer.mha.k_proj.weight.view(orig_n_heads, orig_d_head, orig_d_model).permute(0, 2, 1)
        orig_b_K = encoder_layer.mha.k_proj.bias.view(orig_n_heads, orig_d_head)
        
        if self.device_mesh is not None:
            dim_maps = self.dim_maps()
            model_parallel_rank = self.device_mesh.get_local_rank(mesh_dim="model")
            model_parallel_size = mesh_dim_size(self.device_mesh, "model")
            heads_per_rank = self.cfg.n_qk_heads // model_parallel_size
            lorsa_qk_start_idx = model_parallel_rank * heads_per_rank
            lorsa_qk_end_idx = lorsa_qk_start_idx + heads_per_rank
            lorsa_qk_indices = torch.arange(lorsa_qk_start_idx, lorsa_qk_end_idx)
            expanded_indices = (lorsa_qk_indices // qk_exp_factor).to(orig_W_Q.device)

            W_Q_local = (orig_W_Q[expanded_indices].to(self.cfg.dtype) / input_norm_factor).to(self.cfg.device)
            W_K_local = (orig_W_K[expanded_indices].to(self.cfg.dtype) / input_norm_factor).to(self.cfg.device)
            b_Q_local = orig_b_Q[expanded_indices].to(self.cfg.dtype).to(self.cfg.device)
            b_K_local = orig_b_K[expanded_indices].to(self.cfg.dtype).to(self.cfg.device)

            W_Q_dt = DTensor.from_local(
                W_Q_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["W_Q"].placements(self.device_mesh),
            )
            W_K_dt = DTensor.from_local(
                W_K_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["W_K"].placements(self.device_mesh),
            )
            b_Q_dt = DTensor.from_local(
                b_Q_local,
                    device_mesh=self.device_mesh,
                placements=dim_maps["b_Q"].placements(self.device_mesh),
                )
            b_K_dt = DTensor.from_local(
                b_K_local,
                    device_mesh=self.device_mesh,
                placements=dim_maps["b_K"].placements(self.device_mesh),
            )

            self.W_Q.copy_(W_Q_dt)
            self.W_K.copy_(W_K_dt)
            self.b_Q.copy_(b_Q_dt)
            self.b_K.copy_(b_K_dt)
        else:
            self.W_Q.copy_(
                torch.repeat_interleave(orig_W_Q, qk_exp_factor, dim=0).to(self.cfg.dtype) / input_norm_factor
            )
            self.b_Q.copy_(
                torch.repeat_interleave(orig_b_Q, qk_exp_factor, dim=0).to(self.cfg.dtype)
            )
            self.W_K.copy_(
                torch.repeat_interleave(orig_W_K, qk_exp_factor, dim=0).to(self.cfg.dtype) / input_norm_factor
            )
            self.b_K.copy_(
                torch.repeat_interleave(orig_b_K, qk_exp_factor, dim=0).to(self.cfg.dtype)
            )
        
        if self.cfg.use_smolgen:
            if self.device_mesh is not None:
                # Distributed initialization
                dim_maps = self.dim_maps()
                model_parallel_rank = self.device_mesh.get_local_rank(mesh_dim="model")
                model_parallel_size = mesh_dim_size(self.device_mesh, "model")
                
                # Get source parameters (convert to local if DTensor)
                def _to_local(param):
                    if isinstance(param, DTensor):
                        return param.to_local()
                    return param
                
                orig_compress_weight = _to_local(encoder_layer.mha.smolgen.compress.weight)
                orig_dense1_weight = _to_local(encoder_layer.mha.smolgen.dense1.weight)
                orig_dense1_bias = _to_local(encoder_layer.mha.smolgen.dense1.bias)
                orig_ln1_weight = _to_local(encoder_layer.mha.smolgen.ln1.weight)
                orig_ln1_bias = _to_local(encoder_layer.mha.smolgen.ln1.bias)
                orig_dense2_weight = _to_local(encoder_layer.mha.smolgen.dense2.weight)
                orig_dense2_bias = _to_local(encoder_layer.mha.smolgen.dense2.bias)
                orig_ln2_weight = _to_local(encoder_layer.mha.smolgen.ln2.weight)
                orig_ln2_bias = _to_local(encoder_layer.mha.smolgen.ln2.bias)
                orig_smol_weight_gen_weight = _to_local(encoder_layer.mha.smolgen.smol_weight_gen.weight)
                
                # Compress: shard along d_model (dim 1)
                # compress.weight shape: [32, d_model]
                # Need to extract local slice based on model parallel rank
                compress_weight_full = (orig_compress_weight / input_norm_factor).to(self.cfg.dtype).to(self.cfg.device)
                compress_slices = dim_maps["smolgen.compress.weight"].local_slices(
                    (32, self.cfg.d_model), self.device_mesh
                )
                compress_weight_local = compress_weight_full[compress_slices]
                compress_weight_dt = DTensor.from_local(
                    compress_weight_local,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.compress.weight"].placements(self.device_mesh),
                )
                self.smolgen.compress.weight.copy_(compress_weight_dt)
                
                # Dense1 and LN1: replicate (shared across heads)
                dense1_weight_local = orig_dense1_weight.to(self.cfg.dtype).to(self.cfg.device)
                dense1_weight_dt = DTensor.from_local(
                    dense1_weight_local,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.dense1.weight"].placements(self.device_mesh),
                )
                self.smolgen.dense1.weight.copy_(dense1_weight_dt)
                
                dense1_bias_local = orig_dense1_bias.to(self.cfg.dtype).to(self.cfg.device)
                dense1_bias_dt = DTensor.from_local(
                    dense1_bias_local,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.dense1.bias"].placements(self.device_mesh),
                )
                self.smolgen.dense1.bias.copy_(dense1_bias_dt)
                
                ln1_weight_local = orig_ln1_weight.to(self.cfg.dtype).to(self.cfg.device)
                ln1_weight_dt = DTensor.from_local(
                    ln1_weight_local,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.ln1.weight"].placements(self.device_mesh),
                )
                self.smolgen.ln1.weight.copy_(ln1_weight_dt)
                
                ln1_bias_local = orig_ln1_bias.to(self.cfg.dtype).to(self.cfg.device)
                ln1_bias_dt = DTensor.from_local(
                    ln1_bias_local,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.ln1.bias"].placements(self.device_mesh),
                )
                self.smolgen.ln1.bias.copy_(ln1_bias_dt)

                expanded_dense2_weight = torch.repeat_interleave(
                    orig_dense2_weight, qk_exp_factor, dim=0
                ).to(self.cfg.dtype).to(self.cfg.device)
                expanded_dense2_bias = torch.repeat_interleave(
                    orig_dense2_bias, qk_exp_factor, dim=0
                ).to(self.cfg.dtype).to(self.cfg.device)
                expanded_ln2_weight = torch.repeat_interleave(
                    orig_ln2_weight, qk_exp_factor, dim=0
                ).to(self.cfg.dtype).to(self.cfg.device)
                expanded_ln2_bias = torch.repeat_interleave(
                    orig_ln2_bias, qk_exp_factor, dim=0
                ).to(self.cfg.dtype).to(self.cfg.device)

                dense2_weight_dt = DTensor.from_local(
                    expanded_dense2_weight,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.dense2.weight"].placements(self.device_mesh),
                )
                self.smolgen.dense2.weight.copy_(dense2_weight_dt)
                
                dense2_bias_dt = DTensor.from_local(
                    expanded_dense2_bias,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.dense2.bias"].placements(self.device_mesh),
                )
                self.smolgen.dense2.bias.copy_(dense2_bias_dt)
                
                ln2_weight_dt = DTensor.from_local(
                    expanded_ln2_weight,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.ln2.weight"].placements(self.device_mesh),
                )
                self.smolgen.ln2.weight.copy_(ln2_weight_dt)
                
                ln2_bias_dt = DTensor.from_local(
                    expanded_ln2_bias,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.ln2.bias"].placements(self.device_mesh),
                )
                self.smolgen.ln2.bias.copy_(ln2_bias_dt)
                
                # SmolWeightGen: replicate (shared across heads)
                smol_weight_gen_weight_local = orig_smol_weight_gen_weight.to(self.cfg.dtype).to(self.cfg.device)
                smol_weight_gen_weight_dt = DTensor.from_local(
                    smol_weight_gen_weight_local,
                    device_mesh=self.device_mesh,
                    placements=dim_maps["smolgen.smol_weight_gen.weight"].placements(self.device_mesh),
                )
                self.smolgen.smol_weight_gen.weight.copy_(smol_weight_gen_weight_dt)
            else:
                # Local initialization
                self.smolgen.compress.weight.copy_(encoder_layer.mha.smolgen.compress.weight / input_norm_factor)
                self.smolgen.dense1.weight.copy_(encoder_layer.mha.smolgen.dense1.weight)
                self.smolgen.dense1.bias.copy_(encoder_layer.mha.smolgen.dense1.bias)
                self.smolgen.ln1.weight.copy_(encoder_layer.mha.smolgen.ln1.weight)
                self.smolgen.ln1.bias.copy_(encoder_layer.mha.smolgen.ln1.bias)
                self.smolgen.dense2.weight.copy_(encoder_layer.mha.smolgen.dense2.weight.repeat_interleave(qk_exp_factor, dim=0))
                self.smolgen.dense2.bias.copy_(encoder_layer.mha.smolgen.dense2.bias.repeat_interleave(qk_exp_factor, dim=0))
                self.smolgen.ln2.weight.copy_(encoder_layer.mha.smolgen.ln2.weight.repeat_interleave(qk_exp_factor, dim=0))
                self.smolgen.ln2.bias.copy_(encoder_layer.mha.smolgen.ln2.bias.repeat_interleave(qk_exp_factor, dim=0))
                self.smolgen.smol_weight_gen.weight.copy_(encoder_layer.mha.smolgen.smol_weight_gen.weight) # modified: a shared smol_weight_gen.weight 


    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def init_W_D_with_active_subspace_per_head(
        self, activation_batch: dict[str, torch.Tensor], encoder_layer
    ):
        """
        Initialize W_D with the active subspace for each head.
        """
        
        orig_n_heads = encoder_layer.n_heads
        orig_d_model = encoder_layer.d_model
        orig_d_head = orig_d_model // orig_n_heads
        n_ov_per_orig_head = self.cfg.n_ov_heads // orig_n_heads
        W_V = encoder_layer.mha.v_proj.weight.view(orig_n_heads, orig_d_head, orig_d_model).permute(0, 2, 1)
        W_O = encoder_layer.mha.out_proj.weight.T.view(orig_n_heads, orig_d_head, orig_d_model)
        
        x = self.prepare_input(activation_batch)[0]
        if isinstance(x, DTensor):
            x = x.to_local()

        captured_z = None

        def capture_hook(tensor, hook):
            nonlocal captured_z
            captured_z = tensor.clone().detach()
            return tensor

        encoder_layer.mha.hook_z.add_hook(capture_hook)
        _ = encoder_layer.forward(x)
        # W_O.shape = torch.Size([32, 32, 1024])
        # captured_z.shape = torch.Size([256, 32, 64, 32])
        
        output_per_head = torch.einsum("b n s h, n h d -> b s n d", captured_z, W_O)
        n_ov_per_orig_head = self.cfg.n_ov_heads // encoder_layer.n_heads
        if self.device_mesh is not None:
            assert isinstance(self.W_O, DTensor)
            assert isinstance(self.W_V, DTensor)
            model_parallel_rank = self.device_mesh.get_local_rank(mesh_dim="model")
            model_parallel_size = mesh_dim_size(self.device_mesh, "model")
            orig_start_idx = model_parallel_rank * encoder_layer.n_heads // model_parallel_size
            orig_end_idx = orig_start_idx + encoder_layer.n_heads // model_parallel_size
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
                    W_O_local[start_idx:end_idx] @ (W_V[orig_head_index] @ W_O[orig_head_index]).T
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
            for orig_head_index in range(encoder_layer.n_heads):
                output = output_per_head[:, :, orig_head_index, :]
                output_flattened = output.flatten(0, 1)
                demeaned_output = output_flattened - output_flattened.mean(dim=0)
                U, S, V = torch.svd(demeaned_output.T.to(torch.float32))
                proj_weight = U[:, : self.cfg.d_qk_head]
                self.W_O.data[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head] = (
                    self.W_O.data[
                        orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head,
                        : self.cfg.d_qk_head,
                    ]
                    @ proj_weight.T
                )
                self.W_V.data[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head] = (
                    self.W_O.data[orig_head_index * n_ov_per_orig_head : (orig_head_index + 1) * n_ov_per_orig_head]
                    @ (W_V[orig_head_index] @ W_O[orig_head_index]).T
                )
            self.W_V.copy_(self.W_V.data / self.W_V.data.norm(dim=1, keepdim=True))
            self.W_O.copy_(self.W_O.data / self.W_O.data.norm(dim=1, keepdim=True))

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
        query = q.permute(0, 2, 1, 3)
        key = k.permute(0, 2, 1, 3)
        value = v.reshape(*k.shape[:3], -1).permute(0, 2, 1, 3)
        with sdpa_kernel(
            backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        ):
            z = F.scaled_dot_product_attention(
                query, key, value, scale=1 / self.attn_scale, is_causal=False, enable_gqa=False
            )
        return z.permute(0, 2, 1, 3).reshape(*v.shape)

    @overload
    def compute_attn_scores(
        self, x: Float[torch.Tensor, "batch seq_len d_model"], return_q_k: Literal[False] = False
    ) -> Float[torch.Tensor, "batch n_qk_heads seq_len seq_len"]: ...

    @overload
    def compute_attn_scores(
        self, x: Float[torch.Tensor, "batch seq_len d_model"], return_q_k: Literal[True]
    ) -> Tuple[
        Float[torch.Tensor, "batch n_qk_heads seq_len seq_len"],
        Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
    ]: ...

    def compute_attn_scores(
        self, x: Float[torch.Tensor, "batch seq_len d_model"], return_q_k: bool = False
    ) -> Union[
        Float[torch.Tensor, "batch n_qk_heads seq_len seq_len"],
        Tuple[
            Float[torch.Tensor, "batch n_qk_heads seq_len seq_len"],
            Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
            Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        ],
    ]:
        """Compute the attention scores."""
        q, k, v = self._compute_qkv(x)
        q = self.hook_q(q)
        k = self.hook_k(k)
        q_ = q.permute(2, 0, 1, 3)
        k_ = k.permute(2, 0, 3, 1)
        scores = torch.einsum("nbqd,nbdk->nbqk", q_, k_) / self.attn_scale
        # scores = self._apply_causal_mask(scores)
        scores = scores.permute(1, 0, 2, 3)
        if return_q_k:
            return (
                scores,
                q,
                k,
            )  # (batch, n_qk_heads, seq_len, seq_len), (batch, seq_len, n_qk_heads, d_qk_head), (batch, seq_len, n_qk_heads, d_qk_head)
        else:
            return scores  # (batch, n_qk_heads, seq_len, seq_len)

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        raise NotImplementedError("It does not make sense to init the encoder with the decoder transpose for Lorsa")

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
        **kwargs
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
        attn_mask: Optional[torch.Tensor] = None
        local_n_qk_heads = q.to_local().shape[2] if isinstance(q, DTensor) else None

        # Optional SmolGen additive bias as attention mask: [B, H, S, S]
        if getattr(self.cfg, "use_smolgen", True):
            if hasattr(self, "smolgen"):
                bias = self.smolgen(x)
                if bias.dim() == 3:
                    target_heads = (
                        local_n_qk_heads if local_n_qk_heads is not None and not isinstance(bias, DTensor)
                        else self.cfg.n_qk_heads
                    )
                    bias = bias.unsqueeze(1).repeat_interleave(target_heads, dim=1)
                elif bias.dim() == 4:
                    h = bias.shape[1]
                    if h == 1:
                        target_heads = (
                            local_n_qk_heads if local_n_qk_heads is not None and not isinstance(bias, DTensor)
                            else self.cfg.n_qk_heads
                        )
                        bias = bias.repeat_interleave(target_heads, dim=1)
                    elif h == self.cfg.n_qk_heads:
                        pass
                    elif local_n_qk_heads is not None and not isinstance(bias, DTensor) and h == local_n_qk_heads:
                        pass
                    else:
                        if self.cfg.n_qk_heads % h == 0:
                            bias = bias.repeat_interleave(self.cfg.n_qk_heads // h, dim=1)
                        elif local_n_qk_heads is not None and not isinstance(bias, DTensor) and local_n_qk_heads % h == 0:
                            bias = bias.repeat_interleave(local_n_qk_heads // h, dim=1)
                        else:
                            raise ValueError(
                                f"SmolGen output head dim {h} cannot be aligned with global heads {self.cfg.n_qk_heads}"
                            )
                else:
                    raise ValueError("SmolGen output must be [B,S,S] or [B,1,S,S] or [B,H,S,S]")
                attn_mask = bias.to(q.dtype) * float(self.smolgen_score_scale.item())
            else:
                print("no SmolGen")
                attn_mask = None

        has_dtensor_attn_mask = attn_mask is not None and any(
            isinstance(t, DTensor) for t in (q, k, v, attn_mask)
        )
        use_fast_sdpa = not (return_attention_pattern or return_attention_score or has_dtensor_attn_mask)

        if use_fast_sdpa:
            query = q.permute(0, 2, 1, 3)
            key = k.permute(0, 2, 1, 3)
            value = v.reshape(*k.shape[:3], -1).permute(0, 2, 1, 3)
            with sdpa_kernel(
                backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                z = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    scale=1 / self.attn_scale,
                    is_causal=False,
                    enable_gqa=True,
                )
            hidden_pre = z.permute(0, 2, 1, 3).reshape(*v.shape)
        elif has_dtensor_attn_mask:
            assert self.device_mesh is not None
            assert isinstance(q, DTensor) and isinstance(k, DTensor) and isinstance(v, DTensor)

            q_local = q.to_local()
            k_local = k.to_local()
            v_local = v.to_local()

            attn_mask_local: Optional[torch.Tensor] = None
            if isinstance(attn_mask, DTensor):
                attn_mask_local = attn_mask.to_local(
                    grad_placements=replace_placements(
                        attn_mask.placements,
                        attn_mask.device_mesh,
                        "model",
                        Partial("sum"),
                    )
                )
            elif attn_mask is not None:
                attn_mask_local = attn_mask

            local_n_qk_heads = q_local.shape[2]
            if attn_mask_local is not None and attn_mask_local.shape[1] != local_n_qk_heads:
                if attn_mask_local.shape[1] % local_n_qk_heads != 0:
                    raise ValueError(
                        f"Local smolgen head dim {attn_mask_local.shape[1]} cannot be aligned with local q heads {local_n_qk_heads}"
                    )
                head_slice = DimMap({"model": 1}).local_slices(
                    (1, self.cfg.n_qk_heads, 1, 1),
                    self.device_mesh,
                )[1]
                attn_mask_local = attn_mask_local[:, head_slice, :, :]

            if not (return_attention_pattern or return_attention_score):
                query_local = q_local.permute(0, 2, 1, 3)
                key_local = k_local.permute(0, 2, 1, 3)
                value_local = v_local.reshape(*k_local.shape[:3], -1).permute(0, 2, 1, 3)
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
                        attn_mask=attn_mask_local,
                        scale=1 / self.attn_scale,
                        is_causal=False,
                        enable_gqa=True,
                    )
                hidden_pre_local = z_local.permute(0, 2, 1, 3).reshape(*v_local.shape)
            else:
                q_local_heads = q_local.permute(2, 0, 1, 3)
                k_local_heads = k_local.permute(2, 0, 3, 1)
                scores = torch.einsum("nbqd,nbdk->nbqk", q_local_heads, k_local_heads) / self.attn_scale
                if attn_mask_local is not None:
                    scores = scores + attn_mask_local.permute(1, 0, 2, 3).to(scores.dtype)
                pattern = F.softmax(scores, dim=-1)

                v_local_heads = einops.rearrange(v_local, "b seq h -> b h seq")
                local_n_ov_heads = v_local_heads.shape[1]
                if local_n_qk_heads != local_n_ov_heads:
                    v_local_grouped = v_local_heads.view(
                        v_local_heads.shape[0],
                        local_n_qk_heads,
                        local_n_ov_heads // local_n_qk_heads,
                        v_local_heads.shape[2],
                    )
                    v_local_grouped = v_local_grouped.permute(1, 0, 2, 3)
                    z_local = torch.einsum("hbqk,hbrk->hbrq", pattern, v_local_grouped)
                    hidden_pre_local = z_local.permute(1, 0, 2, 3).flatten(start_dim=1, end_dim=2).permute(0, 2, 1)
                else:
                    z_local = torch.einsum("hbqk,hbk->hbq", pattern, v_local_heads.permute(1, 0, 2))
                    hidden_pre_local = z_local.permute(1, 2, 0)

            hidden_pre = DTensor.from_local(
                hidden_pre_local,
                device_mesh=self.device_mesh,
                placements=v.placements,
            )
        else:
            q = q.permute(2, 0, 1, 3)
            k = k.permute(2, 0, 3, 1)
            scores = torch.einsum("nbqd,nbdk->nbqk", q, k) / self.attn_scale
            if attn_mask is not None:
                scores = scores + attn_mask.permute(1, 0, 2, 3).to(scores.dtype)
            pattern = F.softmax(scores, dim=-1)
            hidden_pre = self._compute_head_outputs(pattern, v)

        # Scale feature activations by decoder norm if configured
        if self.cfg.sparsity_include_decoder_norm:
            hidden_pre = hidden_pre * self.decoder_norm()

        feature_acts = self.activation_function(hidden_pre)

        if self.cfg.sparsity_include_decoder_norm:
            feature_acts = feature_acts / self.decoder_norm()

        return_values: list[torch.Tensor] = [feature_acts]
        if return_hidden_pre:
            return_values.append(hidden_pre)
        if return_attention_pattern and pattern is not None:
            return_values.append(pattern.permute(1, 0, 2, 3))
        if return_attention_score and scores is not None:
            return_values.append(scores.permute(1, 0, 2, 3))
            
        return tuple(return_values) if len(return_values) > 1 else return_values[0]  # type: ignore[return-value]


    @torch.no_grad()
    def init_attn_scale_from_encoder(self, encoder_layer: nn.Module):
        """Initialize this LORSA's attn_scale from an existing encoder's attn_scale."""
        if getattr(self.cfg, "use_learnable_attn_scale", False):
            return
        if not hasattr(self, "_attn_scale_param"):
            raise ValueError("This LORSA has no learnable attn_scale to initialize.")
        if not hasattr(encoder_layer.mha, "qk_scale"):
            raise ValueError("Encoder layer has no qk_scale to initialize from.")

        # Get source qk_scale (convert to local if DTensor)
        source_qk_scale = encoder_layer.mha.qk_scale
        if isinstance(source_qk_scale, DTensor):
            source_qk_scale = source_qk_scale.to_local()
        
        target_value = 1.0 / source_qk_scale.item()
        
        if self.device_mesh is not None:
            # For distributed setting, update the replicated parameter
            if isinstance(self._attn_scale_param, DTensor):
                # All ranks update their local copy (which should be identical due to Replicate)
                self._attn_scale_param.to_local().fill_(target_value)
            else:
                self._attn_scale_param.data.fill_(target_value)
        else:
            # Local case
            self._attn_scale_param.data.fill_(target_value)

    @torch.no_grad()
    def init_smolgen_from_encoder(self, encoder_layer: nn.Module):
        """Initialize this LORSA's SmolGen module from an existing encoder's SmolGen."""
        if not getattr(self.cfg, "use_smolgen", False):
            return
        if not hasattr(self, "smolgen"):
            raise ValueError("This LORSA has no SmolGen to initialize.")
        if not hasattr(encoder_layer, "smolgen"):
            raise ValueError("Encoder layer has no SmolGen to initialize from.")
        
        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        
        # 复制 SmolGen 的状态
        qk_exp_factor = self.cfg.n_qk_heads // encoder_layer.n_heads
        
        if self.device_mesh is not None:
            # Distributed initialization
            dim_maps = self.dim_maps()
            model_parallel_rank = self.device_mesh.get_local_rank(mesh_dim="model")
            model_parallel_size = mesh_dim_size(self.device_mesh, "model")
            
            # Helper function to convert DTensor to local
            def _to_local(param):
                if isinstance(param, DTensor):
                    return param.to_local()
                return param
            
            # Get source parameters
            orig_compress_weight = _to_local(encoder_layer.smolgen.compress.weight)
            orig_dense1_weight = _to_local(encoder_layer.smolgen.dense1.weight)
            orig_dense1_bias = _to_local(encoder_layer.smolgen.dense1.bias)
            orig_ln1_weight = _to_local(encoder_layer.smolgen.ln1.weight)
            orig_ln1_bias = _to_local(encoder_layer.smolgen.ln1.bias)
            orig_dense2_weight = _to_local(encoder_layer.smolgen.dense2.weight)
            orig_dense2_bias = _to_local(encoder_layer.smolgen.dense2.bias)
            orig_ln2_weight = _to_local(encoder_layer.smolgen.ln2.weight)
            orig_ln2_bias = _to_local(encoder_layer.smolgen.ln2.bias)
            orig_smol_weight_gen_weight = _to_local(encoder_layer.smolgen.smol_weight_gen.weight)
            
            # Compress: shard along d_model (dim 1)
            # compress.weight shape: [32, d_model]
            compress_weight_full = orig_compress_weight.to(self.cfg.dtype).to(self.cfg.device)
            compress_slices = dim_maps["smolgen.compress.weight"].local_slices(
                (32, self.cfg.d_model), self.device_mesh
            )
            compress_weight_local = compress_weight_full[compress_slices]
            compress_weight_dt = DTensor.from_local(
                compress_weight_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.compress.weight"].placements(self.device_mesh),
            )
            self.smolgen.compress.weight.copy_(compress_weight_dt)
            
            # Dense1 and LN1: replicate (shared across heads)
            dense1_weight_local = orig_dense1_weight.to(self.cfg.dtype).to(self.cfg.device)
            dense1_weight_dt = DTensor.from_local(
                dense1_weight_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.dense1.weight"].placements(self.device_mesh),
            )
            self.smolgen.dense1.weight.copy_(dense1_weight_dt)
            
            dense1_bias_local = orig_dense1_bias.to(self.cfg.dtype).to(self.cfg.device)
            dense1_bias_dt = DTensor.from_local(
                dense1_bias_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.dense1.bias"].placements(self.device_mesh),
            )
            self.smolgen.dense1.bias.copy_(dense1_bias_dt)
            
            ln1_weight_local = orig_ln1_weight.to(self.cfg.dtype).to(self.cfg.device)
            ln1_weight_dt = DTensor.from_local(
                ln1_weight_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.ln1.weight"].placements(self.device_mesh),
            )
            self.smolgen.ln1.weight.copy_(ln1_weight_dt)
            
            ln1_bias_local = orig_ln1_bias.to(self.cfg.dtype).to(self.cfg.device)
            ln1_bias_dt = DTensor.from_local(
                ln1_bias_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.ln1.bias"].placements(self.device_mesh),
            )
            self.smolgen.ln1.bias.copy_(ln1_bias_dt)
            
            # Dense2 and LN2: shard along n_qk_heads dimension (dim 0)
            # First expand, then extract local slice
            expanded_dense2_weight = torch.repeat_interleave(orig_dense2_weight, qk_exp_factor, dim=0).to(self.cfg.dtype)
            expanded_dense2_bias = torch.repeat_interleave(orig_dense2_bias, qk_exp_factor, dim=0).to(self.cfg.dtype)
            expanded_ln2_weight = torch.repeat_interleave(orig_ln2_weight, qk_exp_factor, dim=0).to(self.cfg.dtype)
            expanded_ln2_bias = torch.repeat_interleave(orig_ln2_bias, qk_exp_factor, dim=0).to(self.cfg.dtype)
            
            # Extract local slice based on model parallel rank
            heads_per_rank = self.cfg.n_qk_heads // model_parallel_size
            dense2_start_idx = model_parallel_rank * heads_per_rank * 256
            dense2_end_idx = (model_parallel_rank + 1) * heads_per_rank * 256
            
            dense2_weight_local = expanded_dense2_weight[dense2_start_idx:dense2_end_idx].to(self.cfg.device)
            dense2_bias_local = expanded_dense2_bias[dense2_start_idx:dense2_end_idx].to(self.cfg.device)
            ln2_weight_local = expanded_ln2_weight[dense2_start_idx:dense2_end_idx].to(self.cfg.device)
            ln2_bias_local = expanded_ln2_bias[dense2_start_idx:dense2_end_idx].to(self.cfg.device)
            
            dense2_weight_dt = DTensor.from_local(
                dense2_weight_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.dense2.weight"].placements(self.device_mesh),
            )
            self.smolgen.dense2.weight.copy_(dense2_weight_dt)
            
            dense2_bias_dt = DTensor.from_local(
                dense2_bias_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.dense2.bias"].placements(self.device_mesh),
            )
            self.smolgen.dense2.bias.copy_(dense2_bias_dt)
            
            ln2_weight_dt = DTensor.from_local(
                ln2_weight_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.ln2.weight"].placements(self.device_mesh),
            )
            self.smolgen.ln2.weight.copy_(ln2_weight_dt)
            
            ln2_bias_dt = DTensor.from_local(
                ln2_bias_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.ln2.bias"].placements(self.device_mesh),
            )
            self.smolgen.ln2.bias.copy_(ln2_bias_dt)
            
            # SmolWeightGen: replicate (shared across heads)
            smol_weight_gen_weight_local = orig_smol_weight_gen_weight.to(self.cfg.dtype).to(self.cfg.device)
            smol_weight_gen_weight_dt = DTensor.from_local(
                smol_weight_gen_weight_local,
                device_mesh=self.device_mesh,
                placements=dim_maps["smolgen.smol_weight_gen.weight"].placements(self.device_mesh),
            )
            self.smolgen.smol_weight_gen.weight.copy_(smol_weight_gen_weight_dt)
            
            # Update smolgen_score_scale (replicated buffer)
            scale_value = (input_norm_factor ** 2)
            if isinstance(self.smolgen_score_scale, DTensor):
                self.smolgen_score_scale.to_local().fill_(scale_value)
            else:
                self.smolgen_score_scale.data.fill_(scale_value)
        else:
            # Local initialization (single device)
            # 复制基础层的权重
            self.smolgen.compress.weight.copy_(encoder_layer.smolgen.compress.weight.to(self.cfg.dtype))
            self.smolgen.dense1.weight.copy_(encoder_layer.smolgen.dense1.weight.to(self.cfg.dtype))
            self.smolgen.dense1.bias.copy_(encoder_layer.smolgen.dense1.bias.to(self.cfg.dtype))
            self.smolgen.ln1.weight.copy_(encoder_layer.smolgen.ln1.weight.to(self.cfg.dtype))
            self.smolgen.ln1.bias.copy_(encoder_layer.smolgen.ln1.bias.to(self.cfg.dtype))
            
            # Expand dense2 and ln2
            original_dense2_weight = encoder_layer.smolgen.dense2.weight
            original_dense2_bias = encoder_layer.smolgen.dense2.bias
            expanded_dense2_weight = torch.repeat_interleave(
                original_dense2_weight, qk_exp_factor, dim=0
            ).to(self.cfg.dtype)
            expanded_dense2_bias = torch.repeat_interleave(
                original_dense2_bias, qk_exp_factor, dim=0
            ).to(self.cfg.dtype)
            
            self.smolgen.dense2.weight.data = expanded_dense2_weight
            self.smolgen.dense2.bias.data = expanded_dense2_bias
            
            # 同样处理ln2
            original_ln2_weight = encoder_layer.smolgen.ln2.weight
            original_ln2_bias = encoder_layer.smolgen.ln2.bias
            
            expanded_ln2_weight = torch.repeat_interleave(
                original_ln2_weight, qk_exp_factor, dim=0
            ).to(self.cfg.dtype)
            expanded_ln2_bias = torch.repeat_interleave(
                original_ln2_bias, qk_exp_factor, dim=0
            ).to(self.cfg.dtype)
            
            self.smolgen.ln2.weight.data = expanded_ln2_weight
            self.smolgen.ln2.bias.data = expanded_ln2_bias
            self.smolgen.smol_weight_gen.weight.copy_(encoder_layer.smolgen.smol_weight_gen.weight.to(self.cfg.dtype))
            
            # Update smolgen_score_scale
            self.smolgen_score_scale.data = self.smolgen_score_scale.data * (input_norm_factor ** 2)


    def _decode_dtensor_local(self, feature_acts: DTensor, *, add_bias: bool) -> DTensor:
        assert self.device_mesh is not None
        assert isinstance(self.W_O, DTensor)

        out_local = torch.einsum(
            "bps,sd->bpd",
            feature_acts.to_local(),
            self.W_O.to_local(
                grad_placements=replace_placements(
                    self.W_O.placements,
                    self.W_O.device_mesh,
                    "data",
                    Partial("sum"),
                )
            ),
        )
        if add_bias and self.cfg.use_decoder_bias:
            assert isinstance(self.b_D, DTensor)
            out_local = out_local + self.b_D.to_local(
                grad_placements=replace_placements(
                    self.b_D.placements,
                    self.b_D.device_mesh,
                    "data",
                    Partial("sum"),
                )
            )

        return DTensor.from_local(
            out_local,
            device_mesh=self.device_mesh,
            placements=DimMap({"data": 0}).placements(self.device_mesh),
        )

    @override
    def decode(self, feature_acts, **kwargs):
        """Decode head activations to output."""
        if feature_acts.layout == torch.sparse_coo:
            return (
                torch.sparse.mm(
                    feature_acts.to(torch.float32),
                    self.W_O.to(torch.float32),
                ).to(self.cfg.dtype)
                + self.b_D
            )
        if feature_acts.ndim == 2:
            feature_acts = feature_acts.unsqueeze(0)
        if self.device_mesh is not None and isinstance(feature_acts, DTensor):
            out = self._decode_dtensor_local(feature_acts, add_bias=True)
        else:
            out = torch.einsum("bps,sd->bpd", feature_acts, self.W_O)
            if self.cfg.use_decoder_bias:
                out = out + self.b_D
        if self.cfg.skip_bos:
            out = out[:, 1:]
        return out

    def _compute_qkv(
        self, x: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> tuple[
        Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        Float[torch.Tensor, "batch seq_len n_qk_heads d_qk_head"],
        Float[torch.Tensor, "batch seq_len n_ov_heads"],
    ]:
        """Compute queries, keys, values."""
        q = (
            torch.einsum(
                "bsd,Qdq->bsQq",
                x,
                self.W_Q,
            )
            + self.b_Q
        )
        k = (
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

        return q, k, v

    def encode_z_pattern_for_head(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        head_idx: Int[torch.Tensor, " n_active_features"],
    ) -> Float[torch.Tensor, "n_active_features k_pos"]:
        assert x.size(0) == 1, f"x must be of shape (1, seq_len, d_model), but got {x.shape}"
        qk_idx: Tensor = head_idx // (self.cfg.n_ov_heads // self.cfg.n_qk_heads)
        q, k, v = self._compute_qkv(x)

        # (n_active_features, q_pos, k_pos)
        pattern = self._compute_attention_pattern(q, k)[qk_idx, 0]
        return pattern.mul_(v[0, :, head_idx, None].permute(1, 2, 0))

    def encode_z_patterns(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
    ):
        assert x.size(0) == 1, f"x must be of shape (1, seq_len, d_model), but got {x.shape}"

        head_idx = torch.arange(self.cfg.d_sae)
        qk_idx: Tensor = head_idx // (self.cfg.n_ov_heads // self.cfg.n_qk_heads)
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
        x_rot = x[:, :, :, : self.cfg.rotary_dim]
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
        # scores = self._apply_causal_mask(scores)
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
            v = v.view(v.shape[0], self.cfg.n_qk_heads, self.cfg.n_ov_heads // self.cfg.n_qk_heads, v.shape[2])
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
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        self.W_O.data = self.W_O.data * value / self.W_O.data.norm(dim=1, keepdim=True)

    @override
    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        """Set encoder weights to fixed norm."""
        raise NotImplementedError("set_encoder_to_fixed_norm does not make sense for lorsa")

    @torch.no_grad()
    def update_dead_latents(self, feature_acts: torch.Tensor):
        """Update the dead latents tracking based on current feature activations.
        
        Args:
            feature_acts: Feature activations tensor of shape (batch, d_sae) or (batch, seq_len, d_sae)
        """
        if not (self.cfg.use_auxk and self.cfg.act_fn == "topk"):
            return
        
        # Convert DTensor to local tensor for operations that don't support DTensor
        is_dtensor = isinstance(feature_acts, DTensor)
        if is_dtensor:
            feature_acts = feature_acts.to_local()
            
        # Calculate batch size (number of tokens in this batch)
        if feature_acts.dim() == 3:  # (batch, seq_len, d_sae)
            batch_size = feature_acts.size(0) * feature_acts.size(1)  # batch * seq_len
        else:  # (batch, d_sae)
            batch_size = feature_acts.size(0)  # batch
            
        # Check which features were activated in this batch
        if feature_acts.dim() == 3:  # (batch, seq_len, d_sae)
            # Use sequential any operations for DTensor compatibility
            activated = feature_acts.gt(0).any(dim=0).any(dim=0)  # (d_sae,)
        else:  # (batch, d_sae)
            activated = feature_acts.gt(0).any(dim=0)  # (d_sae,)
        
        # Convert back to DTensor if needed
        if is_dtensor and isinstance(self.tokens_since_last_activation, DTensor):
            activated = DTensor.from_local(
                activated,
                device_mesh=self.device_mesh,
                placements=self.tokens_since_last_activation.placements,
            )
            
        # Update tokens since last activation
        # If a feature was activated, reset to 0; otherwise, add batch_size
        self.tokens_since_last_activation = torch.where(
            activated,
            torch.zeros_like(self.tokens_since_last_activation),
            self.tokens_since_last_activation + batch_size
        )
        
        # Mark as dead if tokens since last activation exceeds threshold
        self.is_dead = self.tokens_since_last_activation >= self.cfg.dead_threshold
    

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: Literal[True] = True,
        **kwargs,
    ) -> tuple[
        Float[torch.Tensor, " batch"],
        tuple[dict[str, Optional[torch.Tensor]], dict[str, torch.Tensor]],
    ]: ...

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: Literal[False],
        **kwargs,
    ) -> Float[torch.Tensor, " batch"]: ...

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        label: (
            Union[
                Float[torch.Tensor, "batch d_model"],
                Float[torch.Tensor, "batch seq_len d_model"],
            ]
            | None
        ) = None,
        *,
        use_batch_norm_mse: bool = False,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        l1_coefficient: float = 1.0,
        return_aux_data: bool = True,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, " batch"],
        tuple[
            Float[torch.Tensor, " batch"],
            tuple[dict[str, Optional[torch.Tensor]], dict[str, torch.Tensor]],
        ],
    ]:
        """Compute the loss for the autoencoder.
        Ensure that the input activations are normalized by calling `normalize_activations` before calling this method.
        """
        x, encoder_kwargs, decoder_kwargs = self.prepare_input(batch)
        label = self.prepare_label(batch, **kwargs)

        feature_acts, hidden_pre = self.encode(x, return_hidden_pre=True, **encoder_kwargs)
        reconstructed = self.decode(feature_acts, **decoder_kwargs)
        
        # Update dead latents tracking
        self.update_dead_latents(feature_acts)

        l_rec = (reconstructed - label).pow(2)
        if use_batch_norm_mse:
            l_rec = (
                l_rec
                / (label - label.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
            )
        l_rec = l_rec.sum(dim=-1)
        if isinstance(l_rec, DTensor):
            l_rec = l_rec.full_tensor()
        loss_dict: dict[str, Optional[torch.Tensor]] = {
            "l_rec": l_rec,
        }
        loss = l_rec.mean()

        if sparsity_loss_type is not None:
            if sparsity_loss_type == "power":
                l_s = torch.norm(feature_acts * self.decoder_norm(), p=p, dim=-1)
            elif sparsity_loss_type == "tanh":
                l_s = torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm()).sum(dim=-1)
            elif sparsity_loss_type == "tanh-quad":
                approx_frequency = einops.reduce(
                    torch.tanh(tanh_stretch_coefficient * feature_acts * self.decoder_norm()),
                    "... d_sae -> d_sae",
                    "mean",
                )
                l_s = (approx_frequency * (1 + approx_frequency / frequency_scale)).sum(dim=-1)
            else:
                raise ValueError(f"sparsity_loss_type f{sparsity_loss_type} not supported.")
            if isinstance(l_s, DTensor):
                l_s = l_s.full_tensor()
            l_s = l1_coefficient * l_s
            # WARNING: Some DTensor bugs make if l1_coefficient * l_s goes before full_tensor, the l1_coefficient value will be internally cached. Furthermore, it will cause the backward pass to fail with redistribution error. See https://github.com/pytorch/pytorch/issues/153603 and https://github.com/pytorch/pytorch/issues/153615 .
            loss_dict["l_s"] = l_s
            loss = loss + l_s.mean()
        else:
            loss_dict["l_s"] = None

        loss_dict["l_p"] = None

        # Add AuxK auxiliary loss if enabled
        if self.cfg.use_auxk and self.cfg.act_fn == "topk":
            with timer.time("auxk_loss_calculation"):
                # Get reconstruction error
                e = label - reconstructed  # (batch, d_model) or (batch, seq_len, d_model)
                
                # Get the top-k_aux dead latents based on their activation values
                current_k = self.current_k
                if self.device_mesh is not None:
                    assert isinstance(self.is_dead, DTensor)
                    self.current_k = min(self.cfg.k_aux, int(self.is_dead.full_tensor().sum().item()))
                else:
                    self.current_k = min(self.cfg.k_aux, int(self.is_dead.sum().item()))
                
                if self.current_k > 0:
                    # Scale feature activations by decoder norm if configured
                    if self.cfg.sparsity_include_decoder_norm:
                        dead_sparsity_scores = hidden_pre * self.is_dead * self.decoder_norm()
                    else:
                        dead_sparsity_scores = hidden_pre * self.is_dead

                    dead_activation_mask = self.activation_function(dead_sparsity_scores)
                    dead_feature_acts = torch.clamp(hidden_pre * dead_activation_mask * self.is_dead, min=0.0)
                    
                    # Decode auxiliary feature activations without introducing an explicit
                    # DTensor redistribute op into the backward graph.
                    if self.device_mesh is not None and isinstance(dead_feature_acts, DTensor):
                        aux_reconstructed = self._decode_dtensor_local(dead_feature_acts, add_bias=False)
                    else:
                        aux_reconstructed = dead_feature_acts @ self.W_O
                    l_aux = (e - aux_reconstructed).pow(2).sum(dim=-1)
                else:
                    l_aux = torch.zeros_like(l_rec)
                
                
                if isinstance(l_aux, DTensor):
                    l_aux = l_aux.full_tensor()
                loss_dict["l_aux"] = l_aux
                loss = loss + self.cfg.aux_coefficient * l_aux.mean()
                
                self.current_k = current_k

        if return_aux_data:
            aux_data = {
                "feature_acts": feature_acts,
                "reconstructed": reconstructed,
                "hidden_pre": hidden_pre,
            }
            return loss, (loss_dict, aux_data)
        return loss

    @override
    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor], device_mesh: DeviceMesh | None = None) -> None:
        """Load weights, remapping legacy ``attn_scale`` keys to ``_attn_scale_param``."""
        state_dict = dict(state_dict)
        if "attn_scale" in state_dict and "_attn_scale_param" not in state_dict:
            state_dict["_attn_scale_param"] = state_dict.pop("attn_scale")
        super().load_full_state_dict(state_dict, device_mesh=device_mesh)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str,
        strict_loading: bool = True, # TODO set to true
        fold_activation_scale: bool = True,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Load pretrained model."""
        cfg = LorsaConfig.from_pretrained(pretrained_name_or_path, strict_loading=strict_loading, **kwargs)
        model = cls.from_config(cfg, fold_activation_scale=fold_activation_scale, device_mesh=device_mesh)
        return model

    @override
    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        base_maps = super().dim_maps()
        lorsa_maps = {
            **base_maps,
            "W_Q": DimMap({"model": 0}),
            "W_K": DimMap({"model": 0}),
            "W_V": DimMap({"model": 0}),
            "W_O": DimMap({"model": 0}),
            "b_Q": DimMap({"model": 0}),
            "b_K": DimMap({"model": 0}),
            "b_V": DimMap({"model": 0}),
            "b_D": DimMap({}),
            "smolgen.compress.weight": DimMap({}),
            "smolgen.compress.bias": DimMap({}),
            "smolgen.dense1.weight": DimMap({}),
            "smolgen.dense1.bias": DimMap({}),
            "smolgen.ln1.weight": DimMap({}),
            "smolgen.ln1.bias": DimMap({}),
            "smolgen.dense2.weight": DimMap({}),
            "smolgen.dense2.bias": DimMap({}),
            "smolgen.ln2.weight": DimMap({}),
            "smolgen.ln2.bias": DimMap({}),
            "smolgen.smol_weight_gen.weight": DimMap({}),
            "smolgen.smol_weight_gen.bias": DimMap({}),
        }
        if self.cfg.use_auxk and self.cfg.act_fn == "topk":
            lorsa_maps["tokens_since_last_activation"] = DimMap({"model": 0})
            lorsa_maps["is_dead"] = DimMap({"model": 0})
            
        return lorsa_maps


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
        if self.cfg.skip_bos:
            label = label[:, 1:]
        return label

    @override
    @torch.no_grad()
    def compute_activation_frequency_scores(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Compute activation frequency scores for Lorsa (mean over batch)."""
        return (feature_acts > 0).float().sum(0).mean(0)

    @override
    @torch.no_grad()
    def prepare_logging_data(
        self,
        log_info: dict[str, torch.Tensor],
        label: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Prepare logging data by flattening dimensions for Lorsa."""
        log_info = log_info.copy()
        log_info["reconstructed"] = log_info["reconstructed"].flatten(0, 1)
        label = label.flatten(0, 1)
        return log_info, label

    def _configure_gradient_flow(self):
        def stop_gradient(tensor: torch.Tensor, hook: HookPoint):
            return tensor.detach()

        if self.cfg.use_post_qk_ln:
            self.ln_q.hook_scale.add_hook(stop_gradient, is_permanent=True)
            self.ln_k.hook_scale.add_hook(stop_gradient, is_permanent=True)


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
