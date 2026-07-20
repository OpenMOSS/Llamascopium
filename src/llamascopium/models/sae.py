import math
from typing import Any, Literal, Union, overload

import torch
import torch.distributed.tensor
from jaxtyping import Float
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from llamascopium.activation_functions import JumpReLU
from llamascopium.models.protocols import (
    ActiveSubspaceInitializable,
    DatasetNormStandardizable,
    EncoderBiasInitializable,
    EncoderInitializable,
    NormComputing,
    NormConstrainable,
)
from llamascopium.models.sparse_dictionary import (
    SparseDictionary,
    SparseDictionaryConfig,
    register_sae_config,
    register_sae_model,
)
from llamascopium.utils.distributed import DimMap


@register_sae_config("sae")
class SAEConfig(SparseDictionaryConfig):
    sae_type: str = "sae"
    hook_point_in: str
    hook_point_out: str
    use_glu_encoder: bool = False

    @property
    def associated_hook_points(self) -> list[str]:
        return [self.hook_point_in, self.hook_point_out]

    @property
    def hooks_in(self) -> list[str]:
        return [self.hook_point_in]

    @property
    def hooks_out(self) -> list[str]:
        return [self.hook_point_out]


@register_sae_model("sae")
class SparseAutoEncoder(
    SparseDictionary,
    NormComputing,
    NormConstrainable,
    DatasetNormStandardizable,
    EncoderInitializable,
    ActiveSubspaceInitializable,
    EncoderBiasInitializable,
):
    def __init__(self, cfg: SAEConfig, device_mesh: DeviceMesh | None = None):
        super(SparseAutoEncoder, self).__init__(cfg, device_mesh=device_mesh)
        self.cfg = cfg

        if device_mesh is None:
            self.W_E = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            self.b_E = nn.Parameter(torch.empty(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
            self.W_D = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model, device=cfg.device, dtype=cfg.dtype))
            if cfg.use_decoder_bias:
                self.b_D = nn.Parameter(torch.empty(cfg.d_model, device=cfg.device, dtype=cfg.dtype))

            if cfg.use_glu_encoder:
                self.W_E_glu = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
                self.b_E_glu = nn.Parameter(torch.empty(cfg.d_sae, device=cfg.device, dtype=cfg.dtype))
        else:
            self.W_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_model,
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_E"].placements(device_mesh),
                )
            )
            self.b_E = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_sae,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["b_E"].placements(device_mesh),
                )
            )
            self.W_D = nn.Parameter(
                torch.distributed.tensor.empty(
                    cfg.d_sae,
                    cfg.d_model,
                    dtype=cfg.dtype,
                    device_mesh=device_mesh,
                    placements=self.dim_maps()["W_D"].placements(device_mesh),
                )
            )
            if cfg.use_decoder_bias:
                self.b_D = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_model,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["b_D"].placements(device_mesh),
                    )
                )
            if cfg.use_glu_encoder:
                self.W_E_glu = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_model,
                        cfg.d_sae,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["W_E_glu"].placements(device_mesh),
                    )
                )
                self.b_E_glu = nn.Parameter(
                    torch.distributed.tensor.empty(
                        cfg.d_sae,
                        dtype=cfg.dtype,
                        device_mesh=device_mesh,
                        placements=self.dim_maps()["b_E_glu"].placements(device_mesh),
                    )
                )

        self.hook_hidden_pre = HookPoint()
        self.hook_feature_acts = HookPoint()
        self.hook_reconstructed = HookPoint()
        self.setup()

    def _postprocess_feature_acts(self, feature_acts: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply model-specific feature filtering before exposing activations to hooks."""
        return feature_acts

    @override
    def encoder_norm(self, keepdim: bool = False):
        """Compute the norm of the encoder weight."""
        if not isinstance(self.W_E, DTensor):
            return torch.norm(self.W_E, p=2, dim=0, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_E.to_local(), p=2, dim=0, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=DimMap({"model": 1 if keepdim else 0}).placements(self.device_mesh),
            )

    @override
    def decoder_norm(self, keepdim: bool = False) -> torch.Tensor:
        """Compute the norm of the decoder weight."""
        if not isinstance(self.W_D, DTensor):
            return torch.norm(self.W_D, p=2, dim=1, keepdim=keepdim).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.W_D.to_local(), p=2, dim=1, keepdim=keepdim),
                device_mesh=self.device_mesh,
                placements=DimMap({"model": 0}).placements(self.device_mesh),
            )

    @override
    def decoder_bias_norm(self) -> torch.Tensor:
        if not self.cfg.use_decoder_bias:
            raise ValueError("Decoder bias is not used")
        if not isinstance(self.b_D, DTensor):
            return torch.norm(self.b_D, p=2, dim=0, keepdim=True).to(self.cfg.device)
        else:
            assert self.device_mesh is not None
            return DTensor.from_local(
                torch.norm(self.b_D.to_local(), p=2, dim=0, keepdim=True),
                device_mesh=self.device_mesh,
                placements=DimMap({}).placements(self.device_mesh),
            )

    @override
    @torch.no_grad()
    def set_decoder_to_fixed_norm(self, value: float, force_exact: bool):
        """Set the decoder weights to a fixed norm."""
        if force_exact:
            self.W_D.mul_(value / self.decoder_norm(keepdim=True))
        else:
            self.W_D.mul_(value / torch.clamp(self.decoder_norm(keepdim=True), min=value))

    @torch.no_grad()
    def set_encoder_to_fixed_norm(self, value: float):
        """Set the encoder weights to a fixed norm."""
        self.W_E.mul_(value / self.encoder_norm(keepdim=True))

    def dim_maps(self) -> dict[str, DimMap]:
        """Return a dictionary mapping parameter names to dimension maps.

        Returns:
            A dictionary mapping parameter names to DimMap objects.
        """
        parent_maps = super().dim_maps()
        sae_maps = {
            "W_E": DimMap({"model": 1}),
            "W_D": DimMap({"model": 0}),
            "b_E": DimMap({"model": 0}),
        }
        if self.cfg.use_decoder_bias:
            sae_maps["b_D"] = DimMap({})
        if self.cfg.use_glu_encoder:
            sae_maps["W_E_glu"] = DimMap({"model": 1})
            sae_maps["b_E_glu"] = DimMap({"model": 0})
        return parent_maps | sae_maps

    @override
    @torch.no_grad()
    def transform_to_unit_decoder_norm(self):
        self.W_D.mul_(1 / self.decoder_norm(keepdim=True))

    @torch.no_grad()
    def standardize_parameters_of_dataset_norm(self):  # should be overridden by subclasses due to side effects
        """
        Standardize the parameters of the model to account for dataset_norm during inference.
        This function should be called during inference by the Initializer.

        During training, the activations correspond to an input `x` where the norm is sqrt(d_model).
        However, during inference, the norm of the input `x` corresponds to the dataset_norm.
        To ensure consistency between training and inference, the activations during inference
        are scaled by the factor:

            scaled_activation = training_activation * (dataset_norm / sqrt(d_model))

        Note:
            The model must have `dataset_average_activation_norm` set as an attribute,
            which is a dictionary where keys represent in or out and values
            specify the average activation norm of the dataset during inference.

            Example::

                dataset_average_activation_norm = {
                    self.cfg.hook_point_in: 1.0,
                    self.cfg.hook_point_out: 1.0,
                }

            Updates the internal parameters to reflect the standardized activations
            and changes the norm_activation to "inference" mode.
        """
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None
        input_norm_factor: float = (
            math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        )
        output_norm_factor: float = (
            math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_out]
        )
        self.b_E.div_(input_norm_factor)
        if self.cfg.use_decoder_bias:
            assert self.b_D is not None, "Decoder bias should exist if use_decoder_bias is True"
            self.b_D.div_(output_norm_factor)
        self.W_D.mul_(input_norm_factor / output_norm_factor)
        self.cfg.norm_activation = "inference"

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[False] = False,
        **kwargs,
    ) -> Union[Float[torch.Tensor, "batch d_sae"], Float[torch.Tensor, "batch seq_len d_sae"]]: ...

    @overload
    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: Literal[True],
        **kwargs,
    ) -> tuple[
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
    ]: ...

    def encode(
        self,
        x: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
        ],
        return_hidden_pre: bool = False,
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_sae"],
        Float[torch.Tensor, "batch seq_len d_sae"],
        tuple[
            Union[
                Float[torch.Tensor, "batch d_sae"],
                Float[torch.Tensor, "batch seq_len d_sae"],
            ],
            Union[
                Float[torch.Tensor, "batch d_sae"],
                Float[torch.Tensor, "batch seq_len d_sae"],
            ],
        ],
    ]:
        """Encode input tensor through the sparse autoencoder.

        Args:
            x: Input tensor of shape (batch, d_model) or (batch, seq_len, d_model)
            return_hidden_pre: If True, also return the pre-activation hidden states

        Returns:
            If return_hidden_pre is False:
                Feature activations tensor of shape (batch, d_sae) or (batch, seq_len, d_sae)
            If return_hidden_pre is True:
                Tuple of (feature_acts, hidden_pre) where both have shape (batch, d_sae) or (batch, seq_len, d_sae)
        """
        # Pass through encoder
        x = x.to(self.W_E.dtype)
        hidden_pre = x @ self.W_E + self.b_E

        # Apply GLU if configured
        if self.cfg.use_glu_encoder:
            hidden_pre_glu = torch.sigmoid(x @ self.W_E_glu + self.b_E_glu)
            hidden_pre = hidden_pre * hidden_pre_glu

        hidden_pre = self.hook_hidden_pre(hidden_pre)

        # Scale feature activations by decoder norm if configured
        if self.cfg.sparsity_include_decoder_norm:
            hidden_pre = hidden_pre * self.decoder_norm()

        feature_acts = self.activation_function(hidden_pre)

        if self.cfg.sparsity_include_decoder_norm:
            feature_acts = feature_acts / self.decoder_norm()
            hidden_pre = hidden_pre / self.decoder_norm()

        feature_acts = self._postprocess_feature_acts(feature_acts, **kwargs)

        if return_hidden_pre:
            return self.hook_feature_acts(feature_acts), hidden_pre
        return self.hook_feature_acts(feature_acts)

    def decode(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        **kwargs,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:  # may be overridden by subclasses
        max_l0_in_batch = feature_acts.gt(0).to(feature_acts).sum(dim=-1).max()
        sparsity_threshold = self.cfg.d_sae * (1 - self.cfg.sparsity_threshold_for_triton_spmm_kernel)
        if (
            self.cfg.use_triton_kernel and 0 < max_l0_in_batch < sparsity_threshold
        ):  # triton kernel cannot handle empty feature_acts
            from llamascopium.kernels import decode_with_triton_spmm_kernel

            reconstructed = decode_with_triton_spmm_kernel(feature_acts, self.W_D.T.contiguous())
        else:
            reconstructed = feature_acts @ self.W_D

        assert reconstructed is not None, "Reconstructed cannot be None"
        if self.cfg.use_decoder_bias:
            reconstructed = reconstructed + self.b_D
        reconstructed = self.hook_reconstructed(reconstructed)

        if isinstance(reconstructed, DTensor):
            reconstructed = DimMap({"data": 0}).redistribute(reconstructed)

        return reconstructed

    def decode_coo(
        self,
        feature_acts: Float[torch.sparse.Tensor, "seq_len d_sae"],
    ) -> Float[torch.Tensor, "seq_len d_model"]:
        """Decode feature activations back to model space using COO format."""
        reconstructed = feature_acts.to(torch.float32) @ self.W_D.to(torch.float32)
        if self.cfg.use_decoder_bias:
            reconstructed = reconstructed + self.b_D
        return reconstructed.to(self.cfg.dtype)

    @torch.no_grad()
    def _init_encoder_with_decoder_transpose(
        self, encoder: torch.nn.Linear, decoder: torch.nn.Linear, factor: float = 1.0
    ):
        encoder.weight.data = decoder.weight.data.T.clone().contiguous() * factor

    @override
    @torch.no_grad()
    def init_encoder_with_decoder_transpose(self, factor: float = 1.0):
        self.W_E.copy_(self.W_D.contiguous().T * factor)

    @override
    @torch.no_grad()
    def init_parameters(self, **kwargs):
        super().init_parameters(**kwargs)

        W_E = torch.empty(self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
            -kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"]
        )
        W_D = torch.empty(self.cfg.d_sae, self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype).uniform_(
            -kwargs["decoder_uniform_bound"], kwargs["decoder_uniform_bound"]
        )
        b_E = torch.zeros(self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype)

        if self.device_mesh is not None:
            W_E = self.dim_maps()["W_E"].distribute(W_E, self.device_mesh)
            W_D = self.dim_maps()["W_D"].distribute(W_D, self.device_mesh)
            b_E = self.dim_maps()["b_E"].distribute(b_E, self.device_mesh)

        self.W_E.copy_(W_E)
        self.W_D.copy_(W_D)
        self.b_E.copy_(b_E)

        if self.cfg.use_decoder_bias:
            b_D = torch.zeros(self.cfg.d_model, device=self.cfg.device, dtype=self.cfg.dtype)

            if self.device_mesh is not None:
                b_D = self.dim_maps()["b_D"].distribute(b_D, self.device_mesh)

            self.b_D.copy_(b_D)

        if self.cfg.use_glu_encoder:
            W_E_glu = torch.empty(
                self.cfg.d_model, self.cfg.d_sae, device=self.cfg.device, dtype=self.cfg.dtype
            ).uniform_(-kwargs["encoder_uniform_bound"], kwargs["encoder_uniform_bound"])
            if self.device_mesh is not None:
                W_E_glu = self.dim_maps()["W_E_glu"].distribute(W_E_glu, self.device_mesh)
            self.W_E_glu.copy_(W_E_glu)

    @override
    def prepare_input(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        x = batch[self.cfg.hook_point_in]
        return x, {}, {}

    @override
    def prepare_label(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        label = batch[self.cfg.hook_point_out]
        return label

    @override
    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def init_W_D_with_active_subspace(self, batch: dict[str, torch.Tensor], d_active_subspace: int):
        """Initialize W_D with the active subspace.

        Args:
            batch: The batch.
            d_active_subspace: The dimension of the active subspace.
        """
        label = self.prepare_label(batch)
        if self.device_mesh is not None:
            assert isinstance(label, DTensor)
            label = label.to_local()
            torch.distributed.broadcast(tensor=label, group=self.device_mesh.get_group("data"), group_src=0)
        demeaned_label = label - label.mean(dim=0)
        U, S, V = torch.svd(demeaned_label.T.to(torch.float32))
        proj_weight = U[:, :d_active_subspace]  # [d_model, d_active_subspace]
        self.W_D.copy_(self.W_D[:, :d_active_subspace] @ proj_weight.T.to(self.cfg.dtype))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def init_tc_with_mlp(self, batch: dict[str, torch.Tensor], mlp: CanBeUsedAsMLP):
        assert self.cfg.norm_activation == "dataset-wise"
        assert self.dataset_average_activation_norm is not None
        input_norm_factor = math.sqrt(self.cfg.d_model) / self.dataset_average_activation_norm[self.cfg.hook_point_in]
        x = self.prepare_input(batch)[0]

        if self.device_mesh is not None:
            assert isinstance(x, DTensor)
            assert isinstance(self.W_E, DTensor)
            x = x.to_local()
            torch.distributed.broadcast(tensor=x, group=self.device_mesh.get_group("data"), group_src=0)
            orig_x = x / input_norm_factor
            W_E_local = self.W_E.to_local()
            mlp_input_local = (
                orig_x.mean(dim=0)
                + (W_E_local / W_E_local.norm(dim=0, keepdim=True) * (orig_x - orig_x.mean(dim=0)).norm(dim=1).mean()).T
            )
            mlp_output_local = mlp.forward(mlp_input_local)
            mlp_output_demeaned_local = mlp_output_local - mlp_output_local.mean(dim=0)
            self.W_E.copy_(self.W_E / self.W_E.norm(dim=0, keepdim=True))
            W_D = DTensor.from_local(
                mlp_output_demeaned_local / mlp_output_demeaned_local.norm(dim=1, keepdim=True),
                device_mesh=self.device_mesh,
                placements=self.dim_maps()["W_D"].placements(self.device_mesh),
            )
            self.W_D.copy_(W_D)
        else:
            orig_x = x / input_norm_factor
            mlp_input = (
                orig_x.mean(dim=0)
                + (self.W_E / self.W_E.norm(dim=0, keepdim=True) * (orig_x - orig_x.mean(dim=0)).norm(dim=1).mean()).T
            )
            mlp_output = mlp.forward(mlp_input)
            mlp_output_demeaned = mlp_output - mlp_output.mean(dim=0)
            self.W_E.copy_(self.W_E / self.W_E.norm(dim=0, keepdim=True))
            self.W_D.copy_(mlp_output_demeaned / mlp_output_demeaned.norm(dim=1, keepdim=True))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def init_encoder_bias_with_mean_hidden_pre(self, batch: dict[str, torch.Tensor]):
        x = self.prepare_input(batch)[0]
        _, hidden_pre = self.encode(x, return_hidden_pre=True)

        self.b_E.sub_(hidden_pre.mean(dim=0))

    @classmethod
    @torch.no_grad()
    def from_saelens(cls, sae_saelens, **kwargs):
        from sae_lens import JumpReLUSAE, StandardSAE, TopKSAE

        # Check Configuration
        assert (
            isinstance(sae_saelens, JumpReLUSAE)
            or isinstance(sae_saelens, StandardSAE)
            or isinstance(sae_saelens, TopKSAE)
        ), f"Only support JumpReLUSAE, StandardSAE, TopKSAE, but get {type(sae_saelens)}"
        assert sae_saelens.cfg.reshape_activations == "none", (
            f"The 'reshape_activations' should be 'none' but get {sae_saelens.cfg.reshape_activations}."
        )
        assert not sae_saelens.cfg.apply_b_dec_to_input, (
            f"The 'apply_b_dec_to_input' should be 'False' but get {sae_saelens.cfg.apply_b_dec_to_input}."
        )
        assert sae_saelens.cfg.normalize_activations == "none", (
            f"The 'normalize_activations' should be 'false' but get {sae_saelens.cfg.normalize_activations}."
        )

        # Parse
        d_model = sae_saelens.cfg.d_in
        d_sae = sae_saelens.cfg.d_sae
        hook_name = sae_saelens.cfg.metadata.hook_name
        assert isinstance(hook_name, str)
        dtype = sae_saelens.W_enc.dtype

        rescale_acts_by_decoder_norm = False
        jumprelu_threshold_window = 0
        k = 0
        if isinstance(sae_saelens, StandardSAE):
            activation_fn = "relu"
        elif isinstance(sae_saelens, TopKSAE):
            activation_fn = "topk"
            k = sae_saelens.cfg.k
            rescale_acts_by_decoder_norm = sae_saelens.cfg.rescale_acts_by_decoder_norm
        elif isinstance(sae_saelens, JumpReLUSAE):
            activation_fn = "jumprelu"
            jumprelu_threshold_window = 2
        else:
            raise TypeError(f"Only support JumpReLUSAE, StandardSAE, TopKSAE, but get {type(sae_saelens)}")

        # create cfg
        cfg = SAEConfig(
            sae_type="sae",
            hook_point_in=hook_name,
            hook_point_out=hook_name,
            dtype=dtype,
            d_model=d_model,
            act_fn=activation_fn,
            jumprelu_threshold_window=jumprelu_threshold_window,
            top_k=k,
            expansion_factor=d_sae / d_model,
            sparsity_include_decoder_norm=rescale_acts_by_decoder_norm,
            **kwargs,
        )

        model = cls.from_config(cfg, None)

        model.W_D.copy_(sae_saelens.W_dec)
        model.W_E.copy_(sae_saelens.W_enc)
        model.b_D.copy_(sae_saelens.b_dec)
        model.b_E.copy_(sae_saelens.b_enc)

        if isinstance(sae_saelens, JumpReLUSAE):
            assert isinstance(model.activation_function, JumpReLU)
            model.activation_function.log_jumprelu_threshold.copy_(torch.log(sae_saelens.threshold.clone().detach()))

        return model

    @torch.no_grad()
    def to_saelens(self, model_name: str = "unknown"):
        from sae_lens import JumpReLUSAE, JumpReLUSAEConfig, StandardSAE, StandardSAEConfig, TopKSAE, TopKSAEConfig
        from sae_lens.saes.sae import SAEMetadata

        # Check env
        assert self.cfg.hook_point_in != self.cfg.hook_point_out, "Not support transcoder yet."
        assert not isinstance(self.b_D, DTensor), "Not support distributed setting yet."
        assert not self.cfg.use_glu_encoder, "Can't convert sae with use_glu_encoder=True to SAE Lens format."

        # Parse
        d_in = self.cfg.d_model
        d_sae = self.cfg.d_sae
        activation_fn = self.cfg.act_fn
        hook_name = self.cfg.hook_point_in
        dtype = self.cfg.dtype

        # Create model
        if activation_fn == "relu":
            cfg_saelens = StandardSAEConfig(
                d_in=d_in,
                d_sae=d_sae,
                dtype=str(dtype),
                device="cpu",
                apply_b_dec_to_input=False,
                normalize_activations="none",
                metadata=SAEMetadata(
                    model_name=model_name,
                    hook_name=hook_name,
                ),
            )
            model = StandardSAE(cfg_saelens)
        elif activation_fn == "jumprelu":
            cfg_saelens = JumpReLUSAEConfig(
                d_in=d_in,
                d_sae=d_sae,
                dtype=str(dtype),
                device="cpu",
                apply_b_dec_to_input=False,
                normalize_activations="none",
                metadata=SAEMetadata(
                    model_name=model_name,
                    hook_name=hook_name,
                ),
            )
            model = JumpReLUSAE(cfg_saelens)
        elif activation_fn == "topk":
            cfg_saelens = TopKSAEConfig(
                k=self.cfg.top_k,
                d_in=d_in,
                d_sae=d_sae,
                dtype=str(dtype),
                device="cpu",
                apply_b_dec_to_input=False,
                normalize_activations="none",
                rescale_acts_by_decoder_norm=self.cfg.sparsity_include_decoder_norm,
                metadata=SAEMetadata(
                    model_name=model_name,
                    hook_name=hook_name,
                ),
            )
            model = TopKSAE(cfg_saelens)
        else:
            raise TypeError("Not support such activation function yet.")

        # Depulicate weights
        model.W_dec.copy_(self.W_D)
        model.W_enc.copy_(self.W_E)
        model.b_dec.copy_(self.b_D)
        model.b_enc.copy_(self.b_E)

        if isinstance(model, JumpReLUSAE):
            assert isinstance(self.activation_function, JumpReLU)
            model.threshold.copy_(self.activation_function.log_jumprelu_threshold.exp())

        return model
