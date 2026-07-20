"""Matryoshka sparse autoencoder variant.

This module adds a single-device Matryoshka SAE implementation on top of the
standard SAE architecture. It reuses the existing encoder/decoder parameterization
and training pipeline while augmenting the loss with normalized prefix
reconstruction terms. AuxK always uses the standard global residual objective;
the legacy Matryoshka-specific AuxK flag is retained only for config compatibility.
"""

from typing import Any, Literal, Optional, Union, cast, overload

import torch
from jaxtyping import Float
from torch.distributed.tensor import DTensor
from typing_extensions import override

from llamascopium.models.sae import SAEConfig, SparseAutoEncoder
from llamascopium.models.sparse_dictionary import register_sae_config, register_sae_model
from llamascopium.utils.distributed.ops import item
from llamascopium.utils.tensor_specs import apply_token_mask


@register_sae_config("matryoshka_sae")
class MatryoshkaSAEConfig(SAEConfig):
    """Configuration for training a Matryoshka sparse autoencoder."""

    sae_type: str = "matryoshka_sae"
    matryoshka_widths: list[int]
    """Increasing latent widths whose prefix reconstructions are added to the training loss."""

    matryoshka_loss_weights: list[float] | None = None
    """Optional relative weights for each width in `matryoshka_widths`, including the full-width SAE.

    Effective weights are normalized to sum to 1.0 so Matryoshka reconstruction terms
    stay on the same scale regardless of the number of widths.
    """

    use_matryoshka_aux_loss: bool = False
    """Deprecated compatibility flag.

    AuxK always uses the standard global residual objective. This field is retained so
    older configs continue to load without modification.
    """

    def model_post_init(self, __context):
        super().model_post_init(__context)

        if not self.matryoshka_widths:
            raise ValueError("matryoshka_widths must contain at least one width.")

        normalized_widths = list(self.matryoshka_widths)
        if normalized_widths[-1] != self.d_sae:
            normalized_widths.append(self.d_sae)

        if any(width <= 0 for width in normalized_widths):
            raise ValueError("All matryoshka widths must be positive.")
        if any(prev_width >= width for prev_width, width in zip(normalized_widths[:-1], normalized_widths[1:])):
            raise ValueError("matryoshka_widths must be strictly increasing.")
        if normalized_widths[-1] != self.d_sae:
            raise ValueError("The final matryoshka width must equal d_sae.")

        normalized_weights = (
            [1.0] * len(normalized_widths)
            if self.matryoshka_loss_weights is None
            else list(self.matryoshka_loss_weights)
        )
        if len(normalized_weights) == len(normalized_widths) - 1 and normalized_widths[-1] == self.d_sae:
            normalized_weights.append(1.0)
        if len(normalized_weights) != len(normalized_widths):
            raise ValueError("matryoshka_loss_weights must match matryoshka_widths after full-width normalization.")
        if any(weight < 0.0 for weight in normalized_weights):
            raise ValueError("matryoshka_loss_weights must be non-negative.")
        total_weight = sum(normalized_weights)
        if total_weight <= 0.0:
            raise ValueError("matryoshka_loss_weights must sum to a positive value.")
        normalized_weights = [weight / total_weight for weight in normalized_weights]

        self.matryoshka_widths = normalized_widths
        self.matryoshka_loss_weights = normalized_weights


@register_sae_model("matryoshka_sae")
class MatryoshkaSparseAutoEncoder(SparseAutoEncoder):
    """Sparse autoencoder with Matryoshka prefix reconstruction losses.

    Training adds normalized reconstruction losses for multiple cumulative latent
    widths, while inference remains identical to a standard SAE with the same
    parameters. AuxK always follows the standard global residual formulation.
    """

    cfg: MatryoshkaSAEConfig

    def _single_device_only(self) -> None:
        if self.device_mesh is not None or isinstance(self.W_D, DTensor):
            raise NotImplementedError("Matryoshka SAE training currently only supports single-device execution.")

    def _matryoshka_loss_weights(self) -> list[float]:
        assert self.cfg.matryoshka_loss_weights is not None
        return list(self.cfg.matryoshka_loss_weights)

    def _postprocess_feature_acts(
        self,
        feature_acts: torch.Tensor,
        *,
        matryoshka_feature_range: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Expose only one configured prefix or segment during inference."""
        if matryoshka_feature_range is None or matryoshka_feature_range == (0, self.cfg.d_sae):
            return feature_acts
        start, end = matryoshka_feature_range
        boundaries = {0, *self.cfg.matryoshka_widths}
        if start not in boundaries or end not in boundaries or start >= end:
            raise ValueError(
                "matryoshka_feature_range boundaries must come from "
                f"{sorted(boundaries)} with start < end, got {matryoshka_feature_range}."
            )
        return torch.cat(
            [
                torch.zeros_like(feature_acts[..., :start]),
                feature_acts[..., start:end],
                torch.zeros_like(feature_acts[..., end:]),
            ],
            dim=-1,
        )

    @property
    def inner_matryoshka_widths(self) -> list[int]:
        return self.cfg.matryoshka_widths[:-1]

    @property
    def inner_matryoshka_loss_weights(self) -> list[float]:
        return self._matryoshka_loss_weights()[:-1]

    @property
    def full_matryoshka_loss_weight(self) -> float:
        return self._matryoshka_loss_weights()[-1]

    @property
    def matryoshka_segments(self) -> list[tuple[int, int]]:
        """Return non-overlapping latent segments induced by the Matryoshka widths."""
        segment_starts = [0, *self.cfg.matryoshka_widths[:-1]]
        return list(zip(segment_starts, self.cfg.matryoshka_widths))

    def decode_prefix(
        self,
        feature_acts: Union[
            Float[torch.Tensor, "batch d_sae"],
            Float[torch.Tensor, "batch seq_len d_sae"],
        ],
        width: int,
        *,
        apply_hook: bool = False,
    ) -> Union[
        Float[torch.Tensor, "batch d_model"],
        Float[torch.Tensor, "batch seq_len d_model"],
    ]:
        """Decode only the first `width` features of the latent vector."""
        self._single_device_only()
        if width <= 0 or width > self.cfg.d_sae:
            raise ValueError(f"width must be in [1, {self.cfg.d_sae}], got {width}.")

        reconstructed = feature_acts[..., :width] @ cast(torch.Tensor, self.W_D[:width])
        if self.cfg.use_decoder_bias:
            reconstructed = reconstructed + cast(torch.Tensor, self.b_D)
        if apply_hook:
            reconstructed = self.hook_reconstructed(reconstructed)
        return reconstructed

    def _compute_aux_feature_acts(
        self,
        hidden_pre: torch.Tensor,
        dead_mask: torch.Tensor,
        *,
        k_aux: int,
        decoder_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | None:
        current_k = self.current_k
        try:
            self.current_k = min(k_aux, int(item(dead_mask.sum())))
            if self.current_k == 0:
                return None

            dead_hidden_pre = hidden_pre * dead_mask
            if decoder_norm is not None:
                dead_hidden_pre = dead_hidden_pre * decoder_norm

            dead_feature_acts = self.activation_function(dead_hidden_pre)

            if decoder_norm is not None:
                dead_feature_acts = dead_feature_acts / decoder_norm

            return dead_feature_acts
        finally:
            self.current_k = current_k

    def _compute_standard_auxk_loss(
        self,
        *,
        label: torch.Tensor,
        reconstructed: torch.Tensor,
        hidden_pre: torch.Tensor,
        is_dead: torch.Tensor,
        mask: torch.Tensor | None,
        k_aux: int,
        auxk_coefficient: float,
    ) -> torch.Tensor:
        decoder_norm = self.decoder_norm() if self.cfg.sparsity_include_decoder_norm else None
        dead_feature_acts = self._compute_aux_feature_acts(
            hidden_pre,
            is_dead,
            k_aux=k_aux,
            decoder_norm=decoder_norm,
        )
        if dead_feature_acts is None:
            return reconstructed.new_tensor(0.0)

        aux_reconstructed = dead_feature_acts @ cast(torch.Tensor, self.W_D)
        residual = label - reconstructed
        l_aux = (residual - aux_reconstructed).pow(2).sum(dim=-1)
        l_aux, _ = apply_token_mask(l_aux, self.specs.loss(l_aux), mask, "mean")
        return auxk_coefficient * l_aux

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        auxk_coefficient: float = 0.0,
        k_aux: int = 512,
        update_dead_statistics: Any = None,
        return_aux_data: Literal[True] = True,
        **kwargs,
    ) -> dict[str, Any]: ...

    @overload
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        *,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        auxk_coefficient: float = 0.0,
        k_aux: int = 512,
        update_dead_statistics: Any = None,
        return_aux_data: Literal[False] = False,
        **kwargs,
    ) -> torch.Tensor: ...

    @override
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        label: Union[
            Float[torch.Tensor, "batch d_model"],
            Float[torch.Tensor, "batch seq_len d_model"],
            None,
        ] = None,
        *,
        sparsity_loss_type: Literal["power", "tanh", "tanh-quad", None] = None,
        tanh_stretch_coefficient: float = 4.0,
        frequency_scale: float = 0.01,
        p: int = 1,
        l1_coefficient: float = 1.0,
        lp_coefficient: float = 0.0,
        auxk_coefficient: float = 0.0,
        k_aux: int = 512,
        update_dead_statistics: Any = None,
        return_aux_data: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, dict[str, Any]]:
        self._single_device_only()

        ctx = cast(
            dict[str, Any],
            super().compute_loss(
                batch,
                label=label,
                sparsity_loss_type=sparsity_loss_type,
                tanh_stretch_coefficient=tanh_stretch_coefficient,
                frequency_scale=frequency_scale,
                p=p,
                l1_coefficient=l1_coefficient,
                lp_coefficient=lp_coefficient,
                auxk_coefficient=0.0,
                k_aux=k_aux,
                update_dead_statistics=None,
                return_aux_data=True,
                **kwargs,
            ),
        )

        feature_acts = cast(torch.Tensor, ctx["feature_acts"])
        reconstructed = cast(torch.Tensor, ctx["reconstructed"])
        hidden_pre = cast(torch.Tensor, ctx["hidden_pre"])
        label_tensor = cast(torch.Tensor, ctx["label"])
        mask = cast(torch.Tensor | None, ctx.get("mask"))

        full_loss_weight = self.full_matryoshka_loss_weight
        if full_loss_weight != 1.0:
            ctx["loss"] = ctx["loss"] + (full_loss_weight - 1.0) * ctx["l_rec"]

        inner_losses: dict[str, torch.Tensor] = {}
        weighted_inner_loss = reconstructed.new_tensor(0.0)
        for width, weight in zip(self.inner_matryoshka_widths, self.inner_matryoshka_loss_weights):
            if weight == 0.0:
                continue

            inner_reconstructed = self.decode_prefix(feature_acts, width)
            inner_l_rec = (inner_reconstructed - label_tensor).pow(2).sum(dim=-1)
            inner_l_rec, _ = apply_token_mask(inner_l_rec, self.specs.loss(inner_l_rec), mask, "mean")
            inner_losses[f"width_{width}"] = inner_l_rec
            weighted_inner_loss = weighted_inner_loss + weight * inner_l_rec

        ctx["loss"] = ctx["loss"] + weighted_inner_loss
        ctx["l_matryoshka"] = weighted_inner_loss
        ctx["matryoshka_inner_losses"] = inner_losses
        ctx["matryoshka_loss_weights"] = self._matryoshka_loss_weights()

        if auxk_coefficient > 0.0:
            assert update_dead_statistics is not None, "update_dead_statistics must be set when auxk_coefficient > 0.0"
            is_dead = update_dead_statistics(feature_acts, mask, self.specs.feature_acts(feature_acts))

            l_aux = self._compute_standard_auxk_loss(
                label=label_tensor,
                reconstructed=reconstructed,
                hidden_pre=hidden_pre,
                is_dead=is_dead,
                mask=mask,
                k_aux=k_aux,
                auxk_coefficient=auxk_coefficient,
            )

            ctx["l_aux"] = l_aux
            ctx["loss"] = ctx["loss"] + l_aux
        else:
            ctx["l_aux"] = None

        if return_aux_data:
            return ctx
        return cast(torch.Tensor, ctx["loss"])

    @override
    @torch.no_grad()
    def compute_training_metrics(
        self,
        *,
        matryoshka_inner_losses: dict[str, torch.Tensor] | None = None,
        l_matryoshka: torch.Tensor | None = None,
        feature_acts: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        n_tokens: int | None = None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute Matryoshka-specific training metrics.

        Prefix reconstruction losses keep their cumulative semantics:
        `width_8192` means reconstruction error using the first 8192 features.

        Segment metrics are non-cumulative and only summarize each newly added
        latent block, e.g. `segment_4096_8192_*` measures features in
        `[4096:8192)` rather than `[0:8192)`.
        """
        metrics: dict[str, float] = {
            "matryoshka_metrics/full_loss_weight": self.full_matryoshka_loss_weight,
        }

        if l_matryoshka is not None:
            metrics["matryoshka_metrics/inner_loss"] = item(l_matryoshka)

        if matryoshka_inner_losses is not None:
            for width_key, loss in matryoshka_inner_losses.items():
                metrics[f"matryoshka_metrics/{width_key}"] = item(loss)

        if feature_acts is not None and n_tokens is not None:
            specs = self.specs.feature_acts(feature_acts)
            for segment_start, segment_end in self.matryoshka_segments:
                segment_feature_acts = feature_acts[..., segment_start:segment_end]
                active_counts, _ = apply_token_mask((segment_feature_acts > 0).float(), specs, mask, "sum")
                total_acts, _ = apply_token_mask(segment_feature_acts, specs, mask, "sum")

                active_total = active_counts.sum()
                active_total_value = item(active_total)
                mean_feature_act = (
                    total_acts.sum() / active_total if active_total_value > 0 else segment_feature_acts.new_tensor(0.0)
                )
                mean_frequency = active_counts.mean() / n_tokens

                segment_name = f"segment_{segment_start}_{segment_end}"
                metrics[f"matryoshka_metrics/{segment_name}_mean_feature_act"] = item(mean_feature_act)
                metrics[f"matryoshka_metrics/{segment_name}_mean_frequency"] = item(mean_frequency)

        return metrics
