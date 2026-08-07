from abc import ABC
from typing import Generic, TypeVar, cast

import torch
from torch import Tensor
from torch.types import Number
from transformer_lens import HookedTransformer

from llamascopium.models.lorsa import LowRankSparseAttention
from llamascopium.models.sae import SparseAutoEncoder
from llamascopium.models.sparse_dictionary import SparseDictionary
from llamascopium.optim import compute_grad_norm
from llamascopium.utils.distributed.ops import item
from llamascopium.utils.logging import get_distributed_logger
from llamascopium.utils.tensor_specs import apply_token_mask, reduce

logger = get_distributed_logger("metrics")


T = TypeVar("T", bound=Number | Tensor)


class Record(Generic[T]):
    def __init__(self, reduction: str = "mean"):
        self.value: T | None = None
        self.count = 0
        self.reduction = reduction

    def update(self, value: T, count: Tensor | int = 1):
        if isinstance(value, Tensor) and value.dtype in [torch.bfloat16, torch.float16]:
            value = cast(T, value.to(torch.float32))
        self.count += count
        if self.reduction in ["mean", "sum"]:
            self.value = cast(T, self.value + value) if self.value is not None else value
        elif self.reduction == "max":
            self.value = cast(T, max(self.value, value)) if self.value is not None else value  # type: ignore
        elif self.reduction == "min":
            self.value = cast(T, min(self.value, value)) if self.value is not None else value  # type: ignore
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def compute(self) -> T:
        assert self.value is not None, "Record must be updated before computing"
        if self.reduction == "mean":
            result = cast(T, self.value / self.count)
        elif self.reduction in ["max", "min", "sum"]:
            result = cast(T, self.value)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        self.value = None
        self.count = 0
        return result


class Metric(ABC):
    """Base class for all metrics."""

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update the metric with the given context.

        Args:
            ctx: Dictionary of context tensors.

        Returns:
            Update to the context.
        """
        return {}

    def compute(self) -> dict[str, Number]:
        """Compute the metric. This should also reset the metric's internal state.

        Returns:
            Dictionary of metric names to values."""
        raise NotImplementedError("Subclasses must implement this method")


class GradientNormMetric(Metric):
    def __init__(self, sae: SparseDictionary):
        self.sae = sae
        self.gradient_norm: Record[float] = Record("max")
        self.gradient_norm_before_clipping: Record[float] = Record("max")

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gradient_norm_before_clipping = ctx.get("grad_norm_before_clipping")
        assert gradient_norm_before_clipping is not None, "Gradient norm before clipping must be provided"
        self.gradient_norm_before_clipping.update(item(gradient_norm_before_clipping))
        gradient_norm = compute_grad_norm(self.sae.parameters(), self.sae.device_mesh)
        self.gradient_norm.update(item(gradient_norm))
        return {
            "gradient_norm_per_param": gradient_norm,
            "gradient_norm_before_clipping": gradient_norm_before_clipping,
        }

    def compute(self) -> dict[str, Number]:
        return {
            "metrics/gradient_norm_per_param": self.gradient_norm.compute(),
            "metrics/gradient_norm_before_clipping": self.gradient_norm_before_clipping.compute(),
        }


class FrequencyMetric(Metric):
    def __init__(self, sae: SparseDictionary):
        self.sae = sae
        self.act_freq_scores: Record[Tensor] = Record()
        self.specs: tuple[str, ...] | None = None

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_acts = ctx["feature_acts"]
        mask = ctx.get("mask")
        n_tokens = ctx["n_tokens"]
        act_freq_scores, specs = apply_token_mask(
            (feature_acts > 0).float(), self.sae.specs.feature_acts(feature_acts), mask, "sum"
        )
        self.act_freq_scores.update(act_freq_scores, n_tokens)
        self.specs = specs
        return {"act_freq_scores": act_freq_scores}

    def compute(self) -> dict[str, Number]:
        assert self.specs is not None, "Metrics must be updated before computing"
        feature_sparsity = self.act_freq_scores.compute()
        metrics = {
            "sparsity/above_1e-1": item((feature_sparsity > 1e-1).sum()),
            "sparsity/above_1e-2": item((feature_sparsity > 1e-2).sum()),
            "sparsity/below_1e-5": item((feature_sparsity < 1e-5).sum()),
            "sparsity/below_1e-6": item((feature_sparsity < 1e-6).sum()),
            "sparsity/below_1e-7": item((feature_sparsity < 1e-7).sum()),
        }
        if "layers" in self.specs:
            for l in range(feature_sparsity.size(self.specs.index("layers"))):
                metrics[f"sparsity/above_1e-1_layer{l}"] = item(
                    (feature_sparsity > 1e-1).select(self.specs.index("layers"), l).sum()
                )
                metrics[f"sparsity/above_1e-2_layer{l}"] = item(
                    (feature_sparsity > 1e-2).select(self.specs.index("layers"), l).sum()
                )
                metrics[f"sparsity/below_1e-5_layer{l}"] = item(
                    (feature_sparsity < 1e-5).select(self.specs.index("layers"), l).sum()
                )
                metrics[f"sparsity/below_1e-6_layer{l}"] = item(
                    (feature_sparsity < 1e-6).select(self.specs.index("layers"), l).sum()
                )
                metrics[f"sparsity/below_1e-7_layer{l}"] = item(
                    (feature_sparsity < 1e-7).select(self.specs.index("layers"), l).sum()
                )
        return metrics


class LossMetric(Metric):
    def __init__(self, sae: SparseDictionary):
        self.sae = sae
        self.loss: Record[Tensor] = Record()
        self.l_rec: Record[Tensor] = Record()
        self.l_s: Record[Tensor] = Record()
        self.l_p: Record[Tensor] = Record()
        self.l_aux: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        loss, l_rec, l_s, l_p, l_aux = ctx["loss"], ctx["l_rec"], ctx.get("l_s"), ctx.get("l_p"), ctx.get("l_aux")

        self.loss.update(loss)
        self.l_rec.update(l_rec.mean())
        if l_s is not None:
            self.l_s.update(l_s.mean())
        if l_p is not None:
            self.l_p.update(l_p.mean())
        if l_aux is not None:
            self.l_aux.update(l_aux.mean())
        return {}

    def compute(self) -> dict[str, Number]:
        metrics = {
            "losses/overall_loss": item(self.loss.compute()),
            "losses/mse_loss": item(self.l_rec.compute()),
        }
        if self.l_s.value is not None:
            metrics["losses/sparsity_loss"] = item(self.l_s.compute())
        if self.l_p.value is not None:
            metrics["losses/lp_loss"] = item(self.l_p.compute())
        if self.l_aux.value is not None:
            metrics["losses/aux_loss"] = item(self.l_aux.compute())
        return metrics


class MeanFeatureActMetric(Metric):
    def __init__(self, sae: SparseDictionary):
        self.sae = sae
        self.mean_feature_act: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_acts = ctx["feature_acts"]

        act_feature_counts = feature_acts.gt(0).float().sum()
        mean_feature_act = feature_acts.sum() / act_feature_counts

        self.mean_feature_act.update(mean_feature_act)
        return {}

    def compute(self) -> dict[str, Number]:
        return {"metrics/mean_feature_act": item(self.mean_feature_act.compute())}


class ExplainedVarianceMetric(Metric):
    def __init__(self, sae: SparseDictionary):
        self.sae = sae
        self.explained_variance: Record[Tensor] = Record()
        self.explained_variance_legacy: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        label, reconstructed, mask = ctx["label"], ctx["reconstructed"], ctx.get("mask")

        # The following computations assume the model dimension is the last dimension,
        # without further assumptions about the tensor specs.
        label_mean = apply_token_mask(
            label,
            self.sae.specs.label(label),
            mask,
            reduction="mean",
        )[0]  # shape: (d_model)
        per_token_l2_loss = (reconstructed - label).pow(2).sum(dim=-1)
        total_variance = (label - label_mean).pow(2).sum(dim=-1)
        explained_variance_legacy = apply_token_mask(
            1 - per_token_l2_loss / total_variance, self.sae.specs.label(label)[:-1], mask, reduction="mean"
        )[0]
        l2_loss_mean = apply_token_mask(
            per_token_l2_loss,
            self.sae.specs.label(label)[:-1],
            mask,
            reduction="mean",
        )[0]
        total_variance_mean = apply_token_mask(
            total_variance,
            self.sae.specs.label(label)[:-1],
            mask,
            reduction="mean",
        )[0]
        if torch.any(torch.isinf(total_variance_mean)):
            logger.warning("Some of total_variance_mean is inf. Check dtype or scaling.")
        explained_variance = 1 - l2_loss_mean / total_variance_mean
        self.explained_variance.update(explained_variance)
        self.explained_variance_legacy.update(explained_variance_legacy)
        return {
            "explained_variance": explained_variance,
            "explained_variance_legacy": explained_variance_legacy,
        }

    def compute(self) -> dict[str, Number]:
        return {
            "metrics/explained_variance": item(self.explained_variance.compute().mean()),
            "metrics/explained_variance_legacy": item(self.explained_variance_legacy.compute().mean()),
        }


class L0Metric(Metric):
    def __init__(self, sae: SparseDictionary):
        self.sae = sae
        self.l0: Record[Tensor] = Record()
        self.specs: tuple[str, ...] | None = None

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_acts = ctx["feature_acts"]
        mask = ctx.get("mask")
        n_tokens = ctx["n_tokens"]
        l0, l0_specs = apply_token_mask(
            (feature_acts > 0).float(),
            self.sae.specs.feature_acts(feature_acts),
            mask,
            "sum",
        )
        l0, l0_specs = reduce(l0, l0_specs, {"sae": "sum"})

        self.l0.update(l0, n_tokens)
        self.specs = l0_specs

        return {
            "l0": l0,
        }

    def compute(self) -> dict[str, Number]:
        assert self.specs is not None, "Metrics must be updated before computing"
        l0 = self.l0.compute()
        l0 = reduce(l0, self.specs, {"layers": "sum"})[0]
        return {
            "metrics/l0": item(l0.mean()),
            "metrics/l0_mean": item(l0.mean()),
            "metrics/l0_min": item(l0.min()),
            "metrics/l0_max": item(l0.max()),
        }


class L2NormErrorMetric(Metric):
    def __init__(self, sae: SparseDictionary):
        self.sae = sae
        self.l2_norm_error: Record[Tensor] = Record()
        self.l2_norm_error_ratio: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        label, reconstructed, mask = ctx["label"], ctx["reconstructed"], ctx.get("mask")
        l2_norm_error = apply_token_mask(
            (reconstructed - label).pow(2).sum(dim=-1).sqrt(), self.sae.specs.label(label)[:-1], mask, "mean"
        )[0]
        l2_norm_error_ratio = (
            l2_norm_error / apply_token_mask(label.norm(p=2, dim=-1), self.sae.specs.label(label)[:-1], mask, "mean")[0]
        )

        self.l2_norm_error.update(l2_norm_error)
        self.l2_norm_error_ratio.update(l2_norm_error_ratio)

        return {}

    def compute(self) -> dict[str, Number]:
        return {
            "metrics/l2_norm_error": item(self.l2_norm_error.compute()),
            "metrics/l2_norm_error_ratio": item(self.l2_norm_error_ratio.compute()),
        }


class ModelSpecificMetric(Metric):
    def __init__(self, sae: SparseDictionary):
        self.sae = sae
        self.count = 0
        self.metrics: dict[str, Record[float]] = {}

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        metrics = self.sae.compute_training_metrics(**ctx)
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = Record()
            self.metrics[k].update(v)
        return {}

    def compute(self) -> dict[str, Number]:
        return {k: v.compute() for k, v in self.metrics.items()}


class DownstreamMetric(Metric):
    def __init__(self, sae: SparseAutoEncoder | LowRankSparseAttention, model: HookedTransformer):
        self.sae = sae
        self.model = model
        self.downstream_loss_original: Record[Tensor] = Record()
        self.downstream_loss_reconstructed: Record[Tensor] = Record()
        self.downstream_loss_ablated: Record[Tensor] = Record()
        self.downstream_loss_delta_reconstructed: Record[Tensor] = Record()
        self.downstream_loss_delta_ablated: Record[Tensor] = Record()
        self.downstream_loss_recovered: Record[Tensor] = Record()
        self.downstream_loss_ratio: Record[Tensor] = Record()

    def update(self, ctx: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = ctx["tokens"].long()
        mask = ctx["mask"]

        assert tokens.ndim == 2, "Tokens must be a 2D tensor"
        specs = ("batch", "context")

        loss, cache = cast(
            tuple[torch.Tensor, dict[str, torch.Tensor]],
            self.model.run_with_cache(
                tokens,
                return_type="loss",
                loss_per_token=True,
                names_filter=[self.sae.cfg.hook_point_in, self.sae.cfg.hook_point_out],
                return_cache_object=False,
            ),
        )

        batch = {"tokens": tokens, "mask": mask, **cache}

        batch, scale_factors = self.sae.normalize_activations(batch, return_scale_factor=True)

        x, encoder_kwargs, decoder_kwargs = self.sae.prepare_input(batch)
        reconstructed = self.sae.forward(x, encoder_kwargs=encoder_kwargs, decoder_kwargs=decoder_kwargs)

        reconstructed_batch = self.sae.denormalize_activations(
            {self.sae.cfg.hook_point_out: reconstructed},
            scale_factors=scale_factors,
        )

        def replace_activations(
            activations: torch.Tensor,
            replacement: torch.Tensor,
        ) -> torch.Tensor:
            if mask is None:
                return replacement
            hook_mask = mask.to(device=activations.device, dtype=torch.bool)
            while hook_mask.ndim < activations.ndim:
                hook_mask = hook_mask.unsqueeze(-1)
            return torch.where(hook_mask, replacement, activations)

        def replace_hook_reconstructed(activations: torch.Tensor, hook: object | None = None) -> torch.Tensor:
            return replace_activations(activations, reconstructed_batch[self.sae.cfg.hook_point_out])

        reconstructed_loss: torch.Tensor = self.model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(self.sae.cfg.hook_point_out, replace_hook_reconstructed)],
            loss_per_token=True,
        )

        def replace_hook_ablated(activations: torch.Tensor, hook: object | None = None) -> torch.Tensor:
            return replace_activations(activations, torch.zeros_like(activations))

        ablated_loss: torch.Tensor = self.model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(self.sae.cfg.hook_point_out, replace_hook_ablated)],
            loss_per_token=True,
        )

        loss_mask = mask
        if loss_mask is not None and loss.ndim == 2 and loss_mask.ndim == 2:
            if loss_mask.shape[1] == loss.shape[1] + 1:
                loss_mask = loss_mask[:, 1:]

        loss = apply_token_mask(loss, specs, loss_mask, "mean")[0]
        reconstructed_loss = apply_token_mask(reconstructed_loss, specs, loss_mask, "mean")[0]
        ablated_loss = apply_token_mask(ablated_loss, specs, loss_mask, "mean")[0]

        delta_reconstructed = reconstructed_loss - loss
        delta_ablated = ablated_loss - loss
        eps = torch.tensor(1e-8, device=delta_ablated.device, dtype=delta_ablated.dtype)
        recovered = torch.where(
            delta_ablated.abs() > eps,
            (ablated_loss - reconstructed_loss) / delta_ablated,
            torch.full_like(delta_ablated, float("nan")),
        )
        ratio_denominator = ablated_loss - reconstructed_loss
        ratio = torch.where(
            ratio_denominator.abs() > eps,
            delta_ablated / ratio_denominator,
            torch.full_like(ratio_denominator, float("nan")),
        )

        self.downstream_loss_original.update(loss)
        self.downstream_loss_reconstructed.update(reconstructed_loss)
        self.downstream_loss_ablated.update(ablated_loss)
        self.downstream_loss_delta_reconstructed.update(delta_reconstructed)
        self.downstream_loss_delta_ablated.update(delta_ablated)
        self.downstream_loss_recovered.update(recovered)
        self.downstream_loss_ratio.update(ratio)

        return {}

    def compute(self) -> dict[str, Number]:
        return {
            "metrics/downstream_loss_original": item(self.downstream_loss_original.compute()),
            "metrics/downstream_loss_reconstructed": item(self.downstream_loss_reconstructed.compute()),
            "metrics/downstream_loss_ablated": item(self.downstream_loss_ablated.compute()),
            "metrics/downstream_loss_delta_reconstructed": item(self.downstream_loss_delta_reconstructed.compute()),
            "metrics/downstream_loss_delta_ablated": item(self.downstream_loss_delta_ablated.compute()),
            "metrics/downstream_loss_recovered": item(self.downstream_loss_recovered.compute()),
            "metrics/downstream_loss_ratio": item(self.downstream_loss_ratio.compute()),
        }
