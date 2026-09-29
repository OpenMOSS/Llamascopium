"""Dataset-level feature connections from virtual weights and activations."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.distributed.tensor import DTensor

from llamascopium.backend.language_model import TransformerLensLanguageModel
from llamascopium.circuits.hooks import apply_saes
from llamascopium.models.lorsa import LowRankSparseAttention
from llamascopium.models.sae import SparseAutoEncoder


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    order: tuple[int, int]
    encoder: torch.Tensor  # d_model, d_sae; includes LN gain when applicable
    decoder: torch.Tensor  # d_sae, d_model
    scale_hook: str
    is_lorsa: bool = False
    ov_group_size: int = 1
    center_decoder: bool = False


@dataclass(frozen=True)
class GlobalConnection:
    source: str
    source_feature: int
    target: str
    target_feature: int
    weight: float


class GlobalWeights:
    """Accumulate signed global weights around selected features over prompts.

    Each edge uses the source decoder / target encoder virtual weight. The
    activation co-occurrence is transported through the target Lorsa attention
    pattern (or the identity for an MLP transcoder) and divided by the target
    input normalization scale. Incoming weights divide by the upstream
    feature's total activation; outgoing weights divide by the downstream
    feature's total activation. Only earlier sublayers can be sources.
    """

    def __init__(self, specs: list[FeatureSpec], targets: list[tuple[str, int]]):
        if not specs or not targets:
            raise ValueError("At least one dictionary and one target feature are required")
        self.specs = {spec.name: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("Dictionary hook_point_out values must be unique")
        self.targets = list(dict.fromkeys(targets))
        self.activation_sum = {
            spec.name: torch.zeros(spec.decoder.shape[0], device=spec.decoder.device, dtype=torch.float32)
            for spec in specs
        }
        self.numerator: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.downstream_numerator: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        for name, feature in self.targets:
            if name not in self.specs or not 0 <= feature < self.specs[name].encoder.shape[1]:
                raise ValueError(f"Unknown target feature: {(name, feature)}")
            target = self.specs[name]
            self.numerator[name, feature] = {
                source.name: torch.zeros_like(self.activation_sum[source.name])
                for source in specs
                if source.order < target.order
            }
            self.downstream_numerator[name, feature] = {
                later.name: torch.zeros_like(self.activation_sum[later.name])
                for later in specs
                if later.order > target.order
            }
        self.num_samples = 0

    @torch.no_grad()
    def update(
        self,
        activations: dict[str, torch.Tensor],
        scales: dict[str, torch.Tensor],
        patterns: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Add one prompt. Activations are [position, feature]."""
        patterns = patterns or {}
        positions: int | None = None
        for name, spec in self.specs.items():
            acts = activations[name]
            if acts.ndim != 2 or acts.shape[1] != spec.decoder.shape[0]:
                raise ValueError(f"Invalid activation shape for {name}: {tuple(acts.shape)}")
            if positions is not None and acts.shape[0] != positions:
                raise ValueError("All dictionaries must use the same token positions")
            positions = acts.shape[0]
            self.activation_sum[name] += acts.float().sum(0)

        for (name, feature), source_sums in self.numerator.items():
            if not source_sums:
                continue
            target = self.specs[name]
            target_acts = activations[name][:, feature].float()
            if not torch.any(target_acts):
                continue
            if target.is_lorsa:
                pattern = patterns[name]
                head = feature // target.ov_group_size
                if pattern.ndim != 3 or head >= pattern.shape[0] or pattern.shape[1:] != (positions, positions):
                    raise ValueError(f"Invalid attention pattern for {name}: {tuple(pattern.shape)}")
                transported = torch.einsum("q,qk->k", target_acts, pattern[head].float())
            else:
                transported = target_acts
            scale = scales[target.scale_hook].float().reshape(-1)
            if scale.shape[0] != positions or torch.any(scale <= 0):
                raise ValueError(f"Invalid normalization scale for {name}")
            transported = transported / scale
            encoder = target.encoder[:, feature].float()
            for source_name, numerator in source_sums.items():
                source = self.specs[source_name]
                decoder = source.decoder.float()
                if target.center_decoder:
                    decoder = decoder - decoder.mean(dim=-1, keepdim=True)
                virtual_weight = decoder @ encoder
                coactivation = transported @ activations[source_name].float()
                numerator += coactivation * virtual_weight

        for (name, feature), target_sums in self.downstream_numerator.items():
            if not target_sums:
                continue
            source = self.specs[name]
            source_acts = activations[name][:, feature].float()
            if not torch.any(source_acts):
                continue
            decoder = source.decoder[feature].float()
            for target_name, numerator in target_sums.items():
                target = self.specs[target_name]
                target_acts = activations[target_name].float()
                if target.is_lorsa:
                    pattern = patterns[target_name].float()
                    n_heads = target_acts.shape[1] // target.ov_group_size
                    if pattern.shape != (n_heads, positions, positions):
                        raise ValueError(f"Invalid attention pattern for {target_name}: {tuple(pattern.shape)}")
                    grouped = target_acts.reshape(positions, n_heads, target.ov_group_size)
                    transported = torch.einsum("qhg,hqk->khg", grouped, pattern).reshape(positions, -1)
                else:
                    transported = target_acts
                scale = scales[target.scale_hook].float().reshape(-1)
                if scale.shape[0] != positions or torch.any(scale <= 0):
                    raise ValueError(f"Invalid normalization scale for {target_name}")
                coactivation = (source_acts / scale) @ transported
                effective_decoder = decoder - decoder.mean() if target.center_decoder else decoder
                virtual_weight = effective_decoder @ target.encoder.float()
                numerator += coactivation * virtual_weight
        self.num_samples += 1

    def _topk(
        self, selected: tuple[str, int], numerators: dict[str, torch.Tensor], k: int, downstream: bool
    ) -> list[GlobalConnection]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        if k == 0:
            return []
        edges: list[GlobalConnection] = []
        for candidate_name, numerator in numerators.items():
            denominator = self.activation_sum[candidate_name]
            weights = torch.where(denominator != 0, numerator / denominator.clamp_min(1e-30), 0)
            indices = weights.abs().topk(min(k, weights.numel())).indices.tolist()
            for index in indices:
                if weights[index] == 0:
                    continue
                if downstream:
                    edges.append(
                        GlobalConnection(selected[0], selected[1], candidate_name, index, weights[index].item())
                    )
                else:
                    edges.append(
                        GlobalConnection(candidate_name, index, selected[0], selected[1], weights[index].item())
                    )
        return sorted(edges, key=lambda edge: abs(edge.weight), reverse=True)[:k]

    def topk(self, target: tuple[str, int], k: int = 10) -> list[GlobalConnection]:
        """Return strongest incoming edges, ranked by absolute weight."""
        return self._topk(target, self.numerator[target], k, downstream=False)

    def downstream(self, source: tuple[str, int], k: int = 10) -> list[GlobalConnection]:
        """Return strongest outgoing edges, normalized by downstream activity."""
        return self._topk(source, self.downstream_numerator[source], k, downstream=True)


def _feature_specs(
    model: TransformerLensLanguageModel, dictionaries: list[SparseAutoEncoder | LowRankSparseAttention]
) -> list[FeatureSpec]:
    if model.device_mesh is not None or any(isinstance(sae.W_E, DTensor) for sae in dictionaries):
        raise ValueError("Global weights currently require a non-distributed model and dictionaries")
    if model.cfg.parallel_attn_mlp or model.cfg.normalization_type not in ("LN", "RMS"):
        raise ValueError("Global weights require sequential attention/MLP blocks with LN or RMS normalization")
    specs = []
    for sae in dictionaries:
        hook_in, hook_out = sae.cfg.hook_point_in, sae.cfg.hook_point_out
        parts = hook_out.split(".")
        if len(parts) != 3 or parts[0] != "blocks" or not parts[1].isdigit():
            raise ValueError(f"Unsupported dictionary output hook: {hook_out}")
        layer = int(parts[1])
        is_lorsa = isinstance(sae, LowRankSparseAttention)
        sublayer = 0 if is_lorsa else 1
        expected_in = f"blocks.{layer}.ln{1 if is_lorsa else 2}.hook_normalized"
        expected_out = f"blocks.{layer}.hook_{'attn' if is_lorsa else 'mlp'}_out"
        if hook_in != expected_in or hook_out != expected_out:
            raise ValueError(f"Unsupported dictionary hook pair: {hook_in}, {hook_out}")
        norm = getattr(model.blocks[layer], f"ln{1 if is_lorsa else 2}")
        encoder = sae.W_E.detach().float()
        if model.cfg.normalization_type == "LN":
            encoder = encoder * norm.w.detach().float()[:, None]
        specs.append(
            FeatureSpec(
                name=hook_out,
                order=(layer, sublayer),
                encoder=encoder,
                decoder=sae.W_D.detach(),
                scale_hook=hook_in.replace("hook_normalized", "hook_scale"),
                is_lorsa=is_lorsa,
                ov_group_size=sae.cfg.ov_group_size if is_lorsa else 1,
                center_decoder=model.cfg.normalization_type == "LN",
            )
        )
    return specs


@torch.no_grad()
def compute_global_weights(
    model: TransformerLensLanguageModel,
    dictionaries: list[SparseAutoEncoder | LowRankSparseAttention],
    inputs: Iterable[str | torch.Tensor],
    targets: list[tuple[str, int]],
) -> GlobalWeights:
    """Trace dataset-level connections into selected features.

    Target names are dictionary output hooks (for example
    ``blocks.3.hook_mlp_out``). Each input must be a single prompt. Use
    ``result.topk(target)`` for upstream connections and
    ``result.downstream(target)`` for downstream connections. The model must
    use sequential attention/MLP blocks with LN or RMS normalization.
    """
    specs = _feature_specs(model, dictionaries)
    result = GlobalWeights(specs, targets)
    for prompt in inputs:
        activations: dict[str, torch.Tensor] = {}
        scales: dict[str, torch.Tensor] = {}
        patterns: dict[str, torch.Tensor] = {}

        def capture(destination: dict[str, torch.Tensor], name: str):
            def hook(value: torch.Tensor, hook):
                if value.shape[0] != 1:
                    raise ValueError("Global weights require one prompt per forward pass")
                destination[name] = value.detach().squeeze(0)
                return value

            return hook

        hooks = [(spec.scale_hook, capture(scales, spec.scale_hook)) for spec in specs]
        for spec in specs:
            hooks.append((f"{spec.name}.sae.hook_feature_acts", capture(activations, spec.name)))
            if spec.is_lorsa:
                hooks.append((f"{spec.name}.sae.hook_attn_pattern", capture(patterns, spec.name)))
        with apply_saes(model, dictionaries), model.hooks(fwd_hooks=hooks):
            model(prompt)
        result.update(activations, scales, patterns)
    return result
