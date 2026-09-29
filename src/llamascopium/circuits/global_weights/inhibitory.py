"""Observational suppression statistics constrained by negative virtual weights."""

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from llamascopium.backend.language_model import TransformerLensLanguageModel
from llamascopium.circuits.global_weights.weights import FeatureSpec, _feature_specs
from llamascopium.circuits.hooks import apply_saes
from llamascopium.models.lorsa import LowRankSparseAttention
from llamascopium.models.sae import SparseAutoEncoder

Feature = tuple[str, int]


@dataclass(frozen=True)
class InhibitoryConnection:
    source: str
    source_feature: int
    target: str
    target_feature: int
    score: float
    normalized_inhibitory_score: float
    delta: float
    virtual_weight: float
    mu_on: float
    mu_off: float
    kind: Literal["inhibitory"] = "inhibitory"


@dataclass(frozen=True)
class InhibitoryStatistics:
    virtual_weight: torch.Tensor
    mu_on: torch.Tensor
    mu_off: torch.Tensor
    delta: torch.Tensor
    inhibitory_score: torch.Tensor
    normalized_inhibitory_score: torch.Tensor


def _positive_sparse_mask(activations: torch.Tensor) -> torch.Tensor:
    """Keep only positive post-sparsification entries, without an inactive mask."""
    sparse = activations.coalesce() if activations.layout == torch.sparse_coo else activations.to_sparse().coalesce()
    positive = sparse.values() > 0
    indices = sparse.indices()[:, positive]
    values = torch.ones_like(sparse.values()[positive], dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, sparse.shape, device=activations.device).coalesce()


class InhibitoryWeights:
    """Stream conditional pre-TopK means for upstream feature pairs.

    The contrast is observational. A positive score is evidence of co-occurring
    suppression under a negative virtual weight, not a causal effect.
    """

    def __init__(self, specs: list[FeatureSpec], targets: list[Feature]):
        if not specs or not targets:
            raise ValueError("At least one dictionary and one target feature are required")
        self.specs = {spec.name: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("Dictionary hook_point_out values must be unique")
        self.targets = list(dict.fromkeys(targets))
        self.virtual_weights: dict[Feature, dict[str, torch.Tensor]] = {}
        self.s_on: dict[Feature, dict[str, torch.Tensor]] = {}
        for target in self.targets:
            name, index = target
            if name not in self.specs or not 0 <= index < self.specs[name].encoder.shape[1]:
                raise ValueError(f"Unknown target feature: {target}")
            target_spec = self.specs[name]
            encoder = target_spec.encoder[:, index].float()
            self.virtual_weights[target] = {}
            self.s_on[target] = {}
            for source in specs:
                if source.order >= target_spec.order:
                    continue
                decoder = source.decoder.float()
                if target_spec.center_decoder:
                    decoder = decoder - decoder.mean(dim=-1, keepdim=True)
                self.virtual_weights[target][source.name] = decoder @ encoder  # [source_feature]
                self.s_on[target][source.name] = torch.zeros(
                    source.decoder.shape[0], device=decoder.device, dtype=torch.float32
                )
        source_names = {name for sums in self.s_on.values() for name in sums}
        self.n_on = {
            name: torch.zeros(spec.decoder.shape[0], device=spec.decoder.device, dtype=torch.int64)
            for name, spec in self.specs.items()
            if name in source_names
        }
        self.s_total = {target: torch.zeros((), device=self.specs[target[0]].encoder.device) for target in self.targets}
        self.abs_pre_total = {target: torch.zeros_like(total) for target, total in self.s_total.items()}
        self.n_total = 0
        self.num_samples = 0

    @torch.no_grad()
    def update(
        self,
        source_activations: dict[str, torch.Tensor],
        target_pre_activations: dict[Feature, torch.Tensor],
        scales: dict[str, torch.Tensor],
        patterns: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Add one prompt; target pre activations have shape [position]."""
        patterns = patterns or {}
        positions = next(iter(target_pre_activations.values())).numel()
        routed = []
        for target in self.targets:
            spec = self.specs[target[0]]
            pre = target_pre_activations[target].float()
            if pre.shape != (positions,):
                raise ValueError(f"Invalid pre-TopK target shape for {target}: {tuple(pre.shape)}")
            if spec.is_lorsa:
                pattern = patterns[spec.name]
                head = target[1] // spec.ov_group_size
                if pattern.ndim != 3 or head >= pattern.shape[0] or pattern.shape[1:] != (positions, positions):
                    raise ValueError(f"Invalid attention pattern for {spec.name}: {tuple(pattern.shape)}")
                drive = torch.einsum("q,qk->k", pre, pattern[head].float())
            else:
                drive = pre
            scale = scales[spec.scale_hook].float().reshape(-1)
            if scale.shape != (positions,) or torch.any(scale <= 0):
                raise ValueError(f"Invalid normalization scale for {spec.name}")
            drive = drive / scale
            routed.append(drive)
            self.s_total[target] += drive.sum()
            self.abs_pre_total[target] += pre.abs().sum()

        # [position, target] is dense; each source mask stays sparse [position, feature].
        routed_matrix = torch.stack(routed, dim=1)
        for name, count in self.n_on.items():
            acts = source_activations[name]
            if acts.shape != (positions, count.numel()):
                raise ValueError(f"Invalid source activation shape for {name}: {tuple(acts.shape)}")
            mask = _positive_sparse_mask(acts)
            count += torch.bincount(mask.indices()[1], minlength=count.numel())
            # [feature, position] sparse @ [position, target] dense -> [feature, target].
            sums = torch.sparse.mm(mask.transpose(0, 1), routed_matrix)
            for column, target in enumerate(self.targets):
                if name in self.s_on[target]:
                    self.s_on[target][name] += sums[:, column]
        self.n_total += positions
        self.num_samples += 1

    def statistics(self, target: Feature, source: str, eps: float = 1e-8) -> InhibitoryStatistics:
        """Return feature-wise raw conditional statistics for one source dictionary."""
        if eps <= 0:
            raise ValueError("eps must be positive")
        s_on = self.s_on[target][source]
        n_on = self.n_on[source]
        n_off = self.n_total - n_on
        valid = (n_on > 0) & (n_off > 0)
        mu_on = torch.where(valid, s_on / n_on.clamp_min(1), 0)
        mu_off = torch.where(valid, (self.s_total[target] - s_on) / n_off.clamp_min(1), 0)
        delta = mu_off - mu_on
        virtual_weight = self.virtual_weights[target][source]
        score = delta.clamp_min(0) * (-virtual_weight).clamp_min(0)
        denominator = self.abs_pre_total[target] / max(self.n_total, 1) + eps
        return InhibitoryStatistics(virtual_weight, mu_on, mu_off, delta, score, score / denominator)

    def topk(
        self, target: Feature, k: int = 10, normalized: bool = False, eps: float = 1e-8
    ) -> list[InhibitoryConnection]:
        """Rank upstream neighbors by non-negative observational inhibitory score."""
        if k < 0 or eps <= 0:
            raise ValueError("k must be nonnegative and eps must be positive")
        if k == 0 or self.n_total == 0:
            return []
        edges = []
        for name in self.s_on[target]:
            stats = self.statistics(target, name, eps)
            priority = stats.normalized_inhibitory_score if normalized else stats.inhibitory_score
            for index in priority.topk(min(k, priority.numel())).indices.tolist():
                if priority[index] <= 0:
                    continue
                edges.append(
                    InhibitoryConnection(
                        source=name,
                        source_feature=index,
                        target=target[0],
                        target_feature=target[1],
                        score=stats.inhibitory_score[index].item(),
                        normalized_inhibitory_score=stats.normalized_inhibitory_score[index].item(),
                        delta=stats.delta[index].item(),
                        virtual_weight=stats.virtual_weight[index].item(),
                        mu_on=stats.mu_on[index].item(),
                        mu_off=stats.mu_off[index].item(),
                    )
                )
        key = (lambda edge: edge.normalized_inhibitory_score) if normalized else (lambda edge: edge.score)
        return sorted(edges, key=key, reverse=True)[:k]


@torch.no_grad()
def compute_inhibitory_weights(
    model: TransformerLensLanguageModel,
    dictionaries: list[SparseAutoEncoder | LowRankSparseAttention],
    inputs: Iterable[str | torch.Tensor],
    targets: list[Feature],
) -> InhibitoryWeights:
    """Trace upstream observational suppression around selected features.

    Downstream inhibitory search requires pre-TopK activation for candidate
    target features and is intentionally a separate future implementation.
    """
    result = InhibitoryWeights(_feature_specs(model, dictionaries), targets)
    target_indices: dict[str, list[int]] = {}
    for name, index in result.targets:
        target_indices.setdefault(name, []).append(index)

    for prompt in inputs:
        source_activations: dict[str, torch.Tensor] = {}
        target_pre: dict[Feature, torch.Tensor] = {}
        scales: dict[str, torch.Tensor] = {}
        patterns: dict[str, torch.Tensor] = {}

        def capture(destination: dict[str, torch.Tensor], name: str):
            def hook(value: torch.Tensor, hook):
                if value.shape[0] != 1:
                    raise ValueError("Inhibitory weights require one prompt per forward pass")
                destination[name] = value.detach().squeeze(0)
                return value

            return hook

        def capture_pre(name: str):
            def hook(value: torch.Tensor, hook):
                if value.shape[0] != 1:
                    raise ValueError("Inhibitory weights require one prompt per forward pass")
                for index in target_indices[name]:
                    target_pre[name, index] = value[0, :, index].detach()
                return value

            return hook

        def capture_source(name: str):
            def hook(value: torch.Tensor, hook):
                if value.shape[0] != 1:
                    raise ValueError("Inhibitory weights require one prompt per forward pass")
                source_activations[name] = _positive_sparse_mask(value.detach().squeeze(0))
                return value

            return hook

        hooks = [
            (spec.scale_hook, capture(scales, spec.scale_hook))
            for spec in result.specs.values()
            if spec.name in target_indices
        ]
        for name in result.n_on:
            hooks.append((f"{name}.sae.hook_feature_acts", capture_source(name)))
        for name in target_indices:
            hooks.append((f"{name}.sae.hook_hidden_pre", capture_pre(name)))
            if result.specs[name].is_lorsa:
                hooks.append((f"{name}.sae.hook_attn_pattern", capture(patterns, name)))
        with apply_saes(model, dictionaries), model.hooks(fwd_hooks=hooks):
            model(prompt)
        result.update(source_activations, target_pre, scales, patterns)
    return result
