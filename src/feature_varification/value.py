from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .evaluator import collect_feature_activations
from .thresholds import resolve_thresholds
from .types import FeatureSpec, ThresholdSpec, VerificationCase, VerificationCounts


@dataclass(frozen=True)
class ValueOutcomeRule:
    """Classify positions by side-to-move WDL metadata."""

    outcome: Literal["win", "draw", "loss", "decisive"]
    probability_threshold: float = 0.6
    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(self, "name", f"value_outcome::{self.outcome}")

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        wdl = metadata.get("wdl") or {}
        win = float(wdl.get("current_player_win", 0.0))
        draw = float(wdl.get("draw", 0.0))
        loss = float(wdl.get("current_player_loss", 0.0))
        if self.outcome == "win":
            return win >= self.probability_threshold
        if self.outcome == "draw":
            return draw >= self.probability_threshold
        if self.outcome == "loss":
            return loss >= self.probability_threshold
        return max(win, loss) >= self.probability_threshold


def evaluate_feature_value_rule(
    *,
    model: Any,
    lorsas: Sequence[Any],
    transcoders: dict[int, Any] | dict[str, Any],
    feature: FeatureSpec,
    rule: ValueOutcomeRule,
    cases: Sequence[VerificationCase | str],
    aggregation: Literal["max", "mean"] = "max",
    threshold: ThresholdSpec | None = None,
    prepend_bos: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Compare aggregate feature activation in matching vs non-matching WDL positions."""

    normalized, activations = collect_feature_activations(
        model=model,
        lorsas=lorsas,
        transcoders=transcoders,
        feature=feature,
        cases=cases,
        prepend_bos=prepend_bos,
        show_progress=show_progress,
    )
    threshold = threshold or ThresholdSpec(mode="absolute", value=0.0, scope="dataset")
    thresholds = resolve_thresholds(activations, threshold)
    counts = VerificationCounts()
    positives: list[float] = []
    negatives: list[float] = []
    active_cases = 0
    for case, values, resolved_threshold in zip(normalized, activations, thresholds):
        score = float(values.max().item() if aggregation == "max" else values.mean().item())
        matches = rule.matches(case.metadata)
        active = score > float(resolved_threshold)
        if active:
            active_cases += 1
        if active and matches:
            counts.tp += 1
        elif active:
            counts.fp += 1
        elif matches:
            counts.fn += 1
        else:
            counts.tn += 1
        (positives if matches else negatives).append(score)
    positive_mean = sum(positives) / len(positives) if positives else None
    negative_mean = sum(negatives) / len(negatives) if negatives else None
    return {
        "feature": {
            "feature_type": feature.feature_type,
            "layer": feature.layer,
            "feature_id": feature.feature_id,
        },
        "rule_name": rule.name,
        "aggregation": aggregation,
        "threshold": {
            "mode": threshold.mode,
            "value": threshold.value,
            "scope": threshold.scope,
        },
        "counts": counts.to_dict(),
        "n_active_cases": active_cases,
        "n_positive_cases": len(positives),
        "n_negative_cases": len(negatives),
        "mean_activation_positive": positive_mean,
        "mean_activation_negative": negative_mean,
        "mean_difference": (
            positive_mean - negative_mean if positive_mean is not None and negative_mean is not None else None
        ),
    }
