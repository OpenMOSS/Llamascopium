from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

FeatureType = Literal["transcoder", "lorsa"]


@dataclass(frozen=True)
class FeatureSpec:
    """Identify a single SAE feature.

    Attributes
    ----------
    feature_type:
        Either ``"transcoder"`` or ``"lorsa"``.
    layer:
        Transformer layer index.
    feature_id:
        Column index inside the SAE activation tensor.
    """

    feature_type: FeatureType
    layer: int
    feature_id: int


@dataclass(frozen=True)
class VerificationCase:
    """One evaluation sample used for feature verification.

    Attributes
    ----------
    fen:
        Board state in FEN format.
    move_uci:
        Optional supervising move. Rules such as ``move_start_square`` and
        ``move_end_square`` need this field.
    label:
        Optional human-readable identifier. This is only used in debug output
        and stored example rows.
    metadata:
        Optional free-form payload carried through to result examples. This is
        useful when a notebook wants to remember where a case came from
        (dataset split, source PGN, engine score, etc.).
    """

    fen: str
    move_uci: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdSpec:
    """Describe how to convert real-valued activations into a binary active/inactive mask.

    Parameters
    ----------
    mode:
        Thresholding strategy.

        - ``"absolute"``: compare activations against a fixed scalar.
        - ``"ratio_to_max"``: threshold = ``value * max_activation``.
        - ``"percentile"``: threshold = percentile of observed activations.
    value:
        The numeric parameter used by ``mode``.
        For ``"absolute"`` this is the threshold itself.
        For ``"ratio_to_max"`` this is usually between ``0`` and ``1``.
        For ``"percentile"`` this is usually between ``0`` and ``100``.
    scope:
        Whether ``max_activation`` / percentile should be computed
        per-sample or over the whole evaluation dataset.
    """

    mode: Literal["absolute", "ratio_to_max", "percentile"] = "absolute"
    value: float = 0.0
    scope: Literal["sample", "dataset"] = "dataset"


@dataclass(frozen=True)
class RuleEvaluation:
    """Boolean supervision mask for one FEN.

    ``mask`` should have shape ``[64]`` and be aligned with BT4 board positions.
    ``metadata`` is optional rule-specific context that callers may want to inspect
    later when debugging false positives / false negatives.
    """

    mask: list[bool]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationCounts:
    """Aggregate confusion-matrix statistics across board positions."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "total": self.total,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class FeatureVerificationResult:
    """Structured output of a feature-vs-rule evaluation run."""

    feature: FeatureSpec
    rule_name: str
    threshold: ThresholdSpec
    counts: VerificationCounts
    n_fens_evaluated: int
    active_positions: int
    total_positions: int
    max_activation_on_target: float | None
    max_activation_off_target: float | None
    mean_activation_on_target: float | None
    mean_activation_off_target: float | None
    example_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": asdict(self.feature),
            "rule_name": self.rule_name,
            "threshold": asdict(self.threshold),
            "counts": self.counts.to_dict(),
            "n_fens_evaluated": self.n_fens_evaluated,
            "active_positions": self.active_positions,
            "total_positions": self.total_positions,
            "max_activation_on_target": self.max_activation_on_target,
            "max_activation_off_target": self.max_activation_off_target,
            "mean_activation_on_target": self.mean_activation_on_target,
            "mean_activation_off_target": self.mean_activation_off_target,
            "example_rows": self.example_rows,
        }
