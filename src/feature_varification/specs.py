from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .rules import (
    AllOfRule,
    AnyOfRule,
    CapturablePieceRule,
    CheckingMoveDestinationRule,
    EmptySquareRule,
    KingNeighborhoodRule,
    MoveEndPieceRule,
    MoveEndSquareRule,
    MoveStartPieceRule,
    MoveStartSquareRule,
    OccupiedSquareRule,
    PieceAttackRule,
    PieceDestinationRule,
    PieceFrontSpanRule,
    PieceMultiHopDestinationRule,
    PieceNeighborhoodRule,
    PieceRayRule,
    PieceTypeRule,
    ProtectedPieceRule,
    QueenCheckAroundOpponentKingRule,
    RelativeBoardRegionRule,
    RelativeOffsetRule,
    VerificationRule,
)
from .types import FeatureSpec, ThresholdSpec, VerificationCase
from .value import ValueOutcomeRule


@dataclass(frozen=True)
class RuleSpec:
    """JSON-serializable constructor for one reusable verification rule."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuleSpec:
        return cls(type=str(value["type"]), params=dict(value.get("params") or {}))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "params": self.params}


@dataclass(frozen=True)
class FeatureValidationSpec:
    """Pending or completed rule validation stored with an interpretation."""

    rule: RuleSpec
    threshold: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(mode="ratio_to_max", value=0.7, scope="sample")
    )
    method: str = "chess_rule"
    status: str = "pending"
    passed: bool | None = None
    cases_source: str = "top_activation_samples"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeatureValidationSpec:
        threshold = value.get("threshold") or {}
        cases = value.get("cases") or {}
        return cls(
            rule=RuleSpec.from_dict(value["rule"]),
            threshold=ThresholdSpec(**threshold),
            method=str(value.get("method", "chess_rule")),
            status=str(value.get("status", "pending")),
            passed=value.get("passed"),
            cases_source=str(cases.get("source", "top_activation_samples")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": self.method,
            "status": self.status,
            "rule": self.rule.to_dict(),
            "threshold": asdict(self.threshold),
            "cases": {"source": self.cases_source},
        }
        if self.passed is not None:
            result["passed"] = self.passed
        return result


def _nested_rules(params: dict[str, Any]) -> tuple[VerificationRule, ...]:
    raw_rules = params.pop("rules", [])
    return tuple(build_rule(RuleSpec.from_dict(item)) for item in raw_rules)


def build_rule(spec: RuleSpec | Mapping[str, Any]) -> VerificationRule | ValueOutcomeRule:
    """Build a verification rule from a compact interpretation config."""

    resolved = spec if isinstance(spec, RuleSpec) else RuleSpec.from_dict(spec)
    params = dict(resolved.params)
    constructors: dict[str, Any] = {
        "piece_type": PieceTypeRule,
        "empty_squares": EmptySquareRule,
        "occupied_squares": OccupiedSquareRule,
        "piece_neighborhood": PieceNeighborhoodRule,
        "piece_ray": PieceRayRule,
        "piece_front_span": PieceFrontSpanRule,
        "piece_multi_hop_destination": PieceMultiHopDestinationRule,
        "relative_offset": RelativeOffsetRule,
        "piece_destination": PieceDestinationRule,
        "piece_attack": PieceAttackRule,
        "protected_piece": ProtectedPieceRule,
        "capturable_piece": CapturablePieceRule,
        "checking_move_destination": CheckingMoveDestinationRule,
        "move_start": MoveStartSquareRule,
        "move_end": MoveEndSquareRule,
        "move_start_piece": MoveStartPieceRule,
        "move_end_piece": MoveEndPieceRule,
        "king_neighborhood": KingNeighborhoodRule,
        "queen_check_around_opponent_king": QueenCheckAroundOpponentKingRule,
        "relative_board_region": RelativeBoardRegionRule,
        "value_outcome": ValueOutcomeRule,
    }
    if resolved.type == "any_of":
        return AnyOfRule(rules=_nested_rules(params), **params)
    if resolved.type == "all_of":
        return AllOfRule(rules=_nested_rules(params), **params)
    try:
        constructor = constructors[resolved.type]
    except KeyError as error:
        available = ", ".join(sorted([*constructors, "all_of", "any_of"]))
        raise KeyError(f"Unknown rule type '{resolved.type}'. Available types: {available}") from error
    return constructor(**params)


def run_validation_spec(
    *,
    validation: FeatureValidationSpec | Mapping[str, Any],
    model: Any,
    lorsas: Any,
    transcoders: Any,
    feature: Any,
    cases: Any,
    **kwargs: Any,
) -> Any:
    """Run one stored validation config without feature-specific Python code."""

    from .evaluator import evaluate_feature_rule

    resolved = validation if isinstance(validation, FeatureValidationSpec) else FeatureValidationSpec.from_dict(validation)
    if resolved.rule.type == "value_outcome":
        from .value import evaluate_feature_value_rule

        return evaluate_feature_value_rule(
            model=model,
            lorsas=lorsas,
            transcoders=transcoders,
            feature=feature,
            rule=build_rule(resolved.rule),
            cases=cases,
            threshold=resolved.threshold,
            **kwargs,
        )
    return evaluate_feature_rule(
        model=model,
        lorsas=lorsas,
        transcoders=transcoders,
        feature=feature,
        rule=build_rule(resolved.rule),
        cases=cases,
        threshold=resolved.threshold,
        **kwargs,
    )


def feature_spec_from_proposal(proposal: Mapping[str, Any]) -> FeatureSpec:
    """Read feature identity directly from one taxonomy proposal."""

    feature_type = str(proposal["feature_type"])
    if feature_type == "tc":
        feature_type = "transcoder"
    if feature_type not in {"transcoder", "lorsa"}:
        raise ValueError(f"Unsupported proposal feature_type: {feature_type}")
    return FeatureSpec(
        feature_type=feature_type,
        layer=int(proposal["layer"]),
        feature_id=int(proposal["feature_index"]),
    )


def run_proposal_validation(
    *,
    proposal: Mapping[str, Any],
    model: Any,
    lorsas: Any,
    transcoders: Any,
    cases: Sequence[VerificationCase | str],
    validation_index: int = 0,
    **kwargs: Any,
) -> Any:
    """Execute a proposal's declarative rule without feature-specific code."""

    validations = proposal.get("validation") or []
    if not validations:
        raise ValueError("Proposal does not contain a validation rule")
    try:
        validation = validations[validation_index]
    except IndexError as error:
        raise IndexError(f"Proposal validation index {validation_index} does not exist") from error
    return run_validation_spec(
        validation=validation,
        model=model,
        lorsas=lorsas,
        transcoders=transcoders,
        feature=feature_spec_from_proposal(proposal),
        cases=cases,
        **kwargs,
    )
