"""Reusable utilities for verifying whether a chess SAE feature matches a rule.

This package is meant to replace the ad-hoc taxonomy notebook pattern:

1. manually define one rule inside a notebook
2. manually choose an activation threshold
3. manually run the model and compute accuracy / recall / F1

The new package keeps those concerns separate:

- ``rules`` defines *what squares should be positive*
- ``thresholds`` defines *when a real-valued feature counts as active*
- ``evaluator`` defines *how activations and labels are compared*

Typical usage
-------------
```python
from feature_varification import (
    FeatureSpec,
    ThresholdSpec,
    VerificationCase,
    PieceFrontSpanRule,
    evaluate_feature_rule,
)

feature = FeatureSpec(feature_type="transcoder", layer=10, feature_id=1958)
rule = PieceFrontSpanRule(piece_type="own k", include_adjacent_files=True)
threshold = ThresholdSpec(mode="ratio_to_max", value=0.7, scope="sample")

result = evaluate_feature_rule(
    model=model,
    lorsas=lorsas,
    transcoders=transcoders,
    feature=feature,
    rule=rule,
    cases=[VerificationCase(fen=fen, move_uci=move) for fen, move in rows],
    threshold=threshold,
)
```
"""

from .evaluator import collect_feature_activations, evaluate_feature_rule, extract_feature_activations
from .rules import (
    RULE_REGISTRY,
    AllOfRule,
    AnyOfRule,
    CapturablePieceRule,
    CheckingMoveDestinationRule,
    EmptySquareRule,
    FunctionalRule,
    KingNeighborhoodRule,
    MoveEndPieceRule,
    MoveEndSquareRule,
    MoveStartPieceRule,
    MoveStartSquareRule,
    OccupiedSquareRule,
    PieceAttackRule,
    PieceDestinationRule,
    PieceFrontSpanRule,
    PieceNeighborhoodRule,
    PieceRayRule,
    PieceTypeRule,
    ProtectedPieceRule,
    QueenCheckAroundOpponentKingRule,
    RelativeBoardRegionRule,
    RelativeOffsetRule,
    VerificationRule,
    front_cone_rule,
    front_file_rule,
    register_rule,
    same_diagonal_rule,
    same_file_rule,
    same_rank_rule,
)
from .specs import (
    FeatureValidationSpec,
    RuleSpec,
    build_rule,
    feature_spec_from_proposal,
    run_proposal_validation,
    run_validation_spec,
)
from .thresholds import resolve_thresholds
from .types import (
    FeatureSpec,
    FeatureType,
    FeatureVerificationResult,
    RuleEvaluation,
    ThresholdSpec,
    VerificationCase,
    VerificationCounts,
)
from .value import ValueOutcomeRule, evaluate_feature_value_rule

__all__ = [
    "AllOfRule",
    "AnyOfRule",
    "CapturablePieceRule",
    "CheckingMoveDestinationRule",
    "EmptySquareRule",
    "FeatureSpec",
    "FeatureType",
    "FeatureVerificationResult",
    "FeatureValidationSpec",
    "FunctionalRule",
    "KingNeighborhoodRule",
    "MoveEndPieceRule",
    "MoveEndSquareRule",
    "MoveStartPieceRule",
    "MoveStartSquareRule",
    "OccupiedSquareRule",
    "PieceAttackRule",
    "PieceDestinationRule",
    "PieceFrontSpanRule",
    "PieceNeighborhoodRule",
    "PieceRayRule",
    "PieceTypeRule",
    "ProtectedPieceRule",
    "QueenCheckAroundOpponentKingRule",
    "RULE_REGISTRY",
    "RelativeBoardRegionRule",
    "RelativeOffsetRule",
    "RuleSpec",
    "RuleEvaluation",
    "ThresholdSpec",
    "VerificationCase",
    "VerificationCounts",
    "VerificationRule",
    "ValueOutcomeRule",
    "build_rule",
    "collect_feature_activations",
    "evaluate_feature_rule",
    "evaluate_feature_value_rule",
    "extract_feature_activations",
    "feature_spec_from_proposal",
    "front_cone_rule",
    "front_file_rule",
    "register_rule",
    "resolve_thresholds",
    "run_proposal_validation",
    "run_validation_spec",
    "same_diagonal_rule",
    "same_file_rule",
    "same_rank_rule",
]
