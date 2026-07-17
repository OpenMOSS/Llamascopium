from chess_utils import get_pos_from_square
from feature_varification import (
    CapturablePieceRule,
    CheckingMoveDestinationRule,
    FeatureValidationSpec,
    MoveEndPieceRule,
    MoveStartPieceRule,
    PieceAttackRule,
    RelativeBoardRegionRule,
    RuleSpec,
    ValueOutcomeRule,
    build_rule,
)
from feature_varification.selection import select_rule_for_samples

TACTICAL_FEN = "4k3/8/8/3r4/3R4/8/8/4K3 w - - 0 1"


def positions(rule, fen: str = TACTICAL_FEN, move_uci: str | None = None) -> set[int]:
    return {index for index, active in enumerate(rule.evaluate(fen, move_uci).mask) if active}


def test_piece_attack_and_capture_rules_are_owner_aware() -> None:
    target = get_pos_from_square(TACTICAL_FEN, "d5")

    assert target in positions(PieceAttackRule("own r"))
    assert target in positions(CapturablePieceRule(attacker_piece_type="own r"))
    assert target not in positions(CapturablePieceRule(attacker_piece_type="opponent r", attacker_owner="opponent"))


def test_piece_constrained_policy_move_rules() -> None:
    assert positions(MoveStartPieceRule("own r"), move_uci="d4d5") == {
        get_pos_from_square(TACTICAL_FEN, "d4")
    }
    assert positions(MoveEndPieceRule("own r", target_owner="opponent"), move_uci="d4d5") == {
        get_pos_from_square(TACTICAL_FEN, "d5")
    }
    assert not positions(MoveEndPieceRule("own q"), move_uci="d4d5")


def test_relative_back_rank_uses_side_to_move_orientation() -> None:
    white_fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    black_fen = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"
    rule = RelativeBoardRegionRule("own_back_rank")

    assert get_pos_from_square(white_fen, "a1") in positions(rule, white_fen)
    assert get_pos_from_square(black_fen, "a8") in positions(rule, black_fen)


def test_checking_destination_and_rule_factory() -> None:
    fen = "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    rule = build_rule({"type": "checking_move_destination", "params": {"piece_type": "own r"}})

    assert isinstance(rule, CheckingMoveDestinationRule)
    assert positions(rule, fen)


def test_validation_spec_round_trip() -> None:
    validation = FeatureValidationSpec(rule=RuleSpec("piece_type", {"piece_type": "own r"}))

    restored = FeatureValidationSpec.from_dict(validation.to_dict())
    assert restored == validation


def test_value_outcome_rule_reads_wdl_metadata() -> None:
    rule = ValueOutcomeRule("win", probability_threshold=0.6)

    assert rule.matches({"wdl": {"current_player_win": 0.8}})
    assert not rule.matches({"wdl": {"current_player_win": 0.2}})


def test_rule_selection_prefers_specific_piece_occupancy() -> None:
    samples = [
        {
            "fen": TACTICAL_FEN,
            "top_activated_squares": [{"square": "d4", "value": 4.0}],
            "top_moves": [{"uci": "d4d5"}],
        }
    ]

    selected = select_rule_for_samples("Det", samples)
    assert selected is not None
    assert selected.rule == RuleSpec("piece_type", {"piece_type": "own r"})
    assert selected.f1 == 1.0
