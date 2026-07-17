from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from chess_utils import get_pos_from_square

from .specs import RuleSpec, build_rule

PIECE_NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}
PIECE_TYPES = tuple(f"{owner} {piece}" for owner in ("own", "opponent") for piece in PIECE_NAMES)


@dataclass(frozen=True)
class RuleScore:
    rule: RuleSpec
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "samples": self.samples,
        }


def _spec(type_: str, **params: Any) -> RuleSpec:
    return RuleSpec(type=type_, params=params)


def candidate_rule_specs(taxonomy: str) -> list[RuleSpec]:
    """Return reusable, board-relative candidate rules for one taxonomy label."""

    piece_occupancy = [_spec("piece_type", piece_type=piece) for piece in PIECE_TYPES]
    movement = [_spec("piece_destination", piece_type=piece) for piece in PIECE_TYPES]
    attacks = [_spec("piece_attack", piece_type=piece) for piece in PIECE_TYPES]
    neighborhoods = [
        _spec("piece_neighborhood", piece_type=piece, radius=1, include_center=False) for piece in PIECE_TYPES
    ]
    regions = [
        _spec("relative_board_region", region=region)
        for region in (
            "own_back_rank",
            "own_two_ranks",
            "opponent_back_rank",
            "opponent_two_ranks",
            "center_four",
            "extended_center",
            "edge",
            "queenside",
            "kingside",
        )
    ]
    if taxonomy == "Det":
        return [*piece_occupancy, _spec("occupied_squares", owner="own"), _spec("occupied_squares", owner="opponent")]
    if taxonomy == "Src":
        return [_spec("move_start_piece", piece_type=piece) for piece in PIECE_TYPES] + [_spec("move_start")]
    if taxonomy == "Tgt":
        return [
            *(
                _spec("move_end_piece", piece_type=piece, target_owner=target)
                for piece in PIECE_TYPES
                for target in ("empty", "opponent", None)
            ),
            _spec("move_end"),
        ]
    if taxonomy == "Mov":
        return [
            *movement,
            *attacks,
            *(
                _spec(
                    "piece_front_span",
                    piece_type=f"{owner} p",
                    max_steps=1,
                    include_same_file=True,
                    include_adjacent_files=False,
                )
                for owner in ("own", "opponent")
            ),
            *(
                _spec("piece_ray", piece_type=piece, directions=(direction,), stop_at_blockers=True)
                for piece in PIECE_TYPES
                for direction in ("rank", "file", "diagonal")
            ),
        ]
    if taxonomy == "Pro":
        return [
            *(
                _spec("protected_piece", piece_type=piece, protector_owner=piece.split()[0])
                for piece in PIECE_TYPES
            ),
            *attacks,
        ]
    if taxonomy == "Cap":
        return [
            *(
                _spec("capturable_piece", attacker_piece_type=piece, attacker_owner=piece.split()[0])
                for piece in PIECE_TYPES
            ),
            _spec("capturable_piece", attacker_owner="own"),
            _spec("capturable_piece", attacker_owner="opponent"),
        ]
    if taxonomy == "Tac":
        return [
            *(_spec("checking_move_destination", piece_type=f"own {piece}") for piece in PIECE_NAMES),
            _spec("checking_move_destination"),
            *attacks,
            *neighborhoods,
            _spec("king_neighborhood", owner="own"),
            _spec("king_neighborhood", owner="opponent"),
        ]
    if taxonomy in {"Spa", "Reg"}:
        return [*neighborhoods, *regions]
    if taxonomy == "Uninterpretable":
        return [*piece_occupancy, *movement, *attacks, *neighborhoods, *regions, _spec("empty_squares")]
    return []


def _active_mask(sample: Mapping[str, Any], ratio: float) -> list[bool]:
    mask = [False] * 64
    activations = list(sample.get("top_activated_squares") or [])
    if not activations:
        return mask
    maximum = max(float(item.get("value", 0.0)) for item in activations)
    selected = [item for item in activations if float(item.get("value", 0.0)) >= ratio * maximum]
    if not selected:
        selected = activations[:1]
    for item in selected:
        square = str(item.get("square", ""))
        if square:
            mask[get_pos_from_square(str(sample["fen"]), square)] = True
    return mask


def score_rule_on_samples(
    rule_spec: RuleSpec,
    samples: Sequence[Mapping[str, Any]],
    *,
    activation_ratio: float = 0.7,
) -> RuleScore:
    rule = build_rule(rule_spec)
    if not hasattr(rule, "evaluate"):
        raise TypeError(f"Rule {rule_spec.type} is not a square-mask rule")
    tp = fp = fn = used = 0
    for sample in samples:
        fen = str(sample.get("fen", ""))
        if not fen:
            continue
        top_moves = sample.get("top_moves") or []
        move_uci = str(top_moves[0].get("uci")) if top_moves else None
        if rule.requires_move_uci and not move_uci:
            continue
        active = _active_mask(sample, activation_ratio)
        expected = rule.evaluate(fen, move_uci).mask
        tp += sum(left and right for left, right in zip(active, expected))
        fp += sum((not left) and right for left, right in zip(active, expected))
        fn += sum(left and (not right) for left, right in zip(active, expected))
        used += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RuleScore(rule_spec, precision, recall, f1, tp, fp, fn, used)


def select_rule_for_samples(
    taxonomy: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    activation_ratio: float = 0.7,
) -> RuleScore | None:
    candidates = candidate_rule_specs(taxonomy)
    if not candidates:
        return None
    scores = [score_rule_on_samples(spec, samples, activation_ratio=activation_ratio) for spec in candidates]
    return max(scores, key=lambda item: (item.f1, item.precision, item.recall, -len(str(item.rule.params))))


def value_rule_for_samples(samples: Sequence[Mapping[str, Any]]) -> RuleSpec:
    """Choose the WDL bucket with the highest mean observed top activation."""

    buckets: dict[str, list[float]] = {"win": [], "draw": [], "loss": [], "decisive": []}
    for sample in samples:
        activations = sample.get("top_activated_squares") or []
        if not activations:
            continue
        score = max(float(item.get("value", 0.0)) for item in activations)
        wdl = sample.get("wdl") or {}
        win = float(wdl.get("current_player_win", 0.0))
        draw = float(wdl.get("draw", 0.0))
        loss = float(wdl.get("current_player_loss", 0.0))
        if win >= 0.6:
            buckets["win"].append(score)
        if draw >= 0.6:
            buckets["draw"].append(score)
        if loss >= 0.6:
            buckets["loss"].append(score)
        if max(win, loss) >= 0.6:
            buckets["decisive"].append(score)
    outcome = max(buckets, key=lambda key: (sum(buckets[key]) / len(buckets[key]) if buckets[key] else float("-inf")))
    return _spec("value_outcome", outcome=outcome, probability_threshold=0.6)


def describe_rule(spec: RuleSpec) -> str:
    """Render one concise, ownership-aware English interpretation."""

    params = spec.params
    piece = str(params.get("piece_type") or params.get("attacker_piece_type") or "")
    piece_name = ""
    if piece:
        owner, code = piece.split()
        piece_name = f"{owner} {PIECE_NAMES[code]}"
    attacker_side = f"{params.get('attacker_owner') or 'own'} side"
    descriptions = {
        "piece_type": f"Squares occupied by the {piece_name}.",
        "empty_squares": "Empty squares on the current board.",
        "occupied_squares": f"Squares occupied by {params.get('owner', 'either side')} pieces.",
        "piece_destination": f"Legal destination squares of the {piece_name}.",
        "piece_attack": f"Squares attacked or defended by the {piece_name}.",
        "piece_neighborhood": f"Squares adjacent to the {piece_name}.",
        "piece_ray": f"Blocked {params.get('directions', ['line'])[0]} line anchored on the {piece_name}.",
        "piece_front_span": f"The square immediately in front of the {piece_name}.",
        "piece_multi_hop_destination": (
            f"Exact {params.get('hops', 2)}-move destinations of the {piece_name}, "
            f"attending to its {params.get('attended_hop', 1)}-move destinations."
        ),
        "protected_piece": f"The {piece_name} when defended by another {params.get('protector_owner')} piece.",
        "capturable_piece": f"Opponent pieces immediately attacked by the {piece_name or attacker_side}.",
        "checking_move_destination": f"Legal checking destinations of the {piece_name or 'side-to-move pieces'}.",
        "move_start": "Source square of the top policy move.",
        "move_end": "Destination square of the top policy move.",
        "move_start_piece": f"Source square of the top policy move when moved by the {piece_name}.",
        "move_end_piece": (
            f"Destination square of the top policy move by the {piece_name}, with a {params.get('target_owner') or 'any'} target."
        ),
        "king_neighborhood": f"Squares adjacent to the {params.get('owner')} king.",
        "relative_board_region": f"The {str(params.get('region')).replace('_', ' ')} relative to the side to move.",
        "value_outcome": f"Positions with a high side-to-move {params.get('outcome')} probability.",
    }
    return descriptions.get(spec.type, f"Chess relation defined by rule {spec.type}.")


def unique_samples(rows: Iterable[Mapping[str, Any]], *, limit: int = 24) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        for sample in row.get("top_activation_samples") or []:
            fen = str(sample.get("fen", ""))
            if fen and fen not in seen:
                seen.add(fen)
                result.append(sample)
                if len(result) >= limit:
                    return result
    return result
