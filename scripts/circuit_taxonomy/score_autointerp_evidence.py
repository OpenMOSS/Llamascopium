"""Score activation consistency and complexity for exported chess features."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chess
from tqdm import tqdm


def piece_name(board: chess.Board, square: str) -> str:
    piece = board.piece_at(chess.parse_square(square))
    if piece is None:
        return "empty"
    side = "Own" if piece.color == board.turn else "Opponent"
    return f"{side} {chess.piece_name(piece.piece_type)}"


def relation_fact(board: chess.Board, source: str, target: str) -> tuple[str, str] | None:
    source_piece = board.piece_at(chess.parse_square(source))
    target_piece = board.piece_at(chess.parse_square(target))
    if target_piece is None:
        return None
    target_identity = piece_name(board, target)
    if not board.is_attacked_by(target_piece.color, chess.parse_square(source)):
        return None
    if source_piece is None:
        return f"movement/control field of {target_identity}", "movement"
    source_identity = piece_name(board, source)
    if source_piece.color == target_piece.color:
        return f"{target_identity} protects {source_identity}", "protection"
    return f"{target_identity} attacks {source_identity}", "capture"


def region(index: int) -> str:
    row, col = divmod(index, 8)
    if row in {0, 7} or col in {0, 7}:
        return "board edge/back rank"
    if row in {3, 4} and col in {2, 3, 4, 5}:
        return "central region"
    if row <= 2:
        return "Own-side region"
    if row >= 5:
        return "Opponent-side region"
    return "middle-board region"


def analyze_feature(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("top_activation_samples", [])
    usable = []
    candidate_hits: dict[tuple[str, str], set[int]] = defaultdict(set)
    sample_facts: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples):
        fen = sample.get("fen")
        activations = sample.get("top_activated_squares") or []
        if not fen or not activations:
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        usable.append(sample_index)
        main = activations[0]
        square = str(main["square"])
        index = int(main.get("index", -1))
        active_identity = piece_name(board, square)
        if active_identity != "empty":
            candidate_hits[("active_piece", active_identity)].add(sample_index)
        candidate_hits[("fixed_index", str(index))].add(sample_index)
        candidate_hits[("region", region(index))].add(sample_index)

        relation = None
        z_pairs = sample.get("top_z_pairs") or []
        if z_pairs:
            z = z_pairs[0]
            source, target = str(z["source_square"]), str(z["target_square"])
            anchor_identity = piece_name(board, target)
            if anchor_identity != "empty":
                candidate_hits[("z_anchor", anchor_identity)].add(sample_index)
            target_index = int(z.get("target_index", -1))
            source_index = int(z.get("source_index", -1))
            candidate_hits[("z_fixed_index", str(target_index))].add(sample_index)
            candidate_hits[("z_region", region(target_index))].add(sample_index)
            source_row, source_col = divmod(source_index, 8)
            target_row, target_col = divmod(target_index, 8)
            delta_row, delta_col = abs(source_row - target_row), abs(source_col - target_col)
            if delta_row == 0 and delta_col == 0:
                geometry = "same-square z relation"
            elif delta_row == delta_col:
                geometry = "diagonal z relation"
            elif delta_row == 0 or delta_col == 0:
                geometry = "rank/file z relation"
            elif sorted((delta_row, delta_col)) == [1, 2]:
                geometry = "knight-offset z relation"
            else:
                geometry = "nonlocal z relation"
            candidate_hits[("z_geometry", geometry)].add(sample_index)
            relation = relation_fact(board, source, target)
            if relation:
                candidate_hits[(relation[1], relation[0])].add(sample_index)
        sample_facts.append(
            {
                "sample_index": sample_index,
                "square": square,
                "index": index,
                "active_identity": active_identity,
                "relation": relation[0] if relation else None,
            }
        )

    n = len(usable)
    if n == 0:
        return {
            "hypothesis": "No usable activation boards",
            "kind": "none",
            "fit": 0,
            "partial": 0,
            "deviations": 0,
            "consistency": 1,
            "complexity": 1,
            "confidence": 0.2,
            "facts": sample_facts,
        }

    # A feature may encode a compact two-way pattern (for example symmetric
    # b/g-file squares or diagonal/rank z routes). Treat a two-value union as
    # structured, but never merge arbitrary attack/protection hypotheses.
    for pair_kind in {"active_piece", "z_anchor", "fixed_index", "z_fixed_index", "z_geometry"}:
        options = sorted(
            (
                (len(hits), hypothesis, hits)
                for (candidate_kind, hypothesis), hits in candidate_hits.items()
                if candidate_kind == pair_kind
            ),
            reverse=True,
        )
        if len(options) >= 2:
            _, first_name, first_hits = options[0]
            _, second_name, second_hits = options[1]
            union = first_hits | second_hits
            if len(union) > len(first_hits):
                candidate_hits[(pair_kind, f"{first_name} or {second_name}")] = union

    priorities = {
        "capture": 8,
        "protection": 8,
        "movement": 7,
        "active_piece": 6,
        "z_anchor": 5,
        "z_geometry": 4,
        "fixed_index": 3,
        "z_fixed_index": 3,
        "region": 1,
        "z_region": 1,
    }
    candidates = []
    for (kind, hypothesis), hits in candidate_hits.items():
        count = len(hits)
        minimum = max(3, int(n * (0.5 if kind in {"capture", "protection", "movement"} else 0.55)))
        if count >= minimum:
            candidates.append((count, priorities[kind], kind, hypothesis, hits))
    if candidates:
        # Broad regions are fallback themes. Prefer any sufficiently supported
        # piece/relation/exact-geometry rule even when the broad region has more hits.
        structured = [item for item in candidates if item[2] not in {"region", "z_region"}]
        pool = structured or candidates
        count, _, kind, hypothesis, hits = max(pool, key=lambda item: (item[0], item[1]))
    else:
        (kind, hypothesis), hits = max(candidate_hits.items(), key=lambda item: len(item[1]))
        count = len(hits)

    partial_hits: set[int] = set()
    if kind in {"capture", "protection", "movement"}:
        relation_family = {
            idx for (candidate_kind, _), idxs in candidate_hits.items() if candidate_kind == kind for idx in idxs
        }
        partial_hits = relation_family - hits
    elif kind in {"active_piece", "z_anchor"}:
        wanted_piece = hypothesis.split()[-1]
        partial_hits = {
            fact["sample_index"]
            for fact in sample_facts
            if fact["sample_index"] not in hits and fact["active_identity"].endswith(wanted_piece)
        }
    partial = len(partial_hits)
    deviations = max(0, n - count - partial)

    if kind in {"region", "z_region"}:
        consistency = 2 if count >= int(n * 0.6) else 1
    elif count == n:
        consistency = 5
    elif deviations + partial <= 2:
        consistency = 4
    elif count >= max(3, int(n * 0.55)):
        consistency = 3
    elif kind == "region" or count >= int(n * 0.4):
        consistency = 2
    else:
        consistency = 1

    strong_kinds = {
        candidate_kind
        for (candidate_kind, _), idxs in candidate_hits.items()
        if candidate_kind in {"capture", "protection", "movement"} and len(idxs) >= max(3, int(n * 0.6))
    }
    z_geometry_count = max(
        (
            len(idxs)
            for (candidate_kind, candidate_name), idxs in candidate_hits.items()
            if candidate_kind == "z_geometry" and " or " not in candidate_name
        ),
        default=0,
    )
    if kind in {"capture", "protection", "movement"}:
        complexity = 4 if len(strong_kinds) >= 2 else 3
    elif kind == "active_piece":
        complexity = 3 if z_geometry_count >= n * 0.65 else 2
    elif kind == "z_anchor":
        complexity = 3 if z_geometry_count >= n * 0.65 else 2
    elif kind == "z_geometry":
        complexity = 3 if " or " not in hypothesis or z_geometry_count >= n * 0.7 else 2
    elif kind in {"fixed_index", "z_fixed_index", "region", "z_region"}:
        complexity = 3 if z_geometry_count >= n * 0.65 else 2
    else:
        complexity = 1

    confidence = min(0.95, 0.55 + 0.4 * (count + 0.5 * partial) / n)
    return {
        "hypothesis": hypothesis,
        "kind": kind,
        "fit": count,
        "partial": partial,
        "deviations": deviations,
        "consistency": consistency,
        "complexity": complexity,
        "confidence": round(confidence, 2),
        "facts": sample_facts,
    }


def analyze_feature_strict(row: dict[str, Any]) -> dict[str, Any]:
    """Audit full chess rules rather than categorical square coincidence."""
    samples = row.get("top_activation_samples", [])
    candidate_hits: dict[tuple[str, str], set[int]] = defaultdict(set)
    sample_facts: list[dict[str, Any]] = []
    main_indices: dict[int, int] = {}
    usable: list[int] = []

    for sample_index, sample in enumerate(samples):
        fen = sample.get("fen")
        activations = (sample.get("top_activated_squares") or [])[:4]
        if not fen or not activations:
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        usable.append(sample_index)
        main = activations[0]
        main_square = str(main["square"])
        main_index = int(main.get("index", -1))
        main_indices[sample_index] = main_index
        active_identity = piece_name(board, main_square)
        if active_identity != "empty":
            candidate_hits[("detector", active_identity)].add(sample_index)

        activation_squares = [chess.parse_square(str(item["square"])) for item in activations]
        movement_coverage: dict[str, int] = defaultdict(int)
        for activation_square in activation_squares:
            identities = set()
            for origin, piece in board.piece_map().items():
                if activation_square in board.attacks(origin):
                    side = "Own" if piece.color == board.turn else "Opponent"
                    identity = f"{side} {chess.piece_name(piece.piece_type)}"
                    identities.add(identity)
            for identity in identities:
                movement_coverage[identity] += 1

        required_coverage = max(1, (len(activation_squares) + 1) // 2)
        movement_identities = {
            identity for identity, covered in movement_coverage.items() if covered >= required_coverage
        }
        for identity in movement_identities:
            candidate_hits[("movement", f"{identity} coverage/movement field")].add(sample_index)

        own_covering_types = {
            identity
            for identity in movement_identities
            if identity.startswith("Own ") and not identity.endswith("king")
        }
        if len(own_covering_types) >= 2:
            pieces = " + ".join(sorted(own_covering_types)[:3])
            candidate_hits[("multi_piece", f"multi-piece attacking/coverage squares ({pieces})")].add(sample_index)

        direction = 8 if board.turn == chess.WHITE else -8
        pawn_advance_hits = 0
        for activation_square in activation_squares:
            behind = activation_square - direction
            if 0 <= behind < 64:
                piece = board.piece_at(behind)
                pawn_advance_hits += bool(piece and piece.color == board.turn and piece.piece_type == chess.PAWN)
        if pawn_advance_hits >= required_coverage:
            candidate_hits[("movement", "Own pawn advance squares")].add(sample_index)

        relation_names = set()
        anchor_names = set()
        for z in (sample.get("top_z_pairs") or [])[:16]:
            source, target = str(z["source_square"]), str(z["target_square"])
            anchor = piece_name(board, target)
            if anchor != "empty":
                anchor_names.add(anchor)
                candidate_hits[("z_anchor", f"z-pattern attends to {anchor}")].add(sample_index)
            relation = relation_fact(board, source, target)
            if relation:
                relation_names.add(relation[0])
                candidate_hits[(relation[1], relation[0])].add(sample_index)

        if relation_names and movement_identities:
            candidate_hits[("compound", "movement/coverage combined with z-linked attack or protection")].add(
                sample_index
            )
        if relation_names and len(own_covering_types) >= 2:
            candidate_hits[("rich_tactic", "multi-piece coverage combined with z-linked attack or protection")].add(
                sample_index
            )

        sample_facts.append(
            {
                "sample_index": sample_index,
                "square": main_square,
                "index": main_index,
                "active_identity": active_identity,
                "movement_identities": sorted(movement_identities),
                "relation": sorted(relation_names)[0] if relation_names else None,
                "z_anchors": sorted(anchor_names),
            }
        )

    n = len(usable)
    if n == 0:
        return {
            "hypothesis": "No usable activation boards",
            "kind": "none",
            "fit": 0,
            "partial": 0,
            "deviations": 0,
            "consistency": 1,
            "complexity": 1,
            "confidence": 0.2,
            "facts": sample_facts,
        }

    index_counts = Counter(main_indices.values())
    top_indices: list[int] = []
    for index, _ in index_counts.most_common(4):
        top_indices.append(index)
        hits = {
            sample_index for sample_index, sample_main_index in main_indices.items() if sample_main_index in top_indices
        }
        candidate_hits[("spatial", f"model-relative squares {top_indices}")] = hits

    region_hits: dict[str, set[int]] = defaultdict(set)
    for sample_index, index in main_indices.items():
        region_hits[region(index)].add(sample_index)
    for region_name, hits in region_hits.items():
        candidate_hits[("broad_theme", region_name)].update(hits)

    minimums = {
        "rich_tactic": 0.55,
        "compound": 0.55,
        "multi_piece": 0.55,
        "capture": 0.55,
        "protection": 0.55,
        "movement": 0.55,
        "detector": 0.55,
        "z_anchor": 0.55,
        "spatial": 0.7,
        "broad_theme": 0.6,
    }
    priority = {
        "rich_tactic": 10,
        "compound": 9,
        "multi_piece": 8,
        "capture": 7,
        "protection": 7,
        "movement": 6,
        "detector": 5,
        "z_anchor": 4,
        "spatial": 3,
        "broad_theme": 1,
    }
    candidates = [
        (len(hits), priority[kind], kind, hypothesis, hits)
        for (kind, hypothesis), hits in candidate_hits.items()
        if len(hits) >= max(3, int(n * minimums[kind]))
    ]
    if candidates:
        best_count = max(item[0] for item in candidates)
        # A chess-semantic rule may trail a trivial spatial mask by up to two boards.
        close = [item for item in candidates if item[0] >= best_count - 2]
        count, _, kind, hypothesis, hits = max(close, key=lambda item: (item[1], item[0]))
    else:
        (kind, hypothesis), hits = max(candidate_hits.items(), key=lambda item: len(item[1]))
        count = len(hits)

    deviations = n - count
    if kind == "broad_theme":
        consistency = 2
    elif deviations == 0:
        consistency = 5
    elif deviations <= 2:
        consistency = 4
    elif count >= int(n * 0.55):
        consistency = 3
    elif count >= int(n * 0.4):
        consistency = 2
    else:
        consistency = 1
    if n < 3:
        consistency = 2

    if kind == "rich_tactic":
        complexity = 5
    elif kind in {"compound", "multi_piece"}:
        complexity = 4
    elif kind in {"capture", "protection", "movement"}:
        complexity = 3
    elif kind == "detector":
        stable_spatial = max((len(hits) for hits in region_hits.values()), default=0) >= n * 0.8
        complexity = 2 if stable_spatial else 1
    elif kind in {"z_anchor", "spatial", "broad_theme"}:
        complexity = 2
    else:
        complexity = 1

    confidence = 0.35 if n < 3 else round(min(0.97, 0.58 + 0.39 * count / n), 2)
    return {
        "hypothesis": hypothesis,
        "kind": kind,
        "fit": count,
        "partial": 0,
        "deviations": deviations,
        "consistency": consistency,
        "complexity": complexity,
        "confidence": confidence,
        "facts": sample_facts,
    }


def analyze_feature_mlp(row: dict[str, Any]) -> dict[str, Any]:
    """Use a high evidence bar for dense MLP activation maps."""
    candidate_hits: dict[tuple[str, str], set[int]] = defaultdict(set)
    region_hits: dict[str, set[int]] = defaultdict(set)
    sample_facts: list[dict[str, Any]] = []
    usable: list[int] = []
    piece_labels = {
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
    }

    for sample_index, sample in enumerate(row.get("top_activation_samples", [])):
        fen = sample.get("fen")
        activations = sample.get("top_activated_squares") or []
        if not fen or not activations:
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        usable.append(sample_index)
        main = activations[0]
        main_square = str(main["square"])
        main_index = int(main.get("index", -1))
        active_identity = piece_name(board, main_square)
        if active_identity != "empty":
            candidate_hits[("detector", active_identity)].add(sample_index)
        region_hits[region(main_index)].add(sample_index)

        coverage: dict[str, int] = defaultdict(int)
        for activation in activations:
            activation_square = chess.parse_square(str(activation["square"]))
            identities = set()
            for origin, piece in board.piece_map().items():
                if piece.piece_type not in piece_labels:
                    continue
                if activation_square in board.attacks(origin):
                    side = "Own" if piece.color == board.turn else "Opponent"
                    identities.add(f"{side} {piece_labels[piece.piece_type]}")
            for identity in identities:
                coverage[identity] += 1
        required = math.ceil(len(activations) * 0.75)
        movement_identities = sorted(identity for identity, count in coverage.items() if count >= required)
        for identity in movement_identities:
            candidate_hits[("movement", f"{identity} coverage/movement field")].add(sample_index)
        sample_facts.append(
            {
                "sample_index": sample_index,
                "square": main_square,
                "index": main_index,
                "active_identity": active_identity,
                "movement_identities": movement_identities,
                "relation": None,
                "z_anchors": [],
            }
        )

    n = len(usable)
    if n == 0:
        return {
            "hypothesis": "No usable activation boards",
            "kind": "none",
            "fit": 0,
            "partial": 0,
            "deviations": 0,
            "consistency": 1,
            "complexity": 1,
            "confidence": 0.2,
            "facts": sample_facts,
        }

    candidates = [
        (len(hits), 2 if kind == "movement" else 1, kind, hypothesis, hits)
        for (kind, hypothesis), hits in candidate_hits.items()
        if len(hits) >= max(3, math.ceil(n * 0.55))
    ]
    if candidates:
        best_count = max(item[0] for item in candidates)
        close = [item for item in candidates if item[0] >= best_count - 1]
        count, _, kind, hypothesis, _ = max(close, key=lambda item: (item[1], item[0]))
    else:
        broad_name, broad_samples = max(region_hits.items(), key=lambda item: len(item[1]))
        count = len(broad_samples)
        if count >= n * 0.8:
            kind, hypothesis = "broad_theme", broad_name
        else:
            kind, hypothesis, count = "none", "No stable chess-semantic pattern", 0

    deviations = n - count
    if kind == "none":
        consistency = 1
    elif kind == "broad_theme":
        consistency = 2
    elif deviations == 0:
        consistency = 5
    elif deviations <= 2:
        consistency = 4
    elif count >= n * 0.55:
        consistency = 3
    elif count >= n * 0.4:
        consistency = 2
    else:
        consistency = 1

    if kind == "movement":
        complexity = 3
    elif kind == "detector":
        stable_region = max((len(hits) for hits in region_hits.values()), default=0) >= n * 0.8
        complexity = 2 if stable_region else 1
    elif kind == "broad_theme":
        complexity = 2
    else:
        complexity = 1
    confidence = round(0.45 + 0.5 * count / n, 2) if count else 0.45
    return {
        "hypothesis": hypothesis,
        "kind": kind,
        "fit": count,
        "partial": 0,
        "deviations": deviations,
        "consistency": consistency,
        "complexity": complexity,
        "confidence": confidence,
        "facts": sample_facts,
    }


def representative_text(facts: list[dict[str, Any]], hypothesis: str) -> str:
    fits = [fact for fact in facts if fact.get("relation") == hypothesis]
    if not fits:
        fits = facts
    return "; ".join(
        f"s{fact['sample_index']} {fact['square']}={fact['active_identity']}"
        + (f", {fact['relation']}" if fact.get("relation") else "")
        + (f", covered by {', '.join(fact['movement_identities'][:3])}" if fact.get("movement_identities") else "")
        + (f", z anchors {', '.join(fact['z_anchors'][:2])}" if fact.get("z_anchors") else "")
        for fact in fits[:3]
    )


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    audit = analyze_feature_mlp(row) if row.get("feature_type") == "mlp" else analyze_feature_strict(row)
    interpretation = str(row.get("interpretation", "") or "").strip()
    hypothesis = audit["hypothesis"]
    usable = audit["fit"] + audit["partial"] + audit["deviations"]
    examples = representative_text(audit["facts"], hypothesis)
    return {
        "dictionary_name": row["dictionary_name"],
        "feature_index": int(row["feature_index"]),
        "feature_type": row.get("feature_type"),
        "layer": row.get("layer"),
        "interpretation": interpretation or hypothesis,
        "interpretation_source": "existing" if interpretation else "inferred_for_scoring",
        "scoring_basis": (
            "top_activation_samples_and_lorsa_z_pairs_only"
            if row.get("feature_type") == "lorsa"
            else "top_activation_samples_only"
        ),
        "activation_consistency": audit["consistency"],
        "complexity": audit["complexity"],
        "usable_samples": usable,
        "fit_samples": audit["fit"],
        "partial_samples": audit["partial"],
        "deviating_samples": audit["deviations"],
        "confidence": (
            audit["confidence"] if interpretation or usable < 3 else round(max(0.5, audit["confidence"] - 0.1), 2)
        ),
        "consistency_rationale": (
            f"逐一检查 {usable} 个可用棋盘；{audit['fit']} 个符合“{hypothesis}”，"
            f"{audit['partial']} 个部分符合，{audit['deviations']} 个偏离。"
        ),
        "complexity_rationale": {
            1: f"仅根据 top samples，最简解释是单一棋子身份“{hypothesis}”，不需要额外关系。",
            2: f"仅根据 top samples，“{hypothesis}”需要稳定位置/锚点条件，但没有稳定多棋子交互。",
            3: f"仅根据 top samples，“{hypothesis}”要求一个实际移动、覆盖、保护或攻击关系。",
            4: f"仅根据 top samples，“{hypothesis}”要求多个棋子的覆盖图或关系共同成立。",
            5: f"仅根据 top samples，“{hypothesis}”要求跨局面的丰富强制战术/战略序列。",
        }[audit["complexity"]],
        "evidence_summary": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    scored = [score_row(row) for row in tqdm(rows, desc="Scoring AutoInterp", unit="feature")]
    keys = [(row["dictionary_name"], row["feature_index"]) for row in scored]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate feature keys in input")
    for row in scored:
        assert 1 <= row["activation_consistency"] <= 5
        assert 1 <= row["complexity"] <= 5
        assert row["usable_samples"] == row["fit_samples"] + row["partial_samples"] + row["deviating_samples"]
    summary = {
        "feature_count": len(scored),
        "mean_activation_consistency": round(sum(row["activation_consistency"] for row in scored) / len(scored), 4),
        "mean_complexity": round(sum(row["complexity"] for row in scored) / len(scored), 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "features": scored}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
