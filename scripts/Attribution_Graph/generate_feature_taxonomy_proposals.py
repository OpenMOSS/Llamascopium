#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = {"Det", "Src", "Tgt", "Val", "Cap", "Pro", "Mov", "Tac", "Reg", "Spa", "Uninterpretable"}
TAX_RE = re.compile(r"^\[(Det|Src|Tgt|Val|Cap|Pro|Mov|Tac|Reg|Spa|Uninterpretable)\]")
FILES = "abcdefgh"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def taxonomy_from_existing(text: str | None) -> str | None:
    if not text:
        return None
    match = TAX_RE.match(text.strip())
    return match.group(1) if match else None


def square_to_index(square: str | None) -> int | None:
    if not square or not re.match(r"^[a-h][1-8]$", square):
        return None
    return (8 - int(square[1])) * 8 + FILES.index(square[0])


def index_to_square(index: int | None) -> str | None:
    if index is None or index < 0 or index >= 64:
        return None
    return f"{FILES[index % 8]}{8 - index // 8}"


def parse_board(fen: str) -> tuple[dict[str, str], str]:
    parts = fen.split()
    board_part = parts[0]
    stm = parts[1] if len(parts) > 1 else "w"
    board: dict[str, str] = {}
    rank = 8
    file_idx = 0
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file_idx = 0
        elif ch.isdigit():
            file_idx += int(ch)
        else:
            board[f"{FILES[file_idx]}{rank}"] = ch
            file_idx += 1
    return board, stm


def side_piece(piece: str | None, stm: str) -> str:
    if not piece:
        return "empty"
    own = piece.isupper() if stm == "w" else piece.islower()
    side = "Own" if own else "Opponent"
    names = {"p": "Pawn", "n": "Knight", "b": "Bishop", "r": "Rook", "q": "Queen", "k": "King"}
    return f"{side}{names.get(piece.lower(), piece)}"


def top_samples(row: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    return (row.get("top_activation_samples") or [])[:limit]


def activated_squares(row: dict[str, Any], limit_samples: int = 8, limit_per_sample: int = 4) -> list[str]:
    out: list[str] = []
    for sample in top_samples(row, limit_samples):
        for item in (sample.get("top_activated_squares") or [])[:limit_per_sample]:
            square = item.get("square")
            if square:
                out.append(square)
    return out


def activated_piece_types(row: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for sample in top_samples(row):
        board, stm = parse_board(sample.get("fen", ""))
        for item in (sample.get("top_activated_squares") or [])[:3]:
            counter[side_piece(board.get(item.get("square")), stm)] += 1
    return counter


def move_hits(row: dict[str, Any]) -> tuple[int, int, int]:
    src_hits = 0
    tgt_hits = 0
    total = 0
    for sample in top_samples(row):
        squares = {item.get("square") for item in (sample.get("top_activated_squares") or [])[:4]}
        for move in sample.get("top_moves") or []:
            uci = move.get("uci", "")
            if len(uci) >= 4:
                total += 1
                if uci[:2] in squares:
                    src_hits += 1
                if uci[2:4] in squares:
                    tgt_hits += 1
    return src_hits, tgt_hits, total


def edge_region_score(squares: list[str]) -> float:
    if not squares:
        return 0.0
    edge = 0
    for square in squares:
        if square[0] in {"a", "h"} or square[1] in {"1", "8"}:
            edge += 1
    return edge / len(squares)


def high_z_relation(row: dict[str, Any]) -> tuple[Counter[str], Counter[str], int]:
    src_counter: Counter[str] = Counter()
    tgt_counter: Counter[str] = Counter()
    total = 0
    for sample in top_samples(row):
        board, stm = parse_board(sample.get("fen", ""))
        for pair in (sample.get("top_z_pairs") or [])[:3]:
            src = pair.get("source_square")
            tgt = pair.get("target_square")
            src_counter[side_piece(board.get(src), stm)] += 1
            tgt_counter[side_piece(board.get(tgt), stm)] += 1
            total += 1
    return src_counter, tgt_counter, total


def avg_abs_value(row: dict[str, Any]) -> float:
    values = [abs(float(sample.get("value", 0.0))) for sample in top_samples(row)]
    return sum(values) / len(values) if values else 0.0


def incoming_summary(
    node_id: str,
    links_by_target: dict[str, list[dict[str, Any]]],
    evidence_by_node: dict[str, dict[str, Any]],
    limit: int = 2,
) -> str:
    incoming = sorted(links_by_target.get(node_id, []), key=lambda link: abs(float(link.get("weight", 0.0))), reverse=True)[:limit]
    parts = []
    for link in incoming:
        src = link.get("source")
        row = evidence_by_node.get(src)
        if not row:
            parts.append(f"{src} w={float(link.get('weight', 0.0)):.3g}")
            continue
        existing = row.get("existing_interpretation") or ""
        label = taxonomy_from_existing(existing) or "unlabeled"
        parts.append(f"{row.get('dictionary_name')}#{row.get('feature_index')}[{label}] w={float(link.get('weight', 0.0)):.3g}")
    return "; ".join(parts) if parts else "no matched upstream evidence"


def classify_fresh(row: dict[str, Any], node: dict[str, Any], upstream: str) -> tuple[str, float, str, str]:
    layer = int(row.get("layer") or 0)
    feature_type = str(row.get("feature_type") or "")
    squares = activated_squares(row)
    square_counts = Counter(squares)
    piece_counts = activated_piece_types(row)
    src_hits, tgt_hits, move_total = move_hits(row)
    z_src, z_tgt, z_total = high_z_relation(row)
    edge_score = edge_region_score(squares)
    top_square_text = ", ".join(f"{sq}:{cnt}" for sq, cnt in square_counts.most_common(5)) or "none"
    top_piece, top_piece_count = piece_counts.most_common(1)[0] if piece_counts else ("none", 0)
    occupied_count = sum(count for piece, count in piece_counts.items() if piece != "empty")
    total_piece_obs = sum(piece_counts.values())
    value_score = avg_abs_value(row)

    evidence = (
        f"top squares {top_square_text}; piece pattern {dict(piece_counts.most_common(4))}; "
        f"move hits src={src_hits}, tgt={tgt_hits}, total={move_total}; "
        f"z-src {dict(z_src.most_common(3))}; upstream: {upstream}."
    )

    if layer >= 8 and move_total >= 3 and src_hits >= max(3, 0.35 * move_total) and occupied_count >= max(2, 0.45 * total_piece_obs):
        return (
            "Src",
            0.78,
            f"Activated occupied squares often coincide with top-move origins; connected inputs were checked ({upstream}), so this is a late-layer policy source feature.",
            evidence,
        )
    if layer >= 8 and move_total >= 3 and tgt_hits >= max(3, 0.35 * move_total):
        return (
            "Tgt",
            0.78,
            f"Activated squares repeatedly coincide with top-move destinations; connected inputs were checked ({upstream}), so this is a late-layer policy target feature.",
            evidence,
        )
    if layer >= 10 and value_score > 0.72 and edge_score < 0.65 and len(square_counts) >= 5:
        return (
            "Val",
            0.68,
            f"WDL/value evidence is strong across top samples, while square evidence is broad; connected inputs were checked ({upstream}), so value is the best review label.",
            evidence,
        )
    if total_piece_obs and top_piece != "empty" and top_piece_count / total_piece_obs >= 0.58:
        label = "Det"
        confidence = 0.76
        if layer >= 3 and ("Tac" in upstream or "Mov" in upstream or "Cap" in upstream or "Pro" in upstream):
            label = "Tac"
            confidence = 0.68
            rationale = (
                f"Activation alone looks like {top_piece} detection, but this layer receives relational inputs ({upstream}); "
                "treat this as a reviewable tactical composition rather than pure detection."
            )
        else:
            rationale = (
                f"Top activations repeatedly land on {top_piece}; connected inputs were checked ({upstream}) and do not provide a clearer movement or tactical relation."
            )
        return label, confidence, rationale, evidence
    if z_total and z_src:
        src_piece, src_count = z_src.most_common(1)[0]
        tgt_piece, tgt_count = z_tgt.most_common(1)[0] if z_tgt else ("none", 0)
        if src_piece != "empty" and tgt_piece != "empty" and src_piece[:3] != tgt_piece[:3] and src_count >= 2:
            return (
                "Cap",
                0.70,
                f"Lorsa z-pattern repeatedly links opposite-side pieces ({src_piece} to {tgt_piece}); connected inputs were checked ({upstream}), supporting a capture/attack relation.",
                evidence,
            )
        if src_piece != "empty" and tgt_piece != "empty" and src_piece[:3] == tgt_piece[:3] and src_count >= 2:
            return (
                "Pro",
                0.68,
                f"Lorsa z-pattern repeatedly links same-side pieces/squares ({src_piece} to {tgt_piece}); connected inputs were checked ({upstream}), supporting protection/control.",
                evidence,
            )
    if edge_score >= 0.75 and len(square_counts) >= 4:
        label = "Reg" if layer >= 2 else "Spa"
        return (
            label,
            0.67,
            f"Activations concentrate on board-edge/back-rank support squares rather than a stable piece identity; connected inputs were checked ({upstream}), so {label} is the conservative label.",
            evidence,
        )
    if any(piece in upstream for piece in ["Tac", "Cap", "Pro"]) and layer >= 3:
        return (
            "Tac",
            0.63,
            f"Current activations are sparse/ambiguous, but connected inputs carry tactical/protection/capture semantics ({upstream}); mark as low-confidence Tac for review.",
            evidence,
        )
    if squares and len(square_counts) <= 10:
        return (
            "Mov",
            0.64,
            f"Activated squares form a sparse reachable/check-square style pattern rather than fixed occupancy; connected inputs were checked ({upstream}), so Mov is the safest primitive label.",
            evidence,
        )
    return (
        "Uninterpretable",
        0.52,
        f"Activated squares, move evidence, z-pattern, and connected inputs ({upstream}) do not yield a stable chess relation.",
        evidence,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    circuit = load_json(args.circuit)
    rows = load_jsonl(args.evidence)
    node_by_id = {node.get("node_id"): node for node in circuit.get("nodes", [])}
    evidence_by_node = {row.get("node_id"): row for row in rows}
    links_by_target: dict[str, list[dict[str, Any]]] = {}
    for link in circuit.get("links", []):
        links_by_target.setdefault(link.get("target"), []).append(link)

    proposals = []
    conflicts = []
    by_identity: dict[tuple[str, int], str] = {}
    for row in rows:
        node = node_by_id.get(row.get("node_id"), {})
        existing_label = taxonomy_from_existing(row.get("existing_interpretation"))
        layer = int(row.get("layer") or 0)
        upstream = incoming_summary(row.get("node_id"), links_by_target, evidence_by_node)

        if existing_label and not (existing_label in {"Src", "Tgt"} and layer < 8):
            label = existing_label
            confidence = 0.99
            rationale = f"Existing interpretation [{existing_label}]."
            evidence_summary = (
                f"Existing interpretation retained as a hard constraint; node_id={row.get('node_id')}, "
                f"ctx_idx={node.get('ctx_idx')}, activation={node.get('activation')}."
            )
        else:
            label, confidence, rationale, evidence_summary = classify_fresh(row, node, upstream)
            if existing_label in {"Src", "Tgt"} and layer < 8:
                rationale = f"Existing early-layer [{existing_label}] is invalid under the layer prior. {rationale}"

        identity = (row.get("dictionary_name"), int(row.get("feature_index")))
        prior = by_identity.get(identity)
        if prior and prior != label:
            conflicts.append({"dictionary_name": identity[0], "feature_index": identity[1], "labels": sorted({prior, label})})
        else:
            by_identity[identity] = label

        proposals.append(
            {
                "directory_id": row.get("directory_id"),
                "file_name": row.get("file_name"),
                "circuit_index": row.get("circuit_index"),
                "feature_index_in_circuit": row.get("feature_index_in_circuit"),
                "node_id": row.get("node_id"),
                "dictionary_name": row.get("dictionary_name"),
                "feature_index": row.get("feature_index"),
                "layer": row.get("layer"),
                "feature_type": row.get("feature_type"),
                "taxonomy": label,
                "confidence": round(float(confidence), 3),
                "rationale": rationale,
                "evidence_summary": evidence_summary,
                "metadata": {
                    "ctx_idx": node.get("ctx_idx"),
                    "activation": node.get("activation"),
                    "influence": node.get("influence"),
                    "existing_interpretation": row.get("existing_interpretation") or "",
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in proposals) + "\n", encoding="utf-8")

    summary = {
        "output": str(args.output),
        "features": len(proposals),
        "taxonomy_counts": Counter(item["taxonomy"] for item in proposals),
        "duplicate_label_conflicts": conflicts,
    }
    summary["taxonomy_counts"] = dict(summary["taxonomy_counts"])
    if args.summary_output:
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
