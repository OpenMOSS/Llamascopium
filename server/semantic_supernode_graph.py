from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_present(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_position(data: dict[str, Any] | None) -> dict[str, Any]:
    data = data or {}
    board_square = _first_present(data, "board_square", "square")
    board_squares = _as_list(_first_present(data, "board_squares", "squares"))
    if board_square and board_square not in board_squares:
        board_squares.insert(0, board_square)
    return {
        "token_position": _to_int(_first_present(data, "token_position", "ctx_idx", "position")),
        "board_square": board_square,
        "board_squares": [str(square) for square in board_squares if square],
        "rank_file": _first_present(data, "rank_file"),
        "label": _first_present(data, "label"),
    }


def _normalize_child_feature(raw: Any, index: int, warnings: list[str]) -> dict[str, Any] | None:
    if isinstance(raw, str):
        return {
            "feature_id": raw,
            "node_id": raw,
            "layer": None,
            "feature_index": None,
            "feature_type": None,
            "dictionary_name": None,
            "token_position": None,
            "board_square": None,
            "interpretation": "",
            "taxonomy": None,
            "metadata": {},
        }
    if not isinstance(raw, dict):
        warnings.append(f"Skipped child feature #{index}: expected object/string, got {type(raw).__name__}")
        return None

    feature_id = _first_present(raw, "feature_id", "node_id", "id")
    if feature_id is None:
        dictionary_name = _first_present(raw, "dictionary_name", "sae_name", "sae")
        feature_index = _first_present(raw, "feature_index", "feature", "active_feature_idx")
        token_position = _first_present(raw, "token_position", "ctx_idx", "position")
        if dictionary_name is not None and feature_index is not None:
            feature_id = f"{dictionary_name}#{feature_index}@{token_position if token_position is not None else 'na'}"
        else:
            feature_id = f"feature-{index}"
            warnings.append(f"Child feature #{index} had no id; assigned {feature_id}")

    position = _normalize_position(raw.get("position") if isinstance(raw.get("position"), dict) else raw)
    return {
        "feature_id": str(feature_id),
        "node_id": str(_first_present(raw, "node_id", "id", default=feature_id)),
        "layer": _to_int(_first_present(raw, "layer", "layer_index")),
        "feature_index": _to_int(_first_present(raw, "feature_index", "feature", "active_feature_idx")),
        "feature_type": _first_present(raw, "feature_type", "type"),
        "dictionary_name": _first_present(raw, "dictionary_name", "sae_name", "sae"),
        "token_position": position["token_position"],
        "board_square": position["board_square"],
        "board_squares": position["board_squares"],
        "interpretation": _first_present(raw, "interpretation", "label", "clerp", default=""),
        "taxonomy": _first_present(raw, "taxonomy", "role", "semantic_type"),
        "activation": _to_float(_first_present(raw, "activation", "feature_act", "value")),
        "metadata": raw.get("metadata", {}),
        "raw": raw,
    }


def _normalize_supernode(
    raw: Any,
    index: int,
    child_feature_by_id: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any] | None:
    if isinstance(raw, list):
        if not raw:
            warnings.append(f"Skipped supernode #{index}: empty list")
            return None
        raw = {
            "label": raw[0],
            "member_feature_ids": [str(item) for item in raw[1:]],
        }
    if not isinstance(raw, dict):
        warnings.append(f"Skipped supernode #{index}: expected object/list, got {type(raw).__name__}")
        return None

    supernode_id = str(_first_present(raw, "supernode_id", "id", default=f"sn_{index:03d}"))
    inline_features = [
        item
        for item in (
            _as_list(raw.get("child_features")) + _as_list(raw.get("features")) + _as_list(raw.get("member_features"))
        )
        if isinstance(item, (dict, str))
    ]
    normalized_inline_features: list[dict[str, Any]] = []
    for offset, feature in enumerate(inline_features):
        normalized = _normalize_child_feature(feature, len(child_feature_by_id) + offset, warnings)
        if normalized:
            child_feature_by_id.setdefault(normalized["feature_id"], normalized)
            normalized_inline_features.append(normalized)

    member_ids = [
        str(item)
        for item in (
            _as_list(_first_present(raw, "member_feature_ids", "child_feature_ids", "member_node_ids"))
            + [feature["feature_id"] for feature in normalized_inline_features]
        )
        if item is not None
    ]
    member_ids = list(dict.fromkeys(member_ids))

    position_source = raw.get("position") if isinstance(raw.get("position"), dict) else raw
    position = _normalize_position(position_source)
    if not position["token_position"]:
        member_positions = [
            child_feature_by_id[member_id].get("token_position")
            for member_id in member_ids
            if member_id in child_feature_by_id and child_feature_by_id[member_id].get("token_position") is not None
        ]
        if member_positions:
            position["token_position"] = round(sum(member_positions) / len(member_positions))
    if not position["board_squares"]:
        seen_squares: list[str] = []
        for member_id in member_ids:
            feature = child_feature_by_id.get(member_id)
            if not feature:
                continue
            for square in feature.get("board_squares") or [feature.get("board_square")]:
                if square and square not in seen_squares:
                    seen_squares.append(square)
        position["board_squares"] = seen_squares
        position["board_square"] = seen_squares[0] if seen_squares else None

    return {
        "supernode_id": supernode_id,
        "label": str(_first_present(raw, "label", "name", "interpretation", default=supernode_id)),
        "semantic_type": _first_present(raw, "semantic_type", "taxonomy", "type", default="Semantic"),
        "role_id": _first_present(raw, "role_id", "role"),
        "interpretation": _first_present(raw, "interpretation", "description", default=""),
        "position": position,
        "member_feature_ids": member_ids,
        "evidence": raw.get("evidence", {}),
        "visual": raw.get("visual", {}),
        "metadata": raw.get("metadata", {}),
        "raw": raw,
    }


def _normalize_edge(raw: Any, index: int, warnings: list[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        warnings.append(f"Skipped edge #{index}: expected object, got {type(raw).__name__}")
        return None
    source = _first_present(raw, "source", "source_supernode_id", "from")
    target = _first_present(raw, "target", "target_supernode_id", "to")
    if source is None or target is None:
        warnings.append(f"Skipped edge #{index}: missing source/target")
        return None
    return {
        "edge_id": str(_first_present(raw, "edge_id", "id", default=f"edge_{index:03d}")),
        "source": str(source),
        "target": str(target),
        "weight": _to_float(_first_present(raw, "weight", "pct_input", "score")),
        "label": _first_present(raw, "label", "semantic_label", "interpretation", default=""),
        "relation_type": _first_present(raw, "relation_type", "type"),
        "member_edge_ids": [str(item) for item in _as_list(_first_present(raw, "member_edge_ids", "member_link_ids"))],
        "metadata": raw.get("metadata", {}),
        "raw": raw,
    }


def _normalize_child_feature_edge(raw: Any, index: int, warnings: list[str]) -> dict[str, Any] | None:
    edge = _normalize_edge(raw, index, warnings)
    if edge is None:
        return None
    return {
        **edge,
        "edge_id": str(_first_present(raw, "edge_id", "id", default=f"feature_edge_{index:03d}")),
    }


def normalize_semantic_supernode_graph(raw_graph: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_graph, dict):
        raise HTTPException(status_code=400, detail="Semantic supernode graph JSON must be an object")

    warnings: list[str] = []
    raw_child_features = _as_list(_first_present(raw_graph, "child_features", "features", "nodes", default=[]))
    child_features = [
        normalized
        for index, feature in enumerate(raw_child_features)
        if (normalized := _normalize_child_feature(feature, index, warnings)) is not None
    ]
    child_feature_by_id = {feature["feature_id"]: feature for feature in child_features}

    raw_supernodes = _as_list(
        _first_present(raw_graph, "semantic_supernodes", "supernodes", "nodes_semantic", default=[])
    )
    supernodes = [
        normalized
        for index, supernode in enumerate(raw_supernodes)
        if (normalized := _normalize_supernode(supernode, index, child_feature_by_id, warnings)) is not None
    ]

    raw_edges = _as_list(_first_present(raw_graph, "semantic_edges", "supernode_edges", "edges", "links", default=[]))
    edges = [
        normalized
        for index, edge in enumerate(raw_edges)
        if (normalized := _normalize_edge(edge, index, warnings)) is not None
    ]
    raw_child_feature_edges = _as_list(
        _first_present(raw_graph, "child_feature_edges", "feature_edges", "internal_feature_edges", default=[])
    )
    child_feature_edges = [
        normalized
        for index, edge in enumerate(raw_child_feature_edges)
        if (normalized := _normalize_child_feature_edge(edge, index, warnings)) is not None
    ]

    supernode_ids = {supernode["supernode_id"] for supernode in supernodes}
    for edge in edges:
        if edge["source"] not in supernode_ids:
            warnings.append(f"Edge {edge['edge_id']} source {edge['source']} is not a known supernode")
        if edge["target"] not in supernode_ids:
            warnings.append(f"Edge {edge['edge_id']} target {edge['target']} is not a known supernode")

    for supernode in supernodes:
        missing_features = [
            feature_id for feature_id in supernode["member_feature_ids"] if feature_id not in child_feature_by_id
        ]
        if missing_features:
            warnings.append(
                f"Supernode {supernode['supernode_id']} references {len(missing_features)} unknown child feature(s)"
            )
    child_feature_ids = set(child_feature_by_id)
    for edge in child_feature_edges:
        if edge["source"] not in child_feature_ids:
            warnings.append(
                f"Child feature edge {edge['edge_id']} source {edge['source']} is not a known child feature"
            )
        if edge["target"] not in child_feature_ids:
            warnings.append(
                f"Child feature edge {edge['edge_id']} target {edge['target']} is not a known child feature"
            )

    return {
        "schema_version": "semantic_supernode_graph.v1",
        "metadata": raw_graph.get("metadata", {}),
        "board": raw_graph.get("board", {}),
        "action": raw_graph.get("action", {}),
        "child_features": list(child_feature_by_id.values()),
        "child_feature_edges": child_feature_edges,
        "semantic_supernodes": supernodes,
        "semantic_edges": edges,
        "warnings": warnings,
        "raw": raw_graph,
    }


def _safe_resolve_repo_path(repo_root: Path, requested_path: str) -> Path:
    path = Path(requested_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    resolved = path.resolve()
    repo_root_resolved = repo_root.resolve()
    if repo_root_resolved not in [resolved, *resolved.parents]:
        raise HTTPException(status_code=400, detail="Path must be inside the repository")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"JSON file not found: {requested_path}")
    return resolved


def get_semantic_supernode_graph_router(repo_root: Path) -> APIRouter:
    router = APIRouter(prefix="/semantic_supernode_graph", tags=["semantic-supernode-graph"])

    @router.post("/normalize")
    async def normalize_graph(raw_graph: dict[str, Any]) -> dict[str, Any]:
        return normalize_semantic_supernode_graph(raw_graph)

    @router.post("/parse")
    async def parse_graph_file(file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            content = await file.read()
            raw_graph = json.loads(content.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Uploaded file must be UTF-8 JSON") from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
        return normalize_semantic_supernode_graph(raw_graph)

    @router.get("/from_path")
    async def graph_from_path(path: str = Query(..., description="JSON path relative to repo root")) -> dict[str, Any]:
        resolved = _safe_resolve_repo_path(repo_root, path)
        try:
            raw_graph = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in {path}: {exc}") from exc
        graph = normalize_semantic_supernode_graph(raw_graph)
        graph["metadata"] = {
            **graph.get("metadata", {}),
            "source_path": str(resolved.relative_to(repo_root.resolve())),
        }
        return graph

    @router.get("/example")
    async def example_graph() -> dict[str, Any]:
        example_path = (
            repo_root
            / "scripts"
            / "Attribution_Graph"
            / "semantic_supernode_examples"
            / "looking_ahead_semantic_graph.json"
        )
        if example_path.exists():
            raw_graph = json.loads(example_path.read_text(encoding="utf-8"))
            return normalize_semantic_supernode_graph(raw_graph)
        return normalize_semantic_supernode_graph(
            {
                "metadata": {"title": "Minimal semantic supernode graph example"},
                "semantic_supernodes": [
                    {"supernode_id": "sn_det", "label": "OwnKnight detector", "semantic_type": "Det"},
                    {"supernode_id": "sn_mov", "label": "OwnKnight movement", "semantic_type": "Mov"},
                ],
                "semantic_edges": [{"source": "sn_det", "target": "sn_mov", "label": "feeds movement"}],
            }
        )

    return router
