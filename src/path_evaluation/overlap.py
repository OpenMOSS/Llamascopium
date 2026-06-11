"""Pairwise Jaccard overlap between comparable feature files in one folder."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

FeatureKey = Tuple[int, int, int, str]
"""One row in CSV: (position_idx, layer, feature_id, feature_type)."""

WithinLayerKey = Tuple[int, int, str]
"""Feature identity inside a fixed layer: (position_idx, feature_id, feature_type)."""


def _get_feature_set_from_csv(csv_path: Path) -> Set[FeatureKey]:
    """Read a feature CSV and convert rows into a set of comparable feature keys."""
    features: Set[FeatureKey] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.add(
                (
                    int(row["position_idx"]),
                    int(row["layer"]),
                    int(row["feature_id"]),
                    str(row["feature_type"]),
                )
            )
    return features


def _get_feature_set_from_circuit_json(json_path: Path) -> Set[FeatureKey]:
    """Read a circuit JSON and convert its ``nodes`` into comparable feature keys."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])

    features: Set[FeatureKey] = set()
    for node in nodes:
        features.add(
            (
                int(node["ctx_idx"]),
                int(node["layer"]),
                int(node["feature"]),
                str(node["feature_type"]),
            )
        )
    return features


def _subset_for_layer(
    feature_set: Set[FeatureKey], layer: int
) -> Set[WithinLayerKey]:
    """Project features to within-layer keys (position_idx, feature_id, feature_type)."""
    return {(pos, fid, ftype) for pos, lyr, fid, ftype in feature_set if lyr == layer}


def _compute_pair_overlap(a: Set[FeatureKey], b: Set[FeatureKey]) -> float:
    """Compute Jaccard overlap between two feature sets."""
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


def _compute_pair_overlap_for_layer(
    a: Set[FeatureKey], b: Set[FeatureKey], layer: int
) -> float:
    """Jaccard overlap restricted to rows with the given ``layer`` index."""
    sa = _subset_for_layer(a, layer)
    sb = _subset_for_layer(b, layer)
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return inter / union


def _load_folder_feature_sets(folder_path: str) -> Optional[Dict[str, Set[FeatureKey]]]:
    """Load comparable files in a folder into ``name -> feature set``.

    Supported inputs:
    - legacy ``*_top*_features.csv``
    - circuit ``*.json`` files that contain a top-level ``nodes`` list

    At least 2 comparable files are required.
    """
    folder = Path(folder_path)
    csv_files: List[Path] = sorted(
        folder.glob("*_top*_features.csv"), key=lambda p: p.name
    )
    if len(csv_files) < 2:
        json_files: List[Path] = []
        for path in sorted(folder.glob("*.json"), key=lambda p: p.name):
            if path.name == "sample_info.json":
                continue

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            if isinstance(data.get("nodes"), list):
                json_files.append(path)

        if len(json_files) < 2:
            return None
        return {f.name: _get_feature_set_from_circuit_json(f) for f in json_files}

    return {f.name: _get_feature_set_from_csv(f) for f in csv_files}


def _all_layers_in_sets(feature_sets: Dict[str, Set[FeatureKey]]) -> Set[int]:
    layers: Set[int] = set()
    for feats in feature_sets.values():
        for _, lyr, _, _ in feats:
            layers.add(lyr)
    return layers


def compute_folder_overlap(folder_path: str) -> float:
    """Return the mean pairwise overlap among all comparable files in a folder."""
    _, mean_overlap = compute_folder_overlap_details(folder_path)
    return mean_overlap


def compute_folder_overlap_details(
    folder_path: str,
) -> Tuple[List[Tuple[str, str, float]], float]:
    """Return all pairwise overlaps and their mean for supported files in a folder.

    A feature is considered identical iff
    ``(position_idx, layer, feature_id, feature_type)`` are equal.
    For circuit JSONs this maps to
    ``(ctx_idx, layer, feature, feature_type)`` on each node.
    """
    feature_sets = _load_folder_feature_sets(folder_path)
    if feature_sets is None:
        return [], float("nan")

    pair_overlaps: List[Tuple[str, str, float]] = []
    for (name_a, set_a), (name_b, set_b) in combinations(feature_sets.items(), 2):
        pair_overlaps.append((name_a, name_b, _compute_pair_overlap(set_a, set_b)))

    mean_overlap = (
        float(sum(v for _, _, v in pair_overlaps) / len(pair_overlaps))
        if pair_overlaps
        else float("nan")
    )
    return pair_overlaps, mean_overlap


def compute_folder_overlap_per_layer_details(
    folder_path: str,
) -> Tuple[
    Dict[int, List[Tuple[str, str, float]]],
    Dict[int, float],
]:
    """Per-layer pairwise overlaps and per-layer mean."""
    feature_sets = _load_folder_feature_sets(folder_path)
    if feature_sets is None:
        return {}, {}

    layers = sorted(_all_layers_in_sets(feature_sets))
    by_layer: Dict[int, List[Tuple[str, str, float]]] = {lyr: [] for lyr in layers}
    items = list(feature_sets.items())

    for layer in layers:
        for (name_a, set_a), (name_b, set_b) in combinations(items, 2):
            o = _compute_pair_overlap_for_layer(set_a, set_b, layer)
            by_layer[layer].append((name_a, name_b, o))

    means: Dict[int, float] = {
        lyr: float(sum(t[2] for t in pairs) / len(pairs))
        for lyr, pairs in by_layer.items()
        if pairs
    }
    return by_layer, means


def compute_folder_overlap_per_layer(folder_path: str) -> Dict[int, float]:
    """Mean pairwise Jaccard overlap within each layer."""
    _, means = compute_folder_overlap_per_layer_details(folder_path)
    return means
