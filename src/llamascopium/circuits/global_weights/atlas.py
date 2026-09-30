"""Upstream traversal for observational inhibitory feature connections."""

from dataclasses import dataclass
from typing import Callable

from llamascopium.circuits.global_weights.inhibitory import Feature, InhibitoryConnection, InhibitoryWeights


@dataclass
class _Node:
    priority: float
    visited: bool = False


class InhibitoryAtlas:
    """Explore upstream features by non-negative score; export source -> target edges."""

    def __init__(self, target: Feature, normalized: bool = False):
        self.nodes = {target: _Node(priority=1.0, visited=True)}
        self.links: dict[tuple[Feature, Feature], InhibitoryConnection] = {}
        self.normalized = normalized

    def update(self, target: Feature, connections: list[InhibitoryConnection]) -> None:
        if target not in self.nodes or not self.nodes[target].visited:
            raise ValueError(f"Target has not been selected for exploration: {target}")
        for edge in connections:
            if (edge.target, edge.target_feature) != target or edge.kind != "inhibitory" or edge.score < 0:
                raise ValueError("Expected a non-negative inhibitory edge into the current target")
            source = (edge.source, edge.source_feature)
            directed_edge = (source, target)
            if directed_edge in self.links:
                continue
            self.links[directed_edge] = edge
            weight = edge.normalized_inhibitory_score if self.normalized else edge.score
            priority = self.nodes[target].priority * weight
            if source not in self.nodes:
                self.nodes[source] = _Node(priority=priority)
            else:
                self.nodes[source].priority += priority

    def select_top_k_nodes_to_visit(self, k: int) -> list[Feature]:
        if k < 0:
            raise ValueError("k must be nonnegative")
        selected = sorted(
            (feature for feature, node in self.nodes.items() if not node.visited),
            key=lambda feature: self.nodes[feature].priority,
            reverse=True,
        )[:k]
        for feature in selected:
            self.nodes[feature].visited = True
        return selected

    def export_to_dict(self) -> dict:
        return {
            "nodes": [
                {"name": name, "feature": index, "priority": node.priority}
                for (name, index), node in self.nodes.items()
            ],
            "links": [
                {
                    "source": source,
                    "target": target,
                    "kind": edge.kind,
                    "score": edge.score,
                    "delta": edge.delta,
                    "virtual_weight": edge.virtual_weight,
                    "mu_on": edge.mu_on,
                    "mu_off": edge.mu_off,
                }
                for (source, target), edge in self.links.items()
            ],
        }


def search_inhibitory_atlas(
    target: Feature,
    compute: Callable[[list[Feature]], InhibitoryWeights],
    *,
    depth: int,
    top_k: int,
    expansion_width: int,
    normalized: bool = False,
    lorsa_k: int = 1,
) -> tuple[InhibitoryAtlas, InhibitoryWeights]:
    """Expand the strongest upstream nodes in batches, one model pass per level."""
    if depth < 1 or top_k < 1 or expansion_width < 1 or lorsa_k < 0:
        raise ValueError("depth, top_k, and expansion_width must be positive; lorsa_k must be nonnegative")
    atlas = InhibitoryAtlas(target, normalized=normalized)
    frontier = [target]
    root_result = None
    for level in range(depth):
        result = compute(frontier)
        if root_result is None:
            root_result = result
        neighbor_k = top_k if level == 0 else min(top_k, expansion_width * 3)
        for current in frontier:
            atlas.update(current, result.topk(current, k=neighbor_k, normalized=normalized, lorsa_k=lorsa_k))
        if level + 1 == depth:
            break
        frontier = atlas.select_top_k_nodes_to_visit(expansion_width)
        if not frontier:
            break
    assert root_result is not None
    return atlas, root_result
