"""Upstream traversal for observational inhibitory feature connections."""

from dataclasses import dataclass

from llamascopium.circuits.global_weights.inhibitory import Feature, InhibitoryConnection


@dataclass
class _Node:
    priority: float
    visited: bool = False


class InhibitoryAtlas:
    """Explore upstream features by non-negative score; export source -> target edges."""

    def __init__(self, target: Feature):
        self.nodes = {target: _Node(priority=1.0, visited=True)}
        self.links: dict[tuple[Feature, Feature], InhibitoryConnection] = {}

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
            priority = self.nodes[target].priority * edge.score
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
