import torch
from types import SimpleNamespace

from lm_saes.circuit.attribution_qk import (
    AttributionContext,
    _normalize_rows,
    compute_partial_influences,
    merge_qk_graph,
    run_joint_feature_attribution,
)
from lm_saes.circuit.graph_lc0 import compute_influence, normalize_matrix
from lm_saes.circuit.utils.create_graph_files import create_used_nodes_and_edges


def _qk_test_context(batch_size=4):
    ctx = object.__new__(AttributionContext)
    q = torch.arange(batch_size * 3.0).reshape(batch_size, 3, 1).requires_grad_()
    k = (torch.arange(batch_size * 3.0).reshape(batch_size, 3, 1) + 10).requires_grad_()
    ctx._policy_q_activations = q
    ctx._policy_k_activations = k

    def rows_from_root(root, *, retain_graph):
        q_grad, k_grad = torch.autograd.grad(root, (q, k), retain_graph=retain_graph)
        return q_grad.sum((1, 2), keepdim=False)[:, None] + k_grad.sum((1, 2))[:, None]

    ctx._rows_from_root = rows_from_root
    return ctx, q, k


def test_qk_roots_share_one_vjp_without_cross_lane_terms():
    ctx, _, _ = _qk_test_context()
    rows_q, rows_k = ctx.compute_qk_vjp_batch(
        q_positions=torch.tensor([[0], [1]]),
        k_positions=torch.tensor([[1], [2]]),
        q_values=torch.tensor([[[2.0], [0.0], [0.0]], [[0.0], [3.0], [0.0]]]),
        k_values=torch.tensor([[[0.0], [5.0], [0.0]], [[0.0], [0.0], [7.0]]]),
    )

    assert torch.equal(rows_q, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(rows_k, torch.tensor([[5.0], [7.0]]))


def test_policy_positive_and_negative_logits_use_distinct_lanes():
    ctx, q, k = _qk_test_context()
    logits = torch.cat((q.squeeze(-1), k.squeeze(-1)), dim=1)
    q_pos, k_pos, q_neg, k_neg = ctx.compute_policy_qk_gradients(
        logits,
        positive_indices=torch.tensor([1]),
        negative_indices=torch.tensor([4]),
    )

    assert q_pos[0, 1, 0] == 1
    assert k_pos.abs().sum() == 0
    assert q_neg.abs().sum() == 0
    assert k_neg[0, 1, 0] == -1


def test_pre_normalized_influence_matches_raw_matrix():
    matrix = torch.tensor(
        [
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    row_to_node = torch.tensor([2, 3])
    probabilities = torch.tensor([1.0])

    raw = compute_partial_influences(matrix, probabilities, row_to_node, max_iter=4)
    cached = compute_partial_influences(
        _normalize_rows(matrix),
        probabilities,
        row_to_node,
        max_iter=4,
        pre_normalized=True,
    )

    assert torch.allclose(raw, cached)


def test_graph_influence_promotes_mixed_precision_edges_to_float32():
    matrix = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    normalized = normalize_matrix(matrix)
    influence = compute_influence(normalized, torch.tensor([1.0, 0.0, 0.0]))

    assert normalized.dtype == torch.float32
    assert influence.dtype == torch.float32
    assert torch.equal(influence, torch.tensor([0.0, 1.0, 1.0]))


def test_graph_edge_export_accepts_bfloat16_weights():
    graph = SimpleNamespace(
        adjacency_matrix=torch.tensor([[0.0, 0.0], [1.5, 0.0]], dtype=torch.bfloat16)
    )
    nodes = {
        0: SimpleNamespace(node_id="source", feature_type="feature"),
        1: SimpleNamespace(node_id="target", feature_type="logit"),
    }
    used_nodes, used_edges = create_used_nodes_and_edges(
        graph,
        nodes,
        torch.tensor([[False, False], [True, False]]),
    )

    assert len(used_nodes) == 2
    assert used_edges == [{"source": "source", "target": "target", "weight": 1.5}]


def test_gpu_style_merge_uses_tensor_alignment_and_index_add():
    q = {
        "selected_features": torch.tensor([0, 2]),
        "col_read": torch.tensor([0, 2, 3]),
        "edge_matrix": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        "row_to_node_index": torch.tensor([0, 2, 3]),
        "move_positions": torch.tensor([1]),
    }
    k = {
        "selected_features": torch.tensor([1, 2]),
        "col_read": torch.tensor([1, 2, 3]),
        "edge_matrix": torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]),
        "row_to_node_index": torch.tensor([1, 2, 3]),
        "move_positions": torch.tensor([2]),
    }
    result = merge_qk_graph(
        {
            "q": q,
            "k": k,
            "dims": {"total_active_feats": 3, "logit_offset": 3},
            "logits": {"n_logits": 1},
        }
    )

    assert torch.equal(result["selected_features"], torch.tensor([0, 1, 2]))
    expected = torch.tensor(
        [
            [1.0, 0.0, 2.0, 3.0],
            [0.0, 10.0, 20.0, 30.0],
            [4.0, 40.0, 55.0, 66.0],
            [7.0, 70.0, 88.0, 99.0],
        ]
    )
    assert torch.equal(result["adjacency_matrix"], expected)


def test_merge_ignores_unfilled_and_filtered_feature_rows():
    q = {
        "selected_features": torch.tensor([1]),
        "col_read": torch.tensor([1, 3]),
        "edge_matrix": torch.tensor([[2.0, 3.0], [9.0, 9.0], [4.0, 5.0]]),
        "row_to_node_index": torch.tensor([1, 0, 3]),
        "move_positions": torch.tensor([1]),
    }
    k = {
        "selected_features": torch.tensor([1]),
        "col_read": torch.tensor([1, 3]),
        "edge_matrix": torch.tensor([[6.0, 7.0], [8.0, 8.0], [10.0, 11.0]]),
        "row_to_node_index": torch.tensor([1, 4, 3]),
        "move_positions": torch.tensor([2]),
    }
    result = merge_qk_graph(
        {
            "q": q,
            "k": k,
            "dims": {"total_active_feats": 3, "logit_offset": 3},
            "logits": {"n_logits": 1},
        }
    )

    assert torch.equal(result["adjacency_matrix"], torch.tensor([[8.0, 10.0], [14.0, 16.0]]))


class _CountingContext:
    n_layers = 1

    def __init__(self):
        self.calls = 0

    def compute_vjp_batch(self, layers, positions, inject_values, attention_patterns, retain_graph):
        self.calls += 1
        gids = inject_values[:, 0]
        return torch.stack((gids + 1, gids + 2, gids + 3), dim=1)


def test_joint_qk_feature_trace_reuses_rows_between_sides():
    ctx = _CountingContext()
    matrices = {side: torch.zeros(3, 4) for side in ("q", "k")}
    normalized = {side: torch.zeros_like(matrix) for side, matrix in matrices.items()}
    row_indices = {side: torch.tensor([3, 0, 0]) for side in ("q", "k")}

    result = run_joint_feature_attribution(
        ctx=ctx,
        requested_sides=("q", "k"),
        edge_matrices=matrices,
        normalized_matrices=normalized,
        row_to_node_indices=row_indices,
        total_active_feats=2,
        max_feature_nodes=2,
        update_interval=1,
        selection_batch_size=2,
        vjp_batch_size=2,
        n_logits=1,
        logit_p=torch.ones(1),
        logit_offset=3,
        idx_to_layer=lambda gids: torch.zeros_like(gids),
        idx_to_pos=lambda gids: torch.zeros_like(gids),
        idx_to_encoder_rows=lambda gids: gids.float().unsqueeze(1),
        idx_to_pattern=lambda gids: torch.ones(gids.numel(), 1),
        order_mode="abs",
    )

    assert ctx.calls == 1
    assert torch.equal(result["q"]["edge_matrix"], result["k"]["edge_matrix"])
    assert result["q"]["visited"].all()
    assert result["k"]["visited"].all()
