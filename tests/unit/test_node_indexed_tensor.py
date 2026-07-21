"""Unit tests for NodeIndexedTensor, NodeIndexedVector, and NodeIndexedMatrix.

Each axis of a :class:`~llamascopium.backend.language_model.NodeIndexedTensor` is
described by a :class:`~llamascopium.backend.language_model.NodeDimension` (node layout,
offsets, inverse indices). Construction uses ``from_dimensions`` /
``from_data(..., dimensions=...)``. Selector-style APIs accept either a
sequence of :class:`NodeInfo` or a prebuilt :class:`NodeDimension`, while
``nodes_to_offsets`` and ``topk(..., ignore_dimension=...)`` operate on
:class:`NodeDimension` directly.

Index convention
----------------
``NodeInfo.indices`` must be a 2-D tensor of shape ``(n_elements, d_index)``.
For the common 1-D logical-key case (e.g. a token position) that means shape
``(n, 1)``, which is what the helper :func:`_ni` produces.  For a 2-D logical
key (e.g. (token_pos, feature_idx)) use :func:`_ni2`.
"""

import pytest
import torch

from llamascopium.circuits.attribution import prune_attribution
from llamascopium.circuits.indexed_tensor import (
    NodeDimension,
    NodeIndexedMatrix,
    NodeIndexedTensor,
    NodeIndexedVector,
    NodeInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ni(key: str, *positions: int) -> NodeInfo:
    """Build a NodeInfo with 2-D indices ``(n_elements, 1)`` from plain ints."""
    return NodeInfo(key=key, indices=torch.tensor([[p] for p in positions], dtype=torch.long))


def _ni2(key: str, *pairs: tuple[int, int]) -> NodeInfo:
    """Build a NodeInfo with 2-D indices ``(n_elements, 2)`` from (a, b) pairs."""
    return NodeInfo(key=key, indices=torch.tensor(list(pairs), dtype=torch.long))


def _dim(*node_infos: NodeInfo) -> NodeDimension:
    return NodeDimension.from_node_infos(list(node_infos))


def test_prune_attribution_preserves_required_node_path():
    targets = _dim(_ni("logits", 0))
    required = _dim(_ni2("matry", (1, 3000)))
    other = _dim(_ni2("other", (1, 7)))
    rows = targets + required + other
    sources = _dim(_ni("hook_embed", 0), _ni("blocks.0.error", 0)) + required + other
    attribution = NodeIndexedMatrix.from_data(
        torch.tensor(
            [
                [0.0, 0.0, 0.01, 100.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
        dimensions=(rows, sources),
    )

    pruned = prune_attribution(
        attribution,
        reduction_weight=torch.ones(1),
        node_threshold=0.6,
        edge_threshold=0.6,
        targets=targets,
        required_nodes=required,
    )

    target_row = rows.nodes_to_offsets(targets).item()
    required_row = rows.nodes_to_offsets(required).item()
    required_source = sources.nodes_to_offsets(required).item()
    embed_source = sources.nodes_to_offsets(_dim(_ni("hook_embed", 0))).item()
    assert pruned.data[target_row, required_source] != 0
    assert pruned.data[required_row, embed_source] != 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_dimensions_allocates_zeros_of_correct_size():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1)],),
    )
    assert t.data.shape == (3,)
    assert torch.all(t.data == 0)


def test_from_data_stores_values_exactly():
    data = torch.tensor([10.0, 20.0, 30.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=([_ni("a", 0, 3), _ni("b", 7)],),
    )
    assert torch.equal(t.data, data)


def test_from_data_accepts_prebuilt_dimension_instance():
    """``from_data`` may take a :class:`NodeDimension` instead of a node list."""
    node_list = [_ni("a", 0, 3), _ni("b", 7)]
    d0 = NodeDimension.from_node_infos(node_list)
    data = torch.tensor([10.0, 20.0, 30.0])
    t = NodeIndexedTensor.from_data(data=data, dimensions=(d0,))
    assert t.dimensions[0] is d0
    assert torch.equal(t.data, data)


def test_from_dimensions_accepts_prebuilt_dimension_instance():
    d0 = NodeDimension.from_node_infos([_ni("a", 0, 3), _ni("b", 1)])
    t = NodeIndexedTensor.from_dimensions(dimensions=(d0,))
    assert t.dimensions[0] is d0
    assert t.data.shape == (3,)


def test_node_mappings_store_correct_offsets():
    # "a" is registered first → data offsets 0, 1
    # "b" is registered next  → data offset  2
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 7)],),
    )
    assert isinstance(t.dimensions[0], NodeDimension)
    assert torch.equal(t.dimensions[0].node_mappings["a"].offsets, torch.tensor([0, 1]))
    assert torch.equal(t.dimensions[0].node_mappings["b"].offsets, torch.tensor([2]))


def test_node_mappings_store_correct_indices():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 7)],),
    )
    assert torch.equal(t.dimensions[0].node_mappings["a"].indices, torch.tensor([[0], [3]]))
    assert torch.equal(t.dimensions[0].node_mappings["b"].indices, torch.tensor([[7]]))


# ---------------------------------------------------------------------------
# NodeDimension.nodes_to_offsets
# ---------------------------------------------------------------------------


def test_nodes_to_offsets_full_first_node():
    # "a" → offsets 0, 1 ;  "b" → offsets 2, 3
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    assert torch.equal(t.dimensions[0].nodes_to_offsets(_dim(_ni("a", 0, 3))), torch.tensor([0, 1]))


def test_nodes_to_offsets_full_second_node():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    assert torch.equal(t.dimensions[0].nodes_to_offsets(_dim(_ni("b", 1, 2))), torch.tensor([2, 3]))


def test_nodes_to_offsets_partial_first_element():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    assert torch.equal(t.dimensions[0].nodes_to_offsets(_dim(_ni("a", 0))), torch.tensor([0]))


def test_nodes_to_offsets_partial_second_element():
    # logical index 3 is the *second* element of "a" → data offset 1
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    assert torch.equal(t.dimensions[0].nodes_to_offsets(_dim(_ni("a", 3))), torch.tensor([1]))


def test_nodes_to_offsets_cross_node_selection():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    result = t.dimensions[0].nodes_to_offsets(_dim(_ni("a", 3), _ni("b", 1)))
    assert torch.equal(result, torch.tensor([1, 2]))


def test_nodes_to_offsets_accepts_dimension_matching_node_info_list():
    """``NodeDimension.nodes_to_offsets`` accepts another :class:`NodeDimension` layout."""
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    axis = t.dimensions[0]
    selection = _dim(_ni("a", 3), _ni("b", 1))
    via_dimension = axis.nodes_to_offsets(selection)
    assert torch.equal(via_dimension, torch.tensor([1, 2]))


def test_nodes_to_offsets_dimension_branch_caches_by_dimension_hash():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1)],),
    )
    axis = t.dimensions[0]
    selection = _dim(_ni("a", 0))
    first = axis.nodes_to_offsets(selection)
    second = axis.nodes_to_offsets(selection)
    assert first is second


def test_nodes_to_offsets_merged_node_dimension_matches_list():
    """Selection :class:`NodeDimension` built from two :class:`NodeInfo` with the same key."""
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni("a", 0, 3, 7)],))
    axis = t.dimensions[0]
    selection = _dim(_ni("a", 7), _ni("a", 0))
    via_dimension = axis.nodes_to_offsets(selection)
    assert torch.equal(via_dimension, torch.tensor([2, 0]))


# ---------------------------------------------------------------------------
# NodeDimension.nodes_to_offsets — missing (key, index) pairs return -1
# ---------------------------------------------------------------------------


def test_nodes_to_offsets_unknown_key_returns_sentinel():
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni("a", 0, 3)],))
    result = t.dimensions[0].nodes_to_offsets(_dim(_ni("z", 0)))
    assert torch.equal(result, torch.tensor([-1]))


def test_nodes_to_offsets_out_of_bounds_index_returns_sentinel():
    # "a" has indices [0, 3] → inv_indices.shape == (4,). Index 7 is OOB.
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni("a", 0, 3)],))
    result = t.dimensions[0].nodes_to_offsets(_dim(_ni("a", 7)))
    assert torch.equal(result, torch.tensor([-1]))


def test_nodes_to_offsets_in_bounds_unregistered_returns_sentinel():
    # "a" has indices [0, 3] → inv_indices.shape == (4,). Index 1 is in-bounds but unregistered.
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni("a", 0, 3)],))
    result = t.dimensions[0].nodes_to_offsets(_dim(_ni("a", 1)))
    assert torch.equal(result, torch.tensor([-1]))


def test_nodes_to_offsets_mixed_valid_and_missing():
    # "a"→0,3 (offsets 0,1) ; "b"→1,2 (offsets 2,3). inv_indices.shape == (4,) for "a".
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    selection = _dim(_ni("a", 0), _ni("a", 1), _ni("a", 7), _ni("b", 2), _ni("z", 0))
    # a→0 hits offset 0 ; a→1 in-bounds-unregistered → -1 ; a→7 OOB → -1 ;
    # b→2 hits offset 3 ; z→0 unknown key → -1.
    assert torch.equal(t.dimensions[0].nodes_to_offsets(selection), torch.tensor([0, -1, -1, 3, -1]))


def test_nodes_to_offsets_2d_out_of_bounds_returns_sentinel():
    # _ni2("a", (0, 0), (1, 2)) → inv_indices.shape == (2, 3). (3, 0) is OOB on axis 0.
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni2("a", (0, 0), (1, 2))],))
    result = t.dimensions[0].nodes_to_offsets(_dim(_ni2("a", (3, 0))))
    assert torch.equal(result, torch.tensor([-1]))


def test_nodes_to_offsets_2d_in_bounds_unregistered_returns_sentinel():
    # inv_indices.shape == (2, 3) but only (0, 0) and (1, 2) are registered.
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni2("a", (0, 0), (1, 2))],))
    result = t.dimensions[0].nodes_to_offsets(_dim(_ni2("a", (0, 1))))
    assert torch.equal(result, torch.tensor([-1]))


# ---------------------------------------------------------------------------
# NodeDimension.__sub__ — remove (key, index) pairs
# ---------------------------------------------------------------------------


def test_sub_removes_subset_of_single_node():
    full = _dim(_ni("a", 0, 3, 7))
    to_remove = _dim(_ni("a", 3))
    result = (full - to_remove).node_infos
    assert len(result) == 1
    assert result[0].key == "a"
    assert torch.equal(result[0].indices, torch.tensor([[0], [7]]))


def test_sub_removes_entire_node():
    full = _dim(_ni("a", 0, 3), _ni("b", 1, 2))
    to_remove = _dim(_ni("b", 1, 2))
    result = (full - to_remove).node_infos
    assert len(result) == 1
    assert result[0].key == "a"
    assert torch.equal(result[0].indices, torch.tensor([[0], [3]]))


def test_sub_ignores_pairs_absent_from_self():
    full = _dim(_ni("a", 0, 3))
    # Unknown key, OOB index, and in-bounds-unregistered all silently ignored.
    to_remove = _dim(_ni("a", 1), _ni("a", 7), _ni("z", 0))
    result = (full - to_remove).node_infos
    assert len(result) == 1
    assert result[0].key == "a"
    assert torch.equal(result[0].indices, torch.tensor([[0], [3]]))


def test_sub_removes_across_multiple_nodes():
    full = _dim(_ni("a", 0, 3), _ni("b", 1, 2))
    to_remove = _dim(_ni("a", 3), _ni("b", 1))
    result = (full - to_remove).node_infos
    assert len(result) == 2
    assert result[0].key == "a"
    assert torch.equal(result[0].indices, torch.tensor([[0]]))
    assert result[1].key == "b"
    assert torch.equal(result[1].indices, torch.tensor([[2]]))


def test_sub_removing_everything_yields_empty_dimension():
    full = _dim(_ni("a", 0, 3))
    result = full - full
    assert len(result) == 0
    assert result.node_infos == []


def test_sub_yields_contiguous_offsets():
    """Result of __sub__ must have offsets re-numbered from 0 so it can index a tensor."""
    full = _dim(_ni("a", 0, 3, 7), _ni("b", 1, 2))
    to_remove = _dim(_ni("a", 3))
    result = full - to_remove
    # Remaining nodes: a→0 (offset 0), a→7 (1), b→1 (2), b→2 (3)
    assert len(result) == 4
    offsets = result.nodes_to_offsets(_dim(_ni("a", 0, 7), _ni("b", 1, 2)))
    assert torch.equal(offsets, torch.tensor([0, 1, 2, 3]))


def test_sub_with_2d_indices():
    full = _dim(_ni2("a", (0, 0), (0, 1), (1, 2)))
    to_remove = _dim(_ni2("a", (0, 1)))
    result = (full - to_remove).node_infos
    assert len(result) == 1
    assert result[0].key == "a"
    assert torch.equal(result[0].indices, torch.tensor([[0, 0], [1, 2]]))


def test_sub_is_inverse_of_add_for_disjoint_pieces():
    a = _dim(_ni("k", 0, 3))
    b = _dim(_ni("k", 7), _ni("m", 1))
    combined = a + b
    leftover = (combined - b).node_infos
    assert len(leftover) == 1
    assert leftover[0].key == "k"
    assert torch.equal(leftover[0].indices, torch.tensor([[0], [3]]))


# ---------------------------------------------------------------------------
# NodeDimension.offsets_to_nodes
# ---------------------------------------------------------------------------


def test_offsets_to_nodes_full_first_node():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    node_infos = t.dimensions[0].offsets_to_nodes(torch.tensor([0, 1])).node_infos
    assert len(node_infos) == 1
    assert node_infos[0].key == "a"
    assert torch.equal(node_infos[0].indices, torch.tensor([[0], [3]]))


def test_offsets_to_nodes_full_second_node():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    node_infos = t.dimensions[0].offsets_to_nodes(torch.tensor([2, 3])).node_infos
    assert len(node_infos) == 1
    assert node_infos[0].key == "b"
    assert torch.equal(node_infos[0].indices, torch.tensor([[1], [2]]))


def test_offsets_to_nodes_single_offset_second_element():
    # Offset 1 is the second element of "a": logical index 3
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    node_infos = t.dimensions[0].offsets_to_nodes(torch.tensor([1])).node_infos
    assert len(node_infos) == 1
    assert node_infos[0].key == "a"
    assert torch.equal(node_infos[0].indices, torch.tensor([[3]]))


def test_offsets_to_nodes_across_two_nodes():
    # Offset 0 → a→0,  offset 2 → b→1
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    node_infos = t.dimensions[0].offsets_to_nodes(torch.tensor([0, 2])).node_infos
    assert len(node_infos) == 2
    assert node_infos[0].key == "a"
    assert torch.equal(node_infos[0].indices, torch.tensor([[0]]))
    assert node_infos[1].key == "b"
    assert torch.equal(node_infos[1].indices, torch.tensor([[1]]))


# ---------------------------------------------------------------------------
# Duality: nodes_to_offsets ↔ offsets_to_nodes
# ---------------------------------------------------------------------------


def test_nodes_to_offsets_then_to_nodes_full_round_trip():
    # x: offsets 0,1,2 ; y: offsets 3,4
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("x", 5, 7, 9), _ni("y", 1, 3)],),
    )
    original = _dim(_ni("x", 5, 7, 9), _ni("y", 1, 3))
    offsets = t.dimensions[0].nodes_to_offsets(original)
    recovered = t.dimensions[0].offsets_to_nodes(offsets).node_infos
    assert len(recovered) == 2
    assert recovered[0].key == "x"
    assert torch.equal(recovered[0].indices, torch.tensor([[5], [7], [9]]))
    assert recovered[1].key == "y"
    assert torch.equal(recovered[1].indices, torch.tensor([[1], [3]]))


def test_nodes_to_offsets_then_to_nodes_partial_round_trip():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("x", 5, 7, 9), _ni("y", 1, 3)],),
    )
    original = _dim(_ni("x", 7), _ni("y", 1))
    offsets = t.dimensions[0].nodes_to_offsets(original)
    recovered = t.dimensions[0].offsets_to_nodes(offsets).node_infos
    assert len(recovered) == 2
    assert recovered[0].key == "x"
    assert torch.equal(recovered[0].indices, torch.tensor([[7]]))
    assert recovered[1].key == "y"
    assert torch.equal(recovered[1].indices, torch.tensor([[1]]))


def test_offsets_to_nodes_then_to_offsets_round_trip():
    # x[5]=0, x[7]=1, x[9]=2, y[1]=3, y[3]=4
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("x", 5, 7, 9), _ni("y", 1, 3)],),
    )
    original_offsets = torch.tensor([0, 2, 3])  # x→5, x→9, y→1
    node_dimension = t.dimensions[0].offsets_to_nodes(original_offsets)
    recovered_offsets = t.dimensions[0].nodes_to_offsets(node_dimension)
    assert torch.equal(recovered_offsets, original_offsets)


def test_duality_with_merged_node():
    # Register "a" in two separate extend calls so the merge path is exercised.
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni("a", 0, 3)],))
    t.extend(_dim(_ni("a", 7)), dim=0)
    # a: offsets 0,1,2 for logical 0,3,7
    offsets = t.dimensions[0].nodes_to_offsets(_dim(_ni("a", 0, 3, 7)))
    assert torch.equal(offsets, torch.tensor([0, 1, 2]))
    recovered = t.dimensions[0].offsets_to_nodes(offsets).node_infos
    assert len(recovered) == 1
    assert recovered[0].key == "a"
    assert torch.equal(recovered[0].indices, torch.tensor([[0], [3], [7]]))


# ---------------------------------------------------------------------------
# __getitem__
# ---------------------------------------------------------------------------


def test_getitem_first_node_exact_values():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    result = t[_dim(_ni("a", 0, 3))]
    assert torch.equal(result.data, torch.tensor([10.0, 20.0]))


def test_getitem_second_node_exact_values():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    result = t[_dim(_ni("b", 1, 2))]
    assert torch.equal(result.data, torch.tensor([30.0, 40.0]))


def test_getitem_reordered_cross_node_exact_values():
    # Ask for "b"→2 before "a"→0 to verify arbitrary ordering is respected
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    result = t[_dim(_ni("b", 2), _ni("a", 0))]
    assert torch.equal(result.data, torch.tensor([40.0, 10.0]))


def test_getitem_single_element_from_multi_element_node():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    # "a"→3 is the *second* element of "a" → data offset 1 → value 20
    result = t[_dim(_ni("a", 3))]
    assert torch.equal(result.data, torch.tensor([20.0]))


def test_getitem_none_returns_full_tensor():
    data = torch.tensor([1.0, 2.0, 3.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=([_ni("a", 0), _ni("b", 5), _ni("c", 2)],),
    )
    result = t[None]
    assert torch.equal(result.data, data)


def test_getitem_accepts_dimension_key_matches_node_info_list():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    key_dim = _dim(_ni("b", 2), _ni("a", 0))
    out_dim = t[key_dim]
    assert torch.equal(out_dim.data, torch.tensor([40.0, 10.0]))


def test_getitem_dimension_key_is_preserved_on_result_axis():
    """Sliced tensor keeps the selector :class:`NodeDimension` on the indexed axis."""
    data = torch.tensor([10.0, 20.0, 30.0, 40.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    key_dim = _dim(_ni("a", 0, 3))
    result = t[key_dim]
    assert result.dimensions[0] is key_dim


# ---------------------------------------------------------------------------
# __setitem__
# ---------------------------------------------------------------------------


def test_setitem_full_node_exact_values():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    t[_dim(_ni("a", 0, 3))] = torch.tensor([7.0, 8.0])
    assert t.data[0].item() == 7.0  # offset 0  →  "a"→0
    assert t.data[1].item() == 8.0  # offset 1  →  "a"→3
    assert t.data[2].item() == 0.0  # offset 2  →  "b"→1  (untouched)
    assert t.data[3].item() == 0.0  # offset 3  →  "b"→2  (untouched)


def test_setitem_partial_node_leaves_rest_unchanged():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    t[_dim(_ni("a", 3))] = torch.tensor([99.0])
    assert t.data[0].item() == 0.0  # "a"→0 (offset 0) untouched
    assert t.data[1].item() == 99.0  # "a"→3 (offset 1) written


def test_setitem_none_sets_all():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0), _ni("b", 5)],),
    )
    t[None] = torch.tensor([3.0, 7.0])
    assert torch.equal(t.data, torch.tensor([3.0, 7.0]))


def test_setitem_then_getitem_round_trip():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    selection = _dim(_ni("b", 1, 2))
    t[selection] = torch.tensor([11.0, 22.0])
    result = t[selection]
    assert torch.equal(result.data, torch.tensor([11.0, 22.0]))


def test_setitem_accepts_dimension_key():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=([_ni("a", 0, 3), _ni("b", 1, 2)],),
    )
    key_dim = _dim(_ni("a", 0, 3))
    t[key_dim] = torch.tensor([7.0, 8.0])
    assert t.data[0].item() == 7.0
    assert t.data[1].item() == 8.0
    assert t.data[2].item() == 0.0


# ---------------------------------------------------------------------------
# extend
# ---------------------------------------------------------------------------


def test_add_elements_expands_size():
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni("a", 0, 1)],))
    t.extend(_dim(_ni("b", 5, 6)), dim=0)
    assert t.data.shape == (4,)


def test_add_elements_new_node_gets_correct_offsets_and_indices():
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni("a", 0, 1)],))
    t.extend(_dim(_ni("b", 5, 6)), dim=0)
    assert torch.equal(t.dimensions[0].node_mappings["b"].offsets, torch.tensor([2, 3]))
    assert torch.equal(t.dimensions[0].node_mappings["b"].indices, torch.tensor([[5], [6]]))


def test_add_elements_existing_node_is_merged():
    t = NodeIndexedTensor.from_dimensions(dimensions=([_ni("a", 0)],))
    t.extend(_dim(_ni("a", 7)), dim=0)
    assert t.data.shape == (2,)
    assert torch.equal(t.dimensions[0].node_mappings["a"].offsets, torch.tensor([0, 1]))
    assert torch.equal(t.dimensions[0].node_mappings["a"].indices, torch.tensor([[0], [7]]))


def test_add_elements_preserves_existing_values():
    t = NodeIndexedTensor.from_data(
        data=torch.tensor([5.0, 6.0]),
        dimensions=([_ni("a", 0, 1)],),
    )
    t.extend(_dim(_ni("b", 9)), dim=0)
    assert torch.equal(t.data[:2], torch.tensor([5.0, 6.0]))
    assert t.data[2].item() == 0.0


def test_add_elements_newly_added_node_is_addressable():
    t = NodeIndexedTensor.from_data(
        data=torch.tensor([5.0, 6.0]),
        dimensions=([_ni("a", 0, 1)],),
    )
    selection = _dim(_ni("b", 9))
    t.extend(selection, dim=0)
    t[selection] = torch.tensor([42.0])
    result = t[selection]
    assert torch.equal(result.data, torch.tensor([42.0]))


# ---------------------------------------------------------------------------
# NodeIndexedMatrix (2-D)
# ---------------------------------------------------------------------------


def test_matrix_shape():
    m = NodeIndexedMatrix.from_dimensions(
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    assert m.data.shape == (2, 3)


def test_matrix_getitem_row_slice_exact_values():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    result = m[_dim(_ni("r", 0)), None]
    assert torch.equal(result.data, torch.tensor([[1.0, 2.0, 3.0]]))


def test_matrix_getitem_col_slice_exact_values():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    result = m[None, _dim(_ni("c", 1))]
    assert torch.equal(result.data, torch.tensor([[2.0], [5.0]]))


def test_matrix_getitem_submatrix_exact_values():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    # Row 1 ("r"→1), cols 0 and 2 ("c"→0, "c"→2) → [[4, 6]]
    result = m[_dim(_ni("r", 1)), _dim(_ni("c", 0, 2))]
    assert torch.equal(result.data, torch.tensor([[4.0, 6.0]]))


def test_matrix_getitem_none_none_returns_full_data():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1)]),
    )
    result = m[None, None]
    assert torch.equal(result.data, data)


def test_matrix_getitem_accepts_dimension_on_row_and_column_axes():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    row_dim = _dim(_ni("r", 1))
    col_dim = _dim(_ni("c", 0, 2))
    out = m[row_dim, col_dim]
    assert torch.equal(out.data, torch.tensor([[4.0, 6.0]]))
    assert out.dimensions[0] is row_dim
    assert out.dimensions[1] is col_dim


def test_matrix_setitem_accepts_dimension_keys():
    m = NodeIndexedMatrix.from_dimensions(
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    row_dim = _dim(_ni("r", 0))
    col_dim = _dim(_ni("c", 1, 2))
    m[row_dim, col_dim] = torch.tensor([[99.0, 88.0]])
    assert m.data[0, 1].item() == 99.0
    assert m.data[0, 2].item() == 88.0
    assert m.data[1, 0].item() == 0.0


def test_matrix_setitem_submatrix_exact_values():
    m = NodeIndexedMatrix.from_dimensions(
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    m[_dim(_ni("r", 0)), _dim(_ni("c", 1, 2))] = torch.tensor([[99.0, 88.0]])
    assert m.data[0, 0].item() == 0.0  # "c"→0 untouched
    assert m.data[0, 1].item() == 99.0  # "c"→1 written
    assert m.data[0, 2].item() == 88.0  # "c"→2 written
    assert m.data[1, 0].item() == 0.0  # row "r"→1 entirely untouched
    assert m.data[1, 1].item() == 0.0
    assert m.data[1, 2].item() == 0.0


def test_matrix_setitem_then_getitem_round_trip():
    m = NodeIndexedMatrix.from_dimensions(
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    row_selection = _dim(_ni("r", 1))
    col_selection = _dim(_ni("c", 0, 2))
    m[row_selection, col_selection] = torch.tensor([[11.0, 22.0]])
    result = m[row_selection, col_selection]
    assert torch.equal(result.data, torch.tensor([[11.0, 22.0]]))


def test_matrix_add_target_then_addressable():
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m = NodeIndexedMatrix.from_data(
        data=data,
        dimensions=([_ni("r", 0, 1)], [_ni("c", 0, 1, 2)]),
    )
    selection = _dim(_ni("r2", 10))
    m.add_targets(selection)
    m[selection, None] = torch.tensor([[7.0, 8.0, 9.0]])
    result = m[selection, None]
    assert torch.equal(result.data, torch.tensor([[7.0, 8.0, 9.0]]))


# ---------------------------------------------------------------------------
# NodeIndexedVector – topk exercises NodeDimension.offsets_to_nodes
# ---------------------------------------------------------------------------


def test_vector_topk_values_and_nodes_same_key():
    # offsets: p→5=0, p→7=1, p→9=2, q→1=3
    data = torch.tensor([3.0, 1.0, 4.0, 2.0])
    v = NodeIndexedVector.from_data(
        data=data,
        dimensions=([_ni("p", 5, 7, 9), _ni("q", 1)],),
    )
    values, node_infos = v.topk(k=2)
    node_infos = node_infos.node_infos
    # top-2: offset 2 (4.0 → p→9) then offset 0 (3.0 → p→5)
    assert torch.equal(values, torch.tensor([4.0, 3.0]))
    # both belong to "p" (consecutive), so they collapse into one NodeInfo
    assert len(node_infos) == 1
    assert node_infos[0].key == "p"
    assert torch.equal(node_infos[0].indices, torch.tensor([[9], [5]]))


def test_vector_topk_values_and_nodes_different_keys():
    # offsets: a→0=0, b→1=1, a→2=2
    data = torch.tensor([1.0, 4.0, 3.0])
    v = NodeIndexedVector.from_data(
        data=data,
        dimensions=([_ni("a", 0), _ni("b", 1), _ni("a", 2)],),
    )
    values, node_infos = v.topk(k=2)
    node_infos = node_infos.node_infos
    # top-2: offset 1 (4.0 → b→1) then offset 2 (3.0 → a→2)
    assert torch.equal(values, torch.tensor([4.0, 3.0]))
    assert len(node_infos) == 2
    assert node_infos[0].key == "b"
    assert torch.equal(node_infos[0].indices, torch.tensor([[1]]))
    assert node_infos[1].key == "a"
    assert torch.equal(node_infos[1].indices, torch.tensor([[2]]))


# ---------------------------------------------------------------------------
# d_index = 2 (2-D logical keys, e.g. (token_pos, feature_idx))
# ---------------------------------------------------------------------------
#
# Setup:
#   node "feat": 3 active (token_pos, feature_idx) pairs → (0,1), (2,3), (2,5)
#                  offsets 0, 1, 2
#   node "err":  2 pairs → (0,0), (1,0)
#                  offsets 3, 4
#
# inv_indices["feat"] has shape (3, 6):
#   [0,1]→0,  [2,3]→1,  [2,5]→2
# inv_indices["err"]  has shape (2, 1):
#   [0,0]→0,  [1,0]→1
# ---------------------------------------------------------------------------


def _make_2d_tensor() -> NodeIndexedTensor:
    return NodeIndexedTensor.from_dimensions(
        dimensions=(
            [
                _ni2("feat", (0, 1), (2, 3), (2, 5)),
                _ni2("err", (0, 0), (1, 0)),
            ],
        ),
    )


def test_2d_node_mappings_store_correct_indices():
    t = _make_2d_tensor()
    assert torch.equal(
        t.dimensions[0].node_mappings["feat"].indices,
        torch.tensor([[0, 1], [2, 3], [2, 5]]),
    )
    assert torch.equal(
        t.dimensions[0].node_mappings["err"].indices,
        torch.tensor([[0, 0], [1, 0]]),
    )


def test_2d_node_mappings_store_correct_offsets():
    t = _make_2d_tensor()
    assert torch.equal(t.dimensions[0].node_mappings["feat"].offsets, torch.tensor([0, 1, 2]))
    assert torch.equal(t.dimensions[0].node_mappings["err"].offsets, torch.tensor([3, 4]))


def test_2d_nodes_to_offsets_full_node():
    t = _make_2d_tensor()
    offsets = t.dimensions[0].nodes_to_offsets(_dim(_ni2("feat", (0, 1), (2, 3), (2, 5))))
    assert torch.equal(offsets, torch.tensor([0, 1, 2]))


def test_2d_nodes_to_offsets_partial_selection():
    t = _make_2d_tensor()
    # (2, 5) is the third element of "feat" → offset 2
    offsets = t.dimensions[0].nodes_to_offsets(_dim(_ni2("feat", (2, 5))))
    assert torch.equal(offsets, torch.tensor([2]))


def test_2d_nodes_to_offsets_cross_node():
    t = _make_2d_tensor()
    # feat→(2,3)=offset 1, err→(1,0)=offset 4
    offsets = t.dimensions[0].nodes_to_offsets(_dim(_ni2("feat", (2, 3)), _ni2("err", (1, 0))))
    assert torch.equal(offsets, torch.tensor([1, 4]))


def test_2d_offsets_to_nodes_full_feat_node():
    t = _make_2d_tensor()
    node_infos = t.dimensions[0].offsets_to_nodes(torch.tensor([0, 1, 2])).node_infos
    assert len(node_infos) == 1
    assert node_infos[0].key == "feat"
    assert torch.equal(node_infos[0].indices, torch.tensor([[0, 1], [2, 3], [2, 5]]))


def test_2d_offsets_to_nodes_partial_feat():
    t = _make_2d_tensor()
    # offset 2 → feat→(2,5)
    node_infos = t.dimensions[0].offsets_to_nodes(torch.tensor([2])).node_infos
    assert len(node_infos) == 1
    assert node_infos[0].key == "feat"
    assert torch.equal(node_infos[0].indices, torch.tensor([[2, 5]]))


def test_2d_offsets_to_nodes_full_err_node():
    t = _make_2d_tensor()
    node_infos = t.dimensions[0].offsets_to_nodes(torch.tensor([3, 4])).node_infos
    assert len(node_infos) == 1
    assert node_infos[0].key == "err"
    assert torch.equal(node_infos[0].indices, torch.tensor([[0, 0], [1, 0]]))


def test_2d_duality_nodes_to_offsets_to_nodes():
    t = _make_2d_tensor()
    original = _dim(_ni2("feat", (0, 1), (2, 5)), _ni2("err", (0, 0)))
    offsets = t.dimensions[0].nodes_to_offsets(original)
    recovered = t.dimensions[0].offsets_to_nodes(offsets).node_infos
    assert len(recovered) == 2
    assert recovered[0].key == "feat"
    assert torch.equal(recovered[0].indices, torch.tensor([[0, 1], [2, 5]]))
    assert recovered[1].key == "err"
    assert torch.equal(recovered[1].indices, torch.tensor([[0, 0]]))


def test_2d_duality_offsets_to_nodes_to_offsets():
    t = _make_2d_tensor()
    # feat→(0,1)=0, feat→(2,5)=2, err→(0,0)=3
    original_offsets = torch.tensor([0, 2, 3])
    node_dimension = t.dimensions[0].offsets_to_nodes(original_offsets)
    recovered = t.dimensions[0].nodes_to_offsets(node_dimension)
    assert torch.equal(recovered, original_offsets)


def test_2d_getitem_exact_values():
    data = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    t = NodeIndexedTensor.from_data(
        data=data,
        dimensions=(
            [
                _ni2("feat", (0, 1), (2, 3), (2, 5)),
                _ni2("err", (0, 0), (1, 0)),
            ],
        ),
    )
    # feat→(0,1)=offset 0=10, feat→(2,5)=offset 2=30
    result = t[_dim(_ni2("feat", (0, 1), (2, 5)))]
    assert torch.equal(result.data, torch.tensor([10.0, 30.0]))


def test_2d_setitem_exact_values():
    t = NodeIndexedTensor.from_dimensions(
        dimensions=(
            [
                _ni2("feat", (0, 1), (2, 3), (2, 5)),
                _ni2("err", (0, 0), (1, 0)),
            ],
        ),
    )
    t[_dim(_ni2("err", (0, 0), (1, 0)))] = torch.tensor([7.0, 8.0])
    assert t.data[0].item() == 0.0  # feat (untouched)
    assert t.data[3].item() == 7.0  # err→(0,0)
    assert t.data[4].item() == 8.0  # err→(1,0)


def test_2d_topk_exact_values_and_nodes():
    data = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    v = NodeIndexedVector.from_data(
        data=data,
        dimensions=(
            [
                _ni2("feat", (0, 1), (2, 3), (2, 5)),
                _ni2("err", (0, 0), (1, 0)),
            ],
        ),
    )
    values, node_infos = v.topk(k=3)
    node_infos = list(node_infos)
    # top-3: offset 1 (5.0→feat(2,3)), offset 3 (4.0→err(0,0)), offset 2 (3.0→feat(2,5))
    assert torch.equal(values, torch.tensor([5.0, 4.0, 3.0]))
    assert len(node_infos) == 3
    assert node_infos[0].key == "feat"
    assert torch.equal(node_infos[0].indices, torch.tensor([[2, 3]]))
    assert node_infos[1].key == "err"
    assert torch.equal(node_infos[1].indices, torch.tensor([[0, 0]]))
    assert node_infos[2].key == "feat"
    assert torch.equal(node_infos[2].indices, torch.tensor([[2, 5]]))


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
#
# Abbreviations used in comments below:
#   v  = NodeIndexedVector  (shape (n,))
#   M  = NodeIndexedMatrix  (shape (r, c))
# ---------------------------------------------------------------------------


def _make_vector(key: str, *positions: int, values: list[float]) -> NodeIndexedVector:
    return NodeIndexedVector.from_data(
        data=torch.tensor(values, dtype=torch.float32),
        dimensions=([_ni(key, *positions)],),
    )


def _make_matrix(
    row_key: str,
    row_positions: list[int],
    col_key: str,
    col_positions: list[int],
    data: list[list[float]],
) -> NodeIndexedMatrix:
    return NodeIndexedMatrix.from_data(
        data=torch.tensor(data, dtype=torch.float32),
        dimensions=([_ni(row_key, *row_positions)], [_ni(col_key, *col_positions)]),
    )


# ---- vector @ matrix → vector -------------------------------------------


def test_vector_matmul_matrix_data_exact():
    # v = [1, 2, 3], M = [[1,0],[0,1],[1,1]]
    # v @ M = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    v = _make_vector("a", 0, 1, 2, values=[1.0, 2.0, 3.0])
    M = _make_matrix("a", [0, 1, 2], "b", [0, 1], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = v.matmul(M)
    assert isinstance(result, NodeIndexedVector)
    assert torch.equal(result.data, torch.tensor([4.0, 5.0]))


def test_vector_matmul_matrix_inherits_col_node_infos():
    v = _make_vector("a", 0, 1, 2, values=[1.0, 2.0, 3.0])
    M = _make_matrix("a", [0, 1, 2], "b", [10, 20], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = v.matmul(M)
    assert result.dimensions[0] is M.dimensions[1]


def test_vector_matmul_matrix_operator():
    v = _make_vector("a", 0, 1, values=[3.0, 4.0])
    M = _make_matrix("a", [0, 1], "b", [0, 1], [[1.0, 2.0], [3.0, 4.0]])
    result = v @ M
    assert isinstance(result, NodeIndexedVector)
    # [3,4] @ [[1,2],[3,4]] = [3+12, 6+16] = [15, 22]
    assert torch.equal(result.data, torch.tensor([15.0, 22.0]))


# ---- vector @ vector → scalar -------------------------------------------


def test_vector_matmul_vector_dot_product():
    u = _make_vector("a", 0, 1, 2, values=[1.0, 2.0, 3.0])
    v = _make_vector("a", 0, 1, 2, values=[4.0, 5.0, 6.0])
    result = u.matmul(v)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert isinstance(result, float)
    assert result == pytest.approx(32.0)


def test_vector_matmul_vector_orthogonal():
    u = _make_vector("a", 0, 1, values=[1.0, 0.0])
    v = _make_vector("a", 0, 1, values=[0.0, 1.0])
    assert u.matmul(v) == pytest.approx(0.0)


# ---- matrix @ vector → vector -------------------------------------------


def test_matrix_matmul_vector_data_exact():
    # M = [[1,2,3],[4,5,6]], v = [1,0,1]
    # M @ v = [1+0+3, 4+0+6] = [4, 10]
    M = _make_matrix("r", [0, 1], "c", [0, 1, 2], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = _make_vector("c", 0, 1, 2, values=[1.0, 0.0, 1.0])
    result = M.matmul(v)
    assert isinstance(result, NodeIndexedVector)
    assert torch.equal(result.data, torch.tensor([4.0, 10.0]))


def test_matrix_matmul_vector_inherits_row_node_infos():
    M = _make_matrix("r", [5, 7], "c", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    v = _make_vector("c", 0, 1, values=[2.0, 3.0])
    result = M.matmul(v)
    assert result.dimensions[0] is M.dimensions[0]


def test_matrix_matmul_vector_operator():
    M = _make_matrix("r", [0, 1], "c", [0, 1], [[2.0, 0.0], [0.0, 3.0]])
    v = _make_vector("c", 0, 1, values=[4.0, 5.0])
    result = M @ v
    # [[2,0],[0,3]] @ [4,5] = [8, 15]
    assert isinstance(result, NodeIndexedVector)
    assert torch.equal(result.data, torch.tensor([8.0, 15.0]))


# ---- matrix @ matrix → matrix -------------------------------------------


def test_matrix_matmul_matrix_data_exact():
    # A (2×3) @ B (3×2) → C (2×2)
    # A = [[1,0,0],[0,1,0]], B = [[1,2],[3,4],[5,6]]
    # C = [[1,2],[3,4]]
    A = _make_matrix("r", [0, 1], "m", [0, 1, 2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    B = _make_matrix("m", [0, 1, 2], "c", [0, 1], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = A.matmul(B)
    assert isinstance(result, NodeIndexedMatrix)
    assert torch.equal(result.data, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_matrix_matmul_matrix_inherits_outer_node_infos():
    A = _make_matrix("r", [10, 20], "m", [0, 1, 2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    B = _make_matrix("m", [0, 1, 2], "c", [30, 40], [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    result = A @ B
    assert result.dimensions[0] is A.dimensions[0]  # row dimension from A
    assert result.dimensions[1] is B.dimensions[1]  # col dimension from B


def test_matrix_matmul_matrix_identity():
    I = _make_matrix("r", [0, 1], "c", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    M = _make_matrix("c", [0, 1], "d", [0, 1], [[3.0, 4.0], [5.0, 6.0]])
    result = I @ M
    assert torch.equal(result.data, M.data)


def test_matrix_matmul_matrix_operator():
    A = _make_matrix("r", [0, 1], "m", [0, 1], [[1.0, 2.0], [3.0, 4.0]])
    B = _make_matrix("m", [0, 1], "c", [0, 1], [[5.0, 6.0], [7.0, 8.0]])
    result = A @ B
    # [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert torch.equal(result.data, torch.tensor([[19.0, 22.0], [43.0, 50.0]]))


# ---- _check_node_matching -----------------------------------------------
#
# The matmul implementations accept ``_check_node_matching`` for API
# compatibility; node-key validation is not currently enforced.


def test_matmul_check_node_matching_passes_when_equal():
    v = _make_vector("a", 0, 1, values=[1.0, 2.0])
    M = _make_matrix("a", [0, 1], "b", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    result = v.matmul(M, _check_node_matching=True)
    assert isinstance(result, NodeIndexedVector)


def test_matmul_check_node_matching_does_not_raise_when_keys_differ():
    """Numeric matmul still runs when vector/matrix node keys disagree."""
    v = _make_vector("a", 0, 1, values=[1.0, 2.0])
    M = _make_matrix("x", [0, 1], "b", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    result = v.matmul(M, _check_node_matching=True)
    assert isinstance(result, NodeIndexedVector)
    assert torch.equal(result.data, torch.tensor([1.0, 2.0]))


def test_matrix_matmul_vector_check_node_matching_does_not_raise_when_keys_differ():
    M = _make_matrix("r", [0, 1], "c", [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    v = _make_vector("x", 0, 1, values=[1.0, 2.0])
    result = M.matmul(v, _check_node_matching=True)
    assert isinstance(result, NodeIndexedVector)
    assert torch.equal(result.data, torch.tensor([1.0, 2.0]))
