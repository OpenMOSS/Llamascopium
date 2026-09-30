import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from llamascopium.backend.language_model import TransformerLensLanguageModel
from llamascopium.circuits.global_weights import (
    FeatureSpec,
    InhibitoryAtlas,
    InhibitoryConnection,
    InhibitoryWeights,
    compute_inhibitory_global_weights,
    compute_inhibitory_weights,
    search_inhibitory_atlas,
)
from llamascopium.models.lorsa import LorsaConfig, LowRankSparseAttention
from llamascopium.models.sae import SAEConfig, SparseAutoEncoder


def _spec(name, order, encoder, decoder, *, lorsa=False):
    return FeatureSpec(
        name=name,
        order=order,
        encoder=torch.tensor(encoder, dtype=torch.float32),
        decoder=torch.tensor(decoder, dtype=torch.float32),
        scale_hook=f"{name}.scale",
        is_lorsa=lorsa,
    )


def test_sparse_complement_conditional_means_and_structural_sign():
    source = _spec(
        "source",
        (0, 1),
        [[1] * 6, [0] * 6],
        [[-2, 0], [2, 0], [-1, 0], [-1, 0], [-1, 0], [3, 0]],
    )
    target = _spec("target", (1, 0), [[1], [0]], [[1, 0]])
    result = InhibitoryWeights([source, target], [("target", 0)])
    dense = torch.tensor(
        [
            [0, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 1],
        ],
        dtype=torch.float32,
    )
    pre = torch.tensor([4.0, 1.0, 5.0, 0.0])
    result.update({"source": dense.to_sparse()}, {("target", 0): pre}, {"target.scale": torch.ones(4)})
    stats = result.statistics(("target", 0), "source")
    for feature in range(dense.shape[1]):
        active = dense[:, feature] > 0
        explicit_on = pre[active].sum()
        explicit_off = pre[~active].sum()
        assert result.s_on[("target", 0)]["source"][feature] == explicit_on
        assert result.s_total[("target", 0)] - result.s_on[("target", 0)]["source"][feature] == explicit_off
        if active.any() and (~active).any():
            assert stats.mu_on[feature] == pytest.approx(pre[active].mean().item())
            assert stats.mu_off[feature] == pytest.approx(pre[~active].mean().item())
        else:
            assert stats.mu_on[feature] == 0
            assert stats.mu_off[feature] == 0
    assert stats.delta[0] == pytest.approx(4.0)
    assert stats.virtual_weight[0] == pytest.approx(-2.0)
    assert stats.inhibitory_score[0] == pytest.approx(8.0)
    assert stats.inhibitory_score[1] == 0  # Positive virtual weight.
    assert stats.inhibitory_score[4] == 0  # Source active when drive is higher.
    assert stats.inhibitory_score[2] == stats.inhibitory_score[3] == 0  # Empty comparison group.
    assert torch.isfinite(stats.normalized_inhibitory_score).all()
    assert result.topk(("target", 0))[0].source_feature == 0
    assert result.topk(("target", 0))[0].kind == "inhibitory"


def test_on_mean_is_weighted_by_source_activation():
    source = _spec("source", (0, 1), [[1], [0]], [[-2, 0]])
    target = _spec("target", (1, 0), [[1], [0]], [[1, 0]])
    result = InhibitoryWeights([source, target], [("target", 0)])
    result.update(
        {"source": torch.tensor([[0.0], [1.0], [3.0]]).to_sparse()},
        {("target", 0): torch.tensor([10.0, 2.0, 0.0])},
        {"target.scale": torch.ones(3)},
    )
    stats = result.statistics(("target", 0), "source")
    assert stats.mu_on[0] == pytest.approx((1 * 2 + 3 * 0) / (1 + 3))
    assert stats.mu_off[0] == pytest.approx(10.0)
    assert stats.delta[0] == pytest.approx(9.5)
    assert stats.inhibitory_score[0] == pytest.approx(19.0)


def test_search_keeps_positive_lorsa_candidate_beyond_global_top_k():
    transcoder = _spec("transcoder", (0, 1), [[1], [0]], [[-2, 0]])
    lorsa = _spec("lorsa", (1, 0), [[1], [0]], [[-1, 0]], lorsa=True)
    target = _spec("target", (2, 0), [[1], [0]], [[1, 0]])
    result = InhibitoryWeights([transcoder, lorsa, target], [("target", 0)])
    result.update(
        {"transcoder": torch.tensor([[0.0], [1.0]]).to_sparse(),
         "lorsa": torch.tensor([[0.0], [1.0]]).to_sparse()},
        {("target", 0): torch.tensor([10.0, 0.0])},
        {"target.scale": torch.ones(2)},
    )
    assert [edge.source for edge in result.topk(("target", 0), k=1)] == ["transcoder"]
    assert {edge.source for edge in result.topk(("target", 0), k=1, lorsa_k=1)} == {"transcoder", "lorsa"}


def test_lorsa_and_transcoder_routing_use_pre_topk_values():
    source = _spec("source", (0, 0), [[1], [0]], [[-1, 0]])
    lorsa = _spec("lorsa", (1, 0), [[1], [0]], [[1, 0]], lorsa=True)
    transcoder = _spec("transcoder", (1, 1), [[1], [0]], [[1, 0]])
    result = InhibitoryWeights([source, lorsa, transcoder], [("lorsa", 0), ("transcoder", 0)])
    pre = torch.tensor([-1.0, 2.0])
    pattern = torch.tensor([[[1.0, 0.0], [0.25, 0.75]]])
    result.update(
        {
            "source": torch.tensor([[1.0], [0.0]]).to_sparse(),
            "lorsa": torch.zeros(2, 1).to_sparse(),
        },
        {("lorsa", 0): pre, ("transcoder", 0): pre},
        {"lorsa.scale": torch.tensor([2.0, 3.0]), "transcoder.scale": torch.tensor([2.0, 3.0])},
        {"lorsa": pattern},
    )
    explicit = torch.tensor([sum(pre[q] * pattern[0, q, k] for q in range(2)) for k in range(2)])
    routed = explicit / torch.tensor([2.0, 3.0])
    assert explicit.tolist() == pytest.approx([-0.5, 1.5])
    assert result.s_total[("lorsa", 0)] == pytest.approx(routed.sum().item())
    assert result.s_on[("lorsa", 0)]["source"][0] == pytest.approx(routed[0].item())
    assert result.s_total[("transcoder", 0)] == pytest.approx((pre / torch.tensor([2.0, 3.0])).sum().item())
    assert result.s_on[("transcoder", 0)]["source"][0] == pytest.approx((pre[0] / 2).item())


def test_inhibitory_statistics_pool_positions_across_prompts():
    source = _spec("source", (0, 1), [[1], [0]], [[-1, 0]])
    target = _spec("target", (1, 0), [[1], [0]], [[1, 0]])
    result = InhibitoryWeights([source, target], [("target", 0)])
    result.update(
        {"source": torch.tensor([[0.0], [1.0]]).to_sparse()},
        {("target", 0): torch.tensor([4.0, 0.0])},
        {"target.scale": torch.ones(2)},
    )
    result.update(
        {"source": torch.tensor([[1.0]]).to_sparse()},
        {("target", 0): torch.tensor([2.0])},
        {"target.scale": torch.ones(1)},
    )
    stats = result.statistics(("target", 0), "source")
    assert result.n_total == 3
    assert result.n_on["source"][0] == 2
    assert stats.mu_on[0] == pytest.approx(1.0)
    assert stats.mu_off[0] == pytest.approx(4.0)
    assert stats.delta[0] == pytest.approx(3.0)
    assert stats.normalized_inhibitory_score[0] == pytest.approx(1.5)


def test_inhibitory_atlas_uses_positive_priority_and_causal_edge_direction():
    edge = InhibitoryConnection(
        source="source",
        source_feature=2,
        target="target",
        target_feature=1,
        score=3.0,
        normalized_inhibitory_score=1.0,
        delta=2.0,
        virtual_weight=-1.5,
        mu_on=0.0,
        mu_off=2.0,
        kind="inhibitory",
    )
    atlas = InhibitoryAtlas(("target", 1))
    atlas.update(("target", 1), [edge])
    assert atlas.select_top_k_nodes_to_visit(1) == [("source", 2)]
    exported = atlas.export_to_dict()["links"][0]
    assert exported["source"] == ("source", 2)
    assert exported["target"] == ("target", 1)
    assert exported["kind"] == "inhibitory"


def test_search_expands_selected_upstream_feature():
    seed = ("blocks.2.hook_attn_out", 0)
    middle = ("blocks.1.hook_mlp_out", 1)
    lorsa = ("blocks.1.hook_attn_out", 2)

    def edge(source, target, score):
        return InhibitoryConnection(source[0], source[1], target[0], target[1], score, score, 1.0, -score, 0.0, 1.0)

    calls = []

    def compute(targets):
        calls.append(targets)
        edges = {seed: [edge(middle, seed, 2.0)], middle: [edge(lorsa, middle, 1.0)]}
        return type("Result", (), {
            "topk": lambda self, target, k, normalized, lorsa_k: edges[target][:k],
            "num_samples": 1,
        })()

    atlas, result = search_inhibitory_atlas(seed, compute, depth=2, top_k=10, expansion_width=1)
    assert result.num_samples == 1
    assert calls == [[seed], [middle]]
    assert (lorsa, middle) in atlas.links


def test_real_model_captures_selected_pre_topk_target_and_sparse_source():
    assert compute_inhibitory_weights is compute_inhibitory_global_weights
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=8,
        n_ctx=8,
        d_head=4,
        n_heads=2,
        d_mlp=16,
        d_vocab=32,
        act_fn="relu",
        normalization_type="LN",
        device="cpu",
    )
    model = TransformerLensLanguageModel.from_hooked_transformer(HookedTransformer(cfg))
    sae = SparseAutoEncoder(
        SAEConfig(
            d_model=8,
            expansion_factor=1,
            hook_point_in="blocks.0.ln2.hook_normalized",
            hook_point_out="blocks.0.hook_mlp_out",
            device="cpu",
            dtype=torch.float32,
        )
    )
    sae.init_parameters(encoder_uniform_bound=0.1, decoder_uniform_bound=0.1)
    lorsa = LowRankSparseAttention(
        LorsaConfig(
            d_model=8,
            expansion_factor=1,
            hook_point_in="blocks.1.ln1.hook_normalized",
            hook_point_out="blocks.1.hook_attn_out",
            n_qk_heads=2,
            d_qk_head=4,
            rotary_dim=4,
            positional_embedding_type="none",
            n_ctx=8,
            device="cpu",
            dtype=torch.float32,
        )
    )
    lorsa.init_parameters()
    with torch.no_grad():
        sae.b_E.fill_(10)
    target = ("blocks.1.hook_attn_out", 0)
    result = compute_inhibitory_weights(model, [sae, lorsa], [torch.tensor([[1, 2, 3]])], [target])
    assert result.num_samples == 1
    assert result.n_total == 3
    assert result.n_on["blocks.0.hook_mlp_out"].shape == (sae.cfg.d_sae,)
    assert torch.isfinite(result.statistics(target, "blocks.0.hook_mlp_out").inhibitory_score).all()
