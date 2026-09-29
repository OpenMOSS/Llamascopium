from types import SimpleNamespace

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from llamascopium.backend.language_model import TransformerLensLanguageModel
from llamascopium.circuits.global_weights import FeatureSpec, GlobalWeights, compute_global_weights
from llamascopium.circuits.global_weights.weights import _feature_specs
from llamascopium.models.lorsa import LorsaConfig, LowRankSparseAttention
from llamascopium.models.sae import SAEConfig, SparseAutoEncoder


def spec(name, order, encoder, decoder, *, lorsa=False, center=False):
    return FeatureSpec(
        name=name,
        order=order,
        encoder=torch.tensor(encoder, dtype=torch.float32),
        decoder=torch.tensor(decoder, dtype=torch.float32),
        scale_hook=f"{name}.scale",
        is_lorsa=lorsa,
        center_decoder=center,
    )


def test_global_weights_accumulate_across_prompts_and_exclude_later_features():
    source = spec("source", (0, 1), [[1, 1, 1], [0, 0, 0]], [[2, 0], [1, 0], [3, 0]])
    target = spec("target", (1, 0), [[1], [0]], [[1, 0]])
    later = spec("later", (1, 1), [[1], [0]], [[1, 0]])
    weights = GlobalWeights([source, target, later], [("target", 0)])
    weights.update(
        {
            "source": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            "target": torch.tensor([[3.0], [4.0]]),
            "later": torch.ones(2, 1),
        },
        {"target.scale": torch.tensor([1.0, 2.0]), "later.scale": torch.ones(2)},
    )
    weights.update(
        {"source": torch.tensor([[1.0, 1.0, 0.0]]), "target": torch.tensor([[2.0]]), "later": torch.ones(1, 1)},
        {"target.scale": torch.tensor([1.0]), "later.scale": torch.ones(1)},
    )
    edges = weights.topk(("target", 0))
    assert [(edge.source, edge.source_feature) for edge in edges] == [("source", 0), ("source", 1)]
    assert [edge.weight for edge in edges] == pytest.approx([5.0, 2.0])
    assert weights.num_samples == 2


def test_lorsa_target_transports_activation_through_attention_pattern():
    source = spec("source", (0, 0), [[1], [0]], [[2, 0]])
    target = spec("target", (0, 1), [[1], [0]], [[1, 0]], lorsa=True)
    weights = GlobalWeights([source, target], [("target", 0)])
    weights.update(
        {"source": torch.tensor([[1.0], [3.0]]), "target": torch.tensor([[2.0], [4.0]])},
        {"target.scale": torch.tensor([1.0, 2.0])},
        {"target": torch.tensor([[[1.0, 0.0], [0.25, 0.75]]])},
    )
    # Transported target activity is [3, 3], then divided by [1, 2].
    assert weights.topk(("target", 0))[0].weight == pytest.approx(3.75)

    outgoing = GlobalWeights([source, target], [("source", 0)])
    outgoing.update(
        {"source": torch.tensor([[1.0], [3.0]]), "target": torch.tensor([[2.0], [4.0]])},
        {"target.scale": torch.tensor([1.0, 2.0])},
        {"target": torch.tensor([[[1.0, 0.0], [0.25, 0.75]]])},
    )
    edge = outgoing.downstream(("source", 0))[0]
    assert (edge.source, edge.target, edge.target_feature) == ("source", "target", 0)
    assert edge.weight == pytest.approx(2.5)


def test_ln_centers_source_decoder_before_virtual_weight():
    source = spec("source", (0, 0), [[1], [0]], [[2, 0]])
    target = spec("target", (0, 1), [[1], [0]], [[1, 0]], center=True)
    weights = GlobalWeights([source, target], [("target", 0)])
    weights.update(
        {"source": torch.tensor([[2.0]]), "target": torch.tensor([[3.0]])},
        {"target.scale": torch.tensor([1.0])},
    )
    assert weights.topk(("target", 0))[0].weight == pytest.approx(3.0)


@pytest.mark.parametrize("normalization,expected_gain", [("LN", 2.0), ("RMS", 1.0)])
def test_virtual_weight_uses_gain_only_when_hook_is_after_gain(normalization, expected_gain):
    sae = SparseAutoEncoder(
        SAEConfig(
            d_model=2,
            expansion_factor=1,
            hook_point_in="blocks.0.ln2.hook_normalized",
            hook_point_out="blocks.0.hook_mlp_out",
            device="cpu",
            dtype=torch.float32,
        )
    )
    with torch.no_grad():
        sae.W_E.fill_(1)
    model = SimpleNamespace(
        device_mesh=None,
        cfg=SimpleNamespace(parallel_attn_mlp=False, normalization_type=normalization),
        blocks=[SimpleNamespace(ln2=SimpleNamespace(w=torch.tensor([2.0, 3.0])))],
    )
    feature = _feature_specs(model, [sae])[0]
    assert feature.encoder[:, 0].tolist() == pytest.approx([expected_gain, 3.0 if normalization == "LN" else 1.0])
    assert feature.center_decoder == (normalization == "LN")


def test_real_model_hooks_capture_sae_and_lorsa_features():
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
        lorsa.b_V.fill_(10)

    target = ("blocks.1.hook_attn_out", 0)
    result = compute_global_weights(model, [sae, lorsa], [torch.tensor([[1, 2, 3]])], [target])
    assert result.num_samples == 1
    assert len(result.topk(target)) == sae.cfg.d_sae
    assert all(edge.source == "blocks.0.hook_mlp_out" for edge in result.topk(target))
