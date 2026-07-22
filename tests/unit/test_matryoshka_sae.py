import json
import types

import pytest
import safetensors.torch as safe
import torch

from llamascopium import (
    CircuitConfig,
    CircuitFeatureTarget,
    Initializer,
    InitializerConfig,
    MatryoshkaSAEConfig,
    MatryoshkaSparseAutoEncoder,
    Trainer,
    TrainerConfig,
    TrainSAESettings,
)
from llamascopium.circuits.attribution import _validate_active_matryoshka_nodes
from llamascopium.models.sparse_dictionary import SparseDictionary


def _build_range_test_sae() -> MatryoshkaSparseAutoEncoder:
    cfg = MatryoshkaSAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        norm_activation="inference",
        use_decoder_bias=False,
        act_fn="relu",
        sparsity_include_decoder_norm=False,
        matryoshka_widths=[2, 4],
    )
    sae = MatryoshkaSparseAutoEncoder(cfg)
    with torch.no_grad():
        sae.W_E.fill_(1.0)
        sae.b_E.zero_()
    return sae


@pytest.mark.parametrize(
    ("feature_range", "expected"),
    [
        ((0, 2), [2.0, 2.0, 0.0, 0.0]),
        ((2, 4), [0.0, 0.0, 2.0, 2.0]),
        ((0, 4), [2.0, 2.0, 2.0, 2.0]),
    ],
)
def test_matryoshka_feature_range_masks_activations_before_hook(feature_range, expected):
    sae = _build_range_test_sae()
    hooked: list[torch.Tensor] = []
    sae.hook_feature_acts.add_hook(lambda tensor, hook: hooked.append(tensor.detach().clone()) or tensor)

    actual = sae.encode(torch.ones(1, 2), matryoshka_feature_range=feature_range)

    expected_tensor = torch.tensor([expected])
    assert torch.equal(actual, expected_tensor)
    assert len(hooked) == 1
    assert torch.equal(hooked[0], expected_tensor)


def test_matryoshka_feature_range_rejects_unconfigured_boundary():
    sae = _build_range_test_sae()

    with pytest.raises(ValueError, match="boundaries must come from"):
        sae.encode(torch.ones(1, 2), matryoshka_feature_range=(1, 4))


def test_matryoshka_feature_range_supports_combined_intermediate_segments():
    cfg = MatryoshkaSAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        norm_activation="inference",
        use_decoder_bias=False,
        act_fn="relu",
        sparsity_include_decoder_norm=False,
        matryoshka_widths=[1, 2, 4],
    )
    sae = MatryoshkaSparseAutoEncoder(cfg)
    with torch.no_grad():
        sae.W_E.fill_(1.0)
        sae.b_E.zero_()

    actual = sae.encode(torch.ones(1, 2), matryoshka_feature_range=(1, 4))

    assert torch.equal(actual, torch.tensor([[0.0, 2.0, 2.0, 2.0]]))


def test_circuit_accepts_active_matryoshka_feature_at_final_position():
    sae = _build_range_test_sae()
    key = "out.sae.hook_feature_acts"

    result = _validate_active_matryoshka_nodes(
        [sae],
        {key: torch.tensor([[0, 1], [1, 2]])},
        final_position=1,
        feature_range=(2, 4),
    )

    assert result is None


def test_circuit_rejects_segment_without_final_position_activation():
    sae = _build_range_test_sae()
    key = "out.sae.hook_feature_acts"

    with pytest.raises(ValueError, match=r"no active features.*final token.*\[2, 4\)"):
        _validate_active_matryoshka_nodes(
            [sae],
            {key: torch.tensor([[0, 2]])},
            final_position=1,
            feature_range=(2, 4),
        )


def test_circuit_config_supports_named_targets_and_legacy_targets():
    config = CircuitConfig.model_validate(
        {
            "list_of_features": [
                {"sae_name": "matry-layer27", "feature_index": 7000, "position": 3},
                [27, 12, 3, False],
            ],
            "matryoshka_feature_range": [6144, 12288],
        }
    )

    assert isinstance(config.list_of_features[0], CircuitFeatureTarget)
    assert config.list_of_features[1] == (27, 12, 3, False)
    assert config.matryoshka_feature_range == (6144, 12288)


def test_matryoshka_config_normalizes_widths_and_weights():
    cfg = MatryoshkaSAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        norm_activation="inference",
        matryoshka_widths=[2],
        matryoshka_loss_weights=[0.5],
    )

    assert cfg.matryoshka_widths == [2, 4]
    assert cfg.matryoshka_loss_weights is not None
    assert torch.allclose(torch.tensor(cfg.matryoshka_loss_weights), torch.tensor([1.0 / 3.0, 2.0 / 3.0]))


def test_matryoshka_config_defaults_to_equal_normalized_weights():
    cfg = MatryoshkaSAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        norm_activation="inference",
        matryoshka_widths=[2],
    )

    assert cfg.matryoshka_widths == [2, 4]
    assert cfg.matryoshka_loss_weights is not None
    assert torch.allclose(torch.tensor(cfg.matryoshka_loss_weights), torch.tensor([0.5, 0.5]))


def test_matryoshka_compute_loss_adds_prefix_losses():
    cfg = MatryoshkaSAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        norm_activation="inference",
        use_decoder_bias=False,
        act_fn="relu",
        sparsity_include_decoder_norm=False,
        matryoshka_widths=[2, 4],
        matryoshka_loss_weights=[0.5, 2.0],
    )
    sae = MatryoshkaSparseAutoEncoder(cfg)

    with torch.no_grad():
        sae.W_E.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ]
            )
        )
        sae.b_E.zero_()
        sae.W_D.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )

    batch = {
        "in": torch.tensor([[1.0, 1.0]]),
        "out": torch.tensor([[2.0, 2.0]]),
        "tokens": torch.tensor([0]),
    }
    ctx = sae.compute_loss(batch, return_aux_data=True)

    assert torch.allclose(ctx["l_rec"], torch.tensor(0.0))
    assert torch.allclose(ctx["matryoshka_inner_losses"]["width_2"], torch.tensor(2.0))
    assert torch.allclose(ctx["l_matryoshka"], torch.tensor(0.4))
    assert torch.allclose(ctx["loss"], torch.tensor(0.4))


def test_matryoshka_auxk_compat_flag_uses_standard_global_aux_loss():
    def build_model(use_matryoshka_aux_loss: bool) -> MatryoshkaSparseAutoEncoder:
        cfg = MatryoshkaSAEConfig(
            hook_point_in="in",
            hook_point_out="out",
            d_model=2,
            expansion_factor=2,
            device="cpu",
            dtype=torch.float32,
            norm_activation="inference",
            use_decoder_bias=False,
            act_fn="relu",
            sparsity_include_decoder_norm=False,
            matryoshka_widths=[2, 4],
            use_matryoshka_aux_loss=use_matryoshka_aux_loss,
        )
        sae = MatryoshkaSparseAutoEncoder(cfg)
        with torch.no_grad():
            sae.W_D.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 0.0],
                        [0.5, 0.0],
                        [0.0, 0.0],
                    ]
                )
            )

        def fake_encode(self, x, return_hidden_pre=False, **kwargs):
            feature_acts = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=x.dtype, device=x.device)
            hidden_pre = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=x.dtype, device=x.device)
            if return_hidden_pre:
                return feature_acts, hidden_pre
            return feature_acts

        def fake_decode(self, feature_acts, **kwargs):
            return feature_acts @ self.W_D

        sae.encode = types.MethodType(fake_encode, sae)
        sae.decode = types.MethodType(fake_decode, sae)
        return sae

    batch = {
        "in": torch.tensor([[0.0, 0.0]]),
        "out": torch.tensor([[2.0, 0.0]]),
        "tokens": torch.tensor([0]),
    }
    dead_mask = torch.tensor([False, False, True, False])

    for use_matryoshka_aux_loss in [False, True]:
        sae = build_model(use_matryoshka_aux_loss=use_matryoshka_aux_loss)
        ctx = sae.compute_loss(
            batch,
            auxk_coefficient=1.0,
            k_aux=4,
            update_dead_statistics=lambda feature_acts, mask, specs: dead_mask,
            return_aux_data=True,
        )

        assert torch.allclose(ctx["l_aux"], torch.tensor(0.25))
        assert torch.allclose(ctx["loss"], torch.tensor(1.25))


def test_from_local_loads_matryoshka_sae(tmp_path):
    cfg = MatryoshkaSAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        norm_activation="inference",
        use_decoder_bias=False,
        act_fn="relu",
        sparsity_include_decoder_norm=False,
        matryoshka_widths=[2, 4],
    )
    sae = MatryoshkaSparseAutoEncoder(cfg)
    with torch.no_grad():
        sae.W_E.fill_(1.0)
        sae.b_E.fill_(2.0)
        sae.W_D.fill_(3.0)

    with open(tmp_path / "config.json", "w") as f:
        json.dump(cfg.model_dump(), f)
    safe.save_file(sae.state_dict(), tmp_path / "sae_weights.safetensors")

    loaded = SparseDictionary.from_local(str(tmp_path))
    assert isinstance(loaded, MatryoshkaSparseAutoEncoder)
    assert loaded.cfg.matryoshka_widths == [2, 4]
    assert torch.allclose(loaded.W_E, torch.ones_like(loaded.W_E))
    assert torch.allclose(loaded.b_E, torch.full_like(loaded.b_E, 2.0))
    assert torch.allclose(loaded.W_D, torch.full_like(loaded.W_D, 3.0))


def test_train_settings_parse_matryoshka_config(tmp_path):
    settings = TrainSAESettings.model_validate(
        {
            "sae": {
                "sae_type": "matryoshka_sae",
                "hook_point_in": "in",
                "hook_point_out": "out",
                "d_model": 2,
                "expansion_factor": 2,
                "device": "cpu",
                "dtype": "float32",
                "norm_activation": "inference",
                "matryoshka_widths": [2, 4],
            },
            "sae_name": "test",
            "sae_series": "unit",
            "trainer": {"exp_result_path": str(tmp_path)},
            "activation_factory": {
                "sources": [
                    {
                        "type": "activations",
                        "name": "acts",
                        "path": str(tmp_path),
                        "device": "cpu",
                    }
                ],
                "target": "activations-1d",
                "hook_points": ["in"],
                "batch_size": 1,
            },
        }
    )

    assert isinstance(settings.sae, MatryoshkaSAEConfig)


def test_matryoshka_trainer_fit_single_device(tmp_path):
    cfg = MatryoshkaSAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        norm_activation="inference",
        act_fn="relu",
        use_decoder_bias=False,
        sparsity_include_decoder_norm=False,
        matryoshka_widths=[2, 4],
    )
    activation_stream = [
        {
            "in": torch.randn(4, 2),
            "out": torch.randn(4, 2),
            "tokens": torch.arange(4),
            "mask": torch.ones(4, dtype=torch.bool),
        }
        for _ in range(8)
    ]

    initializer = Initializer(InitializerConfig())
    sae = initializer.initialize_sae_from_config(cfg, activation_stream=activation_stream)
    trainer = Trainer(
        TrainerConfig(
            total_training_tokens=16,
            log_frequency=100,
            eval_frequency=100,
            n_checkpoints=0,
            exp_result_path=str(tmp_path),
            amp_dtype=torch.bfloat16,
            lr_scheduler_name="constant",
            l1_coefficient=None,
        )
    )

    trainer.fit(sae=sae, activation_stream=activation_stream, eval_fn=None, wandb_logger=None)

    assert trainer.cur_step == 4


def test_matryoshka_segment_metrics_are_non_cumulative():
    cfg = MatryoshkaSAEConfig(
        hook_point_in="in",
        hook_point_out="out",
        d_model=2,
        expansion_factor=2,
        device="cpu",
        dtype=torch.float32,
        norm_activation="inference",
        matryoshka_widths=[2, 4],
    )
    sae = MatryoshkaSparseAutoEncoder(cfg)

    feature_acts = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [3.0, 4.0, 0.0, 5.0],
        ]
    )
    metrics = sae.compute_training_metrics(
        feature_acts=feature_acts,
        n_tokens=2,
        matryoshka_inner_losses={"width_2": torch.tensor(1.5)},
        l_matryoshka=torch.tensor(1.5),
    )

    assert metrics["matryoshka_metrics/width_2"] == 1.5
    assert metrics["matryoshka_metrics/segment_0_2_mean_frequency"] == 0.75
    assert metrics["matryoshka_metrics/segment_2_4_mean_frequency"] == 0.5
    assert torch.isclose(torch.tensor(metrics["matryoshka_metrics/segment_0_2_mean_feature_act"]), torch.tensor(8 / 3))
    assert torch.isclose(torch.tensor(metrics["matryoshka_metrics/segment_2_4_mean_feature_act"]), torch.tensor(3.5))
